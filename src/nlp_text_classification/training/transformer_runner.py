"""Full encoder fine-tuning with S1 checkpoint and validation evidence."""

from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.transformer_inputs import encode_partition, input_fingerprint
from nlp_text_classification.evaluation.batch_metrics import (
    measure,
    read_predictions,
    select_threshold,
    write_predictions,
)
from nlp_text_classification.experiments.batch_b_resources import label_names, load_encoder
from nlp_text_classification.training.progress import write_atomic_json


class DevicePartition:
    def __init__(self, partition, device):
        self.features = {key: torch.as_tensor(values, device=device)
                         for key, values in partition.features.items()}
        self.lengths = partition.lengths
        self.targets = torch.as_tensor(partition.targets, device=device)

    def batch(self, indices, multiple):
        width = math.ceil(int(self.lengths[indices].max()) / multiple) * multiple
        index = torch.as_tensor(indices, device=self.targets.device)
        return ({key: values.index_select(0, index)[:, :width].contiguous()
                 for key, values in self.features.items()}, self.targets.index_select(0, index))


def prepare_partitions(tokenizer, data, audit, *, smoke=False):
    splits = {"train": data.train, "validation": data.validation}
    if smoke:
        # Drive the training micro-batches with long rows first in the sample inventory.
        longest = sorted(data.train, key=lambda row: len(row.text), reverse=True)[:64]
        spread = [data.train[i] for i in np.linspace(0, len(data.train) - 1, 32, dtype=int)]
        unique = {row.sample_id: row for row in (*longest, *spread)}
        splits = {"train": list(unique.values()), "validation": [
            data.validation[i] for i in np.linspace(0, len(data.validation) - 1, 96, dtype=int)
        ]}
    encoded = {key: encode_partition(tokenizer, data, rows, audit["selected_length"],
                                     audit["padding_multiple"]) for key, rows in splits.items()}
    if not smoke:
        for key, part in encoded.items():
            if input_fingerprint(part) != audit["partitions"][key]["encoded_sha256"]:
                raise ValueError(f"Encoded {key} differs from the tokenization audit")
    return splits, encoded


def classification_loss(logits, targets, binary):
    logits = logits.float()
    if binary:
        return torch.nn.functional.cross_entropy(logits, targets[:, 0].long())
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float())


def _forward_scores(logits, binary):
    return logits.float().softmax(dim=1)[:, 1:2] if binary else logits.float().sigmoid()


def threshold_record(targets, scores, binary, selection):
    result = select_threshold(targets, scores, binary, selection, "sigmoid")
    result["score_kind"] = "softmax_positive" if binary else "sigmoid"
    result["threshold_grid"] = "sigmoid"
    return result


def verify_prediction_file(path, rows, targets, binary, cutoff, *, auc):
    # Both normalized score types use the same float32 replay representation.
    ids, actual_targets, scores, predicted = read_predictions(path, score_kind="sigmoid")
    if ids != [row.sample_id for row in rows] or not np.array_equal(actual_targets, targets):
        raise ValueError(f"Prediction sample IDs or targets differ: {path}")
    if not np.array_equal(predicted, scores >= cutoff):
        raise ValueError(f"Prediction threshold differs: {path}")
    return measure(actual_targets, scores, cutoff, binary, auc=auc), scores


def _optimizer(model, settings, device):
    decay, unregularized = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else unregularized).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings["weight_decay"]},
         {"params": unregularized, "weight_decay": 0.0}],
        lr=settings["learning_rate"], fused=device.type == "cuda",
    )


def _atomic_checkpoint(path, data):
    temp = path.with_suffix(".pending")
    torch.save(data, temp)
    temp.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _train_epoch(model, partition, device, settings, optimizer, scheduler, generator,
                 *, binary, multiple, tick):
    model.train()
    device_kind, bf16 = device.type, settings["precision"] == "bf16"
    effective, micro = settings["effective_batch_size"], settings["micro_batch_size"]
    count = len(partition.targets)
    permutation = torch.randperm(count, generator=generator).numpy()
    loss_sum = torch.zeros((), device=device)
    max_gradient = torch.zeros((), device=device)
    steps = math.ceil(count / effective)
    for step, offset in enumerate(range(0, count, effective), start=1):
        block = permutation[offset:offset + effective]
        block = block[np.argsort(partition.lengths[block], kind="stable")]
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(block), micro):
            part = block[start:start + micro]
            inputs, target = partition.batch(part, multiple)
            with torch.autocast(device_kind, dtype=torch.bfloat16, enabled=bf16):
                loss = classification_loss(model(**inputs).logits, target, binary)
            (loss * (len(part) / len(block))).backward()
            loss_sum += loss.detach() * len(part)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["grad_clip_norm"], error_if_nonfinite=True,
        )
        max_gradient = torch.maximum(max_gradient, norm)
        optimizer.step()
        scheduler.step()
        tick(step, steps, min(offset + effective, count), count)
    loss, gradient = (loss_sum / count).item(), max_gradient.item()
    if not math.isfinite(loss) or not math.isfinite(gradient) or gradient <= 0:
        raise ValueError("Nonfinite loss, nonfinite gradient or zero gradient")
    return loss, gradient, steps


@torch.inference_mode()
def _score(model, partition, device, settings, *, binary, multiple, report=None):
    model.eval()
    elapsed_start = time.perf_counter()
    result = []
    batch_size = settings["evaluation_batch_size"]
    next_update = elapsed_start + 15
    for start in range(0, len(partition.targets), batch_size):
        count = min(start + batch_size, len(partition.targets))
        inputs, _ = partition.batch(np.arange(start, count), multiple)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=settings["precision"] == "bf16"):
            logits = model(**inputs).logits
        expected = (count - start, 2 if binary else 28)
        if tuple(logits.shape) != expected:
            raise ValueError(f"Incorrect output shape: {tuple(logits.shape)}, expected {expected}")
        result.append(_forward_scores(logits, binary))
        now = time.perf_counter()
        if report and now >= next_update:
            report(count, len(partition.targets))
            next_update = now + 15
    values = torch.cat(result).cpu().numpy()
    return values, time.perf_counter() - elapsed_start


def _peak(device):
    if device.type == "cuda":
        return (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
    return (0, 0)


def _restore_history(source, target, epoch):
    with (source / "history.csv").open(newline="", encoding="utf-8") as incoming, (
        target / "history.csv"
    ).open("x", newline="", encoding="utf-8") as outgoing:
        rows = csv.DictReader(incoming)
        writer = csv.DictWriter(outgoing, fieldnames=rows.fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if int(row["epoch"]) <= epoch)


def train_transformer(attempt, data, rows, encoded, resolved, device, resume=None):
    from transformers import get_linear_schedule_with_warmup

    device = torch.device(device)
    settings, selection = resolved["settings"], resolved["selection"]
    labels, binary = label_names(data), data.binary
    multiple = resolved["tokenization"]["padding_multiple"]
    torch.manual_seed(attempt.identity["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(attempt.identity["seed"])
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(attempt.identity["seed"] + 127)
    model, loading = load_encoder(Path(resolved["resource"]["cache"]), resolved["experiment"], labels)
    initial_dtype = str(next(model.parameters()).dtype)
    model = model.to(device)
    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = next(parameter for name, parameter in model.base_model.named_parameters()
                    if parameter.ndim == 2 and "embeddings" not in name)
    head = model.classifier.weight
    initial_backbone = backbone.detach().clone()
    initial_head = head.detach().clone()
    optimizer = _optimizer(model, settings, device)
    partitions = {key: DevicePartition(item, device) for key, item in encoded.items()}
    steps_per_epoch = math.ceil(len(rows["train"]) / settings["effective_batch_size"])
    total_steps = steps_per_epoch * settings["epochs"]
    warmup_steps = math.ceil(total_steps * settings["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    checkpoint = attempt.path / "checkpoint"
    model.config.save_pretrained(checkpoint)
    epoch_offset, best_epoch, best, stop = 0, 0, -1.0, "epoch_cap"
    saved_scores = None
    if resume is not None:
        state = load_checkpoint(resume / "checkpoint/resume.pt")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        epoch_offset, best_epoch, best = state["epoch"], state["best_epoch"], state["best"]
        saved_scores = state["best_scores"].numpy().copy()
        filename = f"best-epoch-{best_epoch}.pt"
        source = resume / "checkpoint" / filename
        if sha256_file(source) != state["best_sha256"]:
            raise ValueError("Best checkpoint does not match the recovery state")
        shutil.copyfile(source, checkpoint / filename)
        _restore_history(resume, attempt.path, epoch_offset)
        attempt.event("RESUME", f"checkpoint epoch={epoch_offset}", source=str(resume))
        del state
    best_file = checkpoint / f"best-epoch-{best_epoch}.pt"
    if epoch_offset - best_epoch >= settings["patience"]:
        stop = "early_stopping"
    else:
        for epoch in range(epoch_offset + 1, settings["epochs"] + 1):
            started = time.perf_counter()
            next_update = started

            def tick(step, steps, seen, count, current=epoch, start_time=started):
                nonlocal next_update
                now = time.perf_counter()
                if now >= next_update or step == steps:
                    allocated, reserved = _peak(device)
                    attempt.event(
                        "TRAIN", f"epoch {current}/{settings['epochs']} step {step}/{steps}",
                        epoch=current, epoch_cap=settings["epochs"], step=step, steps=steps,
                        samples=seen, total_samples=count, train_samples_per_second=seen /
                        max(now - start_time, 1e-6), peak_allocated_bytes=allocated,
                        peak_reserved_bytes=reserved,
                    )
                    next_update = now + 20

            loss, gradient, steps = _train_epoch(
                model, partitions["train"], device, settings, optimizer, scheduler, generator,
                binary=binary, multiple=multiple, tick=tick,
            )
            attempt.event("VALIDATE", f"epoch {epoch}/{settings['epochs']}", epoch=epoch)
            scores, val_seconds = _score(model, partitions["validation"], device,
                                         settings, binary=binary, multiple=multiple,
                                         report=lambda n, total: attempt.event(
                                             "VALIDATE", f"{n}/{total} samples",
                                             samples=n, total_samples=total,
                                         ))
            outcome = threshold_record(encoded["validation"].targets, scores, binary, selection)
            score = outcome["macro_f1"]
            if score > best + selection["score_tie_atol"]:
                best, best_epoch, saved_scores = score, epoch, scores
                best_file = checkpoint / f"best-epoch-{epoch}.pt"
                _atomic_checkpoint(best_file, model.state_dict())
            allocated, reserved = _peak(device)
            attempt.epoch(epoch=epoch, val_macro_f1=score, threshold=outcome["threshold"],
                          best_epoch=best_epoch, loss=loss, grad_norm=gradient,
                          learning_rate=scheduler.get_last_lr()[0], optimizer_steps=steps,
                          val_seconds=val_seconds, peak_allocated_bytes=allocated,
                          peak_reserved_bytes=reserved, elapsed_seconds=time.perf_counter()-started)
            attempt.event("CHECKPOINT", f"saving full state after epoch {epoch}", epoch=epoch)
            _atomic_checkpoint(checkpoint / "resume.pt", {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
                "best": best, "best_scores": torch.from_numpy(saved_scores),
                "best_sha256": sha256_file(best_file), "sampler_rng": generator.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            })
            if epoch - best_epoch >= settings["patience"]:
                stop = "early_stopping"
                break
    if torch.equal(backbone, initial_backbone) or torch.equal(head, initial_head):
        raise ValueError("Backbone or task head did not update during fine-tuning")
    del optimizer, scheduler, initial_backbone, initial_head
    model.zero_grad(set_to_none=True)
    model.load_state_dict(load_checkpoint(best_file))
    attempt.event("RELOAD", f"verifying selected checkpoint epoch {best_epoch}")
    replay, val_seconds = _score(model, partitions["validation"], device, settings,
                                 binary=binary, multiple=multiple)
    tuning = threshold_record(encoded["validation"].targets, saved_scores, binary, selection)
    cutoff = tuning["threshold"]
    if not np.allclose(replay, saved_scores, atol=resolved["quality"]["reload_score_atol"],
                       rtol=resolved["quality"]["reload_score_rtol"]):
        raise ValueError("Reloaded logits yield different scores")
    if not np.array_equal(replay >= cutoff, saved_scores >= cutoff):
        raise ValueError("Reloaded checkpoint changes thresholded validation predictions")
    train_scores, train_seconds = _score(
        model, partitions["train"], device, settings, binary=binary, multiple=multiple,
        report=lambda n, total: attempt.event("TRAIN_PREDICT", f"{n}/{total} samples",
                                             samples=n, total_samples=total),
    )
    for key, values in (("train", train_scores), ("validation", saved_scores)):
        target = attempt.path / f"predictions_{key}.jsonl"
        write_predictions(target, rows[key], encoded[key].targets, values, cutoff)
        metrics, _ = verify_prediction_file(target, rows[key], encoded[key].targets, binary,
                                            cutoff, auc=key == "validation")
        write_atomic_json(attempt.path / f"metrics_{key}.json", metrics)
    write_atomic_json(attempt.path / "thresholds.json", tuning)
    allocated, reserved = _peak(device)
    write_atomic_json(attempt.path / "quality.json", {
        "status": "PASS", "backbone_loaded": True, "loading": loading, "label_order": labels,
        "logit_width": len(labels), "loss": "cross_entropy" if binary else "bce_with_logits",
        "score_kind": tuning["score_kind"], "fp32_weights": initial_dtype == "torch.float32",
        "parameter_updates": {"backbone": True, "classifier": True},
        "finite_loss_and_gradients": True, "reloaded_predictions": True,
        "max_reload_abs_error": float(np.max(np.abs(replay - saved_scores))),
        "replayed_predictions": True, "splits_used": ["train", "validation"],
        "fresh_optimizer": resume is None, "optimizer": "adamw", "scheduler": "linear",
        "planned_steps": total_steps, "warmup_steps": warmup_steps,
    })
    return {
        "macro_f1": best, "best_epoch": best_epoch, "threshold": cutoff, "stop_reason": stop,
        "checkpoint": best_file.relative_to(attempt.path).as_posix(),
        "parameters": sum(param.numel() for param in model.parameters()),
        "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
        "validation_inference": {"samples": len(rows["validation"]), "seconds": val_seconds,
                                 "batch_size": settings["evaluation_batch_size"]},
        "train_inference_seconds": train_seconds, "device": str(device),
    }

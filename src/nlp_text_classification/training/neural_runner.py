"""Train and reload deterministic neural Batch A checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nlp_text_classification.data.features import encode, vocabulary
from nlp_text_classification.evaluation.batch_metrics import select_threshold
from nlp_text_classification.models.neural import make_model
from nlp_text_classification.training.batch_run import PrunedRun
from nlp_text_classification.training.resources import CPU_THREADS


def _save_atomic(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _restore_history(previous, attempt, through_epoch):
    source = previous.parent / "history.csv"
    if source.is_file():
        with source.open(newline="", encoding="utf-8") as stream, (
            attempt.path / "history.csv"
        ).open("x", newline="", encoding="utf-8") as target:
            reader = csv.DictReader(stream)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames or ())
            writer.writeheader()
            writer.writerows(row for row in reader if int(row["epoch"]) <= through_epoch)


def _fingerprint(state, *, embedding):
    digest = hashlib.sha256()
    for key, tensor in state.items():
        if key.startswith("embedding.") == embedding:
            digest.update(key.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _initialize(model, vocab, counts, embedding, word2vec):
    role = {"type": embedding}
    if embedding != "random_trainable":
        if word2vec is None:
            raise ValueError("Pretrained model needs a Word2Vec source")
        vectors, metadata = word2vec
        matched = [word for word in vocab if word in vectors]
        with torch.no_grad():
            if matched:
                rows = torch.as_tensor([vocab[word] for word in matched])
                values = np.stack([vectors[word] for word in matched])
                model.embedding.weight.index_copy_(0, rows, torch.from_numpy(values))
        role.update(metadata)
        role["vocabulary_coverage"] = len(matched) / len(vocab)
        role["token_coverage"] = sum(counts[word] for word in matched) / sum(counts.values())
    if embedding == "word2vec_frozen":
        model.embedding.weight.requires_grad_(False)
    return role


def features(data, rows, vocab):
    result = encode(rows, data.settings["tokenizer"], vocab, data.settings["max_length"])
    return tuple(torch.from_numpy(item) for item in result)


def _batch(design, idx, minimum, device):
    tokens, lengths = design
    local_lengths = lengths[idx]
    selected = tokens[idx, :max(minimum, int(local_lengths.max()))].to(device)
    return selected, local_lengths


def _collect(model, design, minimum, device, batch_size=128):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for indices in torch.arange(len(design[0])).split(batch_size):
            x, lengths = _batch(design, indices, minimum, device)
            outputs.append(model(x, lengths).sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def _minimum(resolved):
    experiment = resolved["experiment"]
    return max(resolved["settings"]["textcnn"]["kernel_sizes"]) if (
        experiment["model"] == "textcnn"
    ) else 1


def _train_epoch(model, train, target, minimum, device, optimizer, settings, generator, tick):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    effective = settings["effective_batch_size"]
    micro = settings.get("micro_batch_size", effective)
    if micro <= 0 or effective % micro:
        raise ValueError("Micro-batch size must divide effective batch size")
    indices = torch.randperm(len(target), generator=generator)
    trainable = [param for param in model.parameters() if param.requires_grad]
    total = torch.zeros((), device=device)
    max_grad = torch.zeros((), device=device)
    blocks = indices.split(effective)
    refreshed = time.monotonic()
    for step, block in enumerate(blocks, 1):
        optimizer.zero_grad(set_to_none=True)
        for subset in block.split(micro):
            x, lengths = _batch(train, subset, minimum, device)
            y = target[subset].to(device)
            logits = model(x, lengths)
            if logits.shape != y.shape:
                raise ValueError(f"Logits {logits.shape} do not match targets {y.shape}")
            loss = loss_fn(logits, y)
            (loss * (len(subset) / len(block))).backward()
            total += loss.detach() * len(subset)
        if any(param.grad is None for param in trainable):
            raise ValueError("Trainable parameter has no gradient")
        grad = nn.utils.clip_grad_norm_(
            trainable, settings["grad_clip_norm"], error_if_nonfinite=True
        )
        max_grad = torch.maximum(max_grad, grad.detach())
        optimizer.step()
        if time.monotonic() - refreshed >= 20:
            tick(step, len(blocks))
            refreshed = time.monotonic()
    train_loss = float(total.item() / len(target))
    gradient = float(max_grad.item())
    if not np.isfinite(train_loss) or not np.isfinite(gradient) or gradient <= 0:
        raise ValueError("No usable training loss or gradients")
    return train_loss, gradient


def fit(data, resolved, attempt, seed, device, word2vec, pruning, resume):
    settings, experiment = resolved["settings"], resolved["experiment"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(min(torch.get_num_threads(), CPU_THREADS))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this neural run")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    vocab, counts = vocabulary(data.train, data.settings["tokenizer"])
    if not vocab:
        raise ValueError("Training vocabulary is empty")
    model = make_model(experiment["model"], settings, len(vocab) + 2, data.outputs)
    role = _initialize(model, vocab, counts, experiment["embedding"], word2vec)
    initial_embedding = _fingerprint(model.state_dict(), embedding=True)
    initial_network = _fingerprint(model.state_dict(), embedding=False)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed + 127)
    torch.manual_seed(seed + 785)
    train = features(data, data.train, vocab)
    validation = features(data, data.validation, vocab)
    target = torch.from_numpy(data.encode(data.train)).float()
    y_val = data.encode(data.validation)
    minimum = _minimum(resolved)
    checkpoint = attempt.path / "checkpoint" / "model.pt"
    resume_file = attempt.path / "checkpoint" / "resume.pt"
    epoch_offset, best_epoch, best_score = 0, 0, -1.0
    best = None
    if resume is not None:
        previous = Path(resume)
        saved = torch.load(previous / "resume.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampler_rng"].cpu())
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        epoch_offset, best_epoch, best_score = saved["epoch"], saved["best_epoch"], saved["best_score"]
        best = saved["best_checkpoint"]
        _save_atomic(checkpoint, best)
        _restore_history(previous, attempt, epoch_offset)
        attempt.event("RESUME", f"checkpoint epoch={epoch_offset}", source=str(previous))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    already_stopped = epoch_offset - best_epoch >= settings["patience"]
    stop = "early_stopping" if already_stopped else "epoch_cap"
    first = settings["epoch_cap"] + 1 if already_stopped else epoch_offset + 1
    for epoch in range(first, settings["epoch_cap"] + 1):
        started = time.perf_counter()

        def tick(step, total_steps, current=epoch):
            attempt.event("TRAIN", f"epoch={current} batch={step}/{total_steps}",
                          epoch=current, batch=step, total_batches=total_steps)

        loss, grad = _train_epoch(model, train, target, minimum, device, optimizer, settings,
                                  generator, tick)
        val_scores = _collect(model, validation, minimum, device)
        result = select_threshold(y_val, val_scores, data.binary, resolved["selection"], "sigmoid")
        score = result["macro_f1"]
        if score > best_score + resolved["selection"]["score_tie_atol"]:
            best_score, best_epoch = score, epoch
            best = {"model": {key: value.detach().cpu().clone()
                             for key, value in model.state_dict().items()},
                    "vocabulary": vocab, "validation_scores": torch.from_numpy(val_scores)}
            _save_atomic(checkpoint, best)
        attempt.epoch(epoch=epoch, val_macro_f1=score, threshold=result["threshold"],
                      loss=loss, grad_norm=grad, best_epoch=best_epoch,
                      elapsed_seconds=time.perf_counter() - started)
        _save_atomic(resume_file, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_epoch": best_epoch, "best_score": best_score,
            "best_checkpoint": best,
            "sampler_rng": generator.get_state(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
        })
        if epoch - best_epoch >= settings["patience"]:
            stop = "early_stopping"
            break
        if pruning and pruning(epoch, score):
            raise PrunedRun(f"Optuna median pruning at epoch {epoch}")
    actual_embedding = _fingerprint(model.state_dict(), embedding=True)
    actual_network = _fingerprint(model.state_dict(), embedding=False)
    if actual_network == initial_network:
        raise ValueError("Non-embedding model parameters did not update")
    restored = load(checkpoint, resolved, data.outputs, device)
    best_embedding = _fingerprint(restored.state_dict(), embedding=True)
    frozen = experiment["embedding"] == "word2vec_frozen"
    if frozen and (initial_embedding != actual_embedding or best_embedding != initial_embedding
                   or model.get_parameter("embedding.weight").grad is not None):
        raise ValueError("Frozen embedding changed or received a gradient")
    if best is None:
        raise ValueError("No validation checkpoint was selected")
    reloaded = _collect(restored, validation, minimum, device)
    trained = best["validation_scores"].cpu().numpy()
    train_scores = _collect(restored, train, minimum, device)
    return trained, reloaded, train_scores, {
        "best_epoch": best_epoch, "stop_reason": stop, "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "device": device,
        "peak_device_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "word2vec": role, "initial_embedding_sha256": initial_embedding,
        "best_embedding_sha256": best_embedding, "final_embedding_sha256": actual_embedding,
        "non_embedding_init_sha256": initial_network, "frozen_embedding_verified": frozen,
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(vocab, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


def load(path, resolved, outputs, device):
    bundle = torch.load(path, map_location=device, weights_only=True)
    vocab = bundle["vocabulary"]
    network = make_model(resolved["experiment"]["model"], resolved["settings"],
                         len(vocab) + 2, outputs).to(device)
    network.load_state_dict(bundle["model"])
    return network


def predict_model(path, data, resolved, rows, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network = load(path, resolved, data.outputs, device)
    design = features(data, rows, checkpoint["vocabulary"])
    return _collect(network, design, _minimum(resolved), device)

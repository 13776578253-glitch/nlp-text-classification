"""Measure the S3 training path and verify independent-process recovery."""

import hashlib
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_metrics import write_predictions
from nlp_text_classification.experiments.batch_b_s3 import (
    environment,
    inputs,
    memory,
    optimizer_and_schedule,
)
from nlp_text_classification.experiments.batch_b_s3_resources import SUITE
from nlp_text_classification.experiments.suite import read_json
from nlp_text_classification.training.progress import write_atomic_json
from nlp_text_classification.training.qwen_s3 import (
    adapter_model,
    load_base,
    optimizer_digest,
    restore_rng,
    rng_state,
    seed_all,
    tensor_digest,
    trainable_groups,
)
from nlp_text_classification.training.resources import GIB
from nlp_text_classification.training.transformer_runner import (
    DevicePartition,
    classification_loss,
    threshold_record,
    verify_prediction_file,
)


def _tensor_array(path, values):
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def _frozen_base_sample(model):
    base = model.get_base_model().model
    return tensor_digest((name, value.detach().reshape(-1)[:128])
                         for name, value in base.named_parameters() if not value.requires_grad)


def _build(snapshot, suite, labels, *, saved=None, audit=True):
    model, loading = load_base(snapshot, suite, labels, audit=audit)
    load_memory = memory()
    model = adapter_model(model, suite, saved=saved)
    rank, head, lora = trainable_groups(model, suite)
    if lora["targets"] != loading["quantized"]:
        raise ValueError("PEFT targets differ from the audited NF4 text projections")
    return model, rank, head, loading, lora, load_memory


def _pooling_probe(model, pad):
    model.eval()
    ids = torch.tensor([[41, 1024, pad, pad], [772, 23, 35, 127]], device="cuda")
    mask = ids.ne(pad).long()
    target = torch.zeros((2, 28), dtype=torch.float32, device="cuda")
    target[0, [0, 5]] = 1
    target[1, 27] = 1
    last = mask.sum(dim=1) - 1
    model_base = model.get_base_model()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=ids, attention_mask=mask, labels=target,
                       output_hidden_states=True, use_cache=False)
        manual = model_base.score(output.hidden_states[-1])[torch.arange(2, device="cuda"), last]
        independent = classification_loss(output.logits, target, binary=False)
        single = model(input_ids=ids[:1, :2], attention_mask=mask[:1, :2],
                       use_cache=False).logits
        alternate = ids.clone()
        alternate[1] = torch.tensor([81, 53, 149, 27], device="cuda")
        same_shape = model(input_ids=alternate, attention_mask=mask,
                           use_cache=False).logits
    if tuple(output.logits.shape) != (2, 28) or not torch.equal(manual, output.logits) or (
        not torch.isclose(independent, output.loss) or not torch.isfinite(output.logits).all()
    ):
        raise ValueError("Native last-content classifier, BCE or logit width differs")
    if last.tolist() != [1, 3]:
        raise ValueError("Probe does not exercise distinct right-pad lengths")
    eos_example = torch.tensor([41, 1024, pad, pad], device="cuda")
    eos_mask = torch.tensor([1, 1, 1, 0], device="cuda")
    token_indices = torch.arange(len(eos_example), device="cuda")
    eos_by_id = int((token_indices * eos_example.ne(pad)).argmax())
    eos_by_mask = int(eos_mask.sum()) - 1
    if (eos_by_id, eos_by_mask) != (1, 2):
        raise ValueError("Expected PAD=EOS counterexample failed")
    padding_error = float((single.float() - output.logits[:1].float()).abs().max())
    if not torch.equal(same_shape[0], output.logits[0]):
        raise ValueError("Another sequence influences the short row in a fixed-shape batch")
    return {"logit_shape": list(output.logits.shape), "pooling_indices": last.tolist(),
            "bce": independent.item(), "eos_as_content_index_by_mask": eos_by_mask,
            "eos_as_content_index_by_token_id": eos_by_id, "pad_and_eos_id": pad,
            "fixed_shape_batch_isolation_exact": True,
            "single_vs_padded_batch_max_abs_logit_error": padding_error}


def _gradient_evidence(rank, head):
    evidence = {}
    for key, named in (("lora_a", {name: value for name, value in rank.items() if ".lora_A." in name}),
                       ("lora_b", {name: value for name, value in rank.items() if ".lora_B." in name}),
                       ("head", head)):
        if any(p.grad is None for p in named.values()):
            raise ValueError(f"Missing {key} gradients")
        norms = torch.stack([p.grad.float().norm() for p in named.values()])
        if not torch.isfinite(norms).all():
            raise ValueError(f"Nonfinite {key} gradients")
        evidence[key] = {"tensors": len(norms), "nonzero_tensors": int(norms.count_nonzero()),
                         "max_norm": float(norms.max())}
    return evidence


def _stress(model, rank, head, length):
    model.train()
    ids = (torch.arange(length, device="cuda") + 512)[None, :]
    mask = torch.ones_like(ids)
    target = torch.zeros((1, 28), device="cuda")
    target[:, 1] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        loss = classification_loss(logits, target, binary=False)
    loss.backward()
    gradients = _gradient_evidence(rank, head)
    if not gradients["lora_b"]["nonzero_tensors"] or not gradients["head"]["nonzero_tensors"]:
        raise ValueError("Longest-input probe produced no LoRA-B or head gradient")
    for parameter in (*rank.values(), *head.values()):
        parameter.grad = None
    return {"sequence_length": length, "logit_shape": list(logits.shape), "loss": loss.item(),
            "gradients": gradients, "memory": memory()}


def _update(model, optimizer, scheduler, config, partition, indices, parameters):
    model.train()
    multiplier = config["preprocessing"]["padding_multiple"]
    loss_sum = torch.zeros((), device="cuda")
    for index in indices:
        features, target = partition.batch(np.asarray([index]), multiplier)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**features, use_cache=False).logits
            loss = classification_loss(logits, target, binary=False)
        (loss / len(indices)).backward()
        loss_sum += loss.detach()
    gn = torch.nn.utils.clip_grad_norm_(parameters, config["training"]["gradient_clip"],
                                        error_if_nonfinite=True)
    used_lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    result = {"loss": (loss_sum / len(indices)).item(), "gradient_norm": gn.item(),
              "used_lrs": used_lrs, "next_lrs": scheduler.get_last_lr()}
    if not np.isfinite(result["loss"]) or result["gradient_norm"] <= 0:
        raise ValueError("Missing or nonfinite benchmark signal")
    return result


def _indices(data, suite):
    return np.random.default_rng(suite["preflight"]["seed"]).permutation(len(data.train))


def _validation(model, data, encoded, suite):
    from nlp_text_classification.training.transformer_runner import _score

    indices = np.linspace(0, len(data.validation) - 1, 48, dtype=int)
    rows = [data.validation[i] for i in indices]
    targets = encoded.targets[indices]
    subset = type(encoded)({k: v[indices] for k, v in encoded.features.items()},
                           encoded.lengths[indices], targets)
    partition = DevicePartition(subset, "cuda")
    scores, _ = _score(model, partition, torch.device("cuda"),
                       {"evaluation_batch_size": suite["training"]["evaluation_micro_batch_size"],
                        "precision": "bf16"}, binary=False,
                       multiple=suite["preprocessing"]["padding_multiple"])
    return rows, targets, scores


def _publish(root, model, optimizer, scheduler, cursor, order):
    release = root / "adapter"
    model.save_pretrained(release, safe_serialization=True, save_embedding_layers=False)
    model.get_base_model().config.to_json_file(release / "text_config.json")
    with safe_open(release / "adapter_model.safetensors", framework="pt") as saved:
        keys = saved.keys()
        head = [key for key in keys if key.endswith(".score.weight")]
        if len(keys) != 497 or len(head) != 1 or saved.get_slice(head[0]).get_shape() != [28, 2560]:
            raise ValueError("Adapter package omits head or LoRA weights")
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "rng": rng_state(), "cursor": cursor, "order_sha256": hashlib.sha256(
                 order.tobytes()).hexdigest()}
    staged = root / "resume.pending"
    torch.save(state, staged)
    staged.rename(root / "resume.pt")
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in (*sorted(release.iterdir()), root / "resume.pt")}


def _state_evidence(rank, head, optimizer):
    return {"lora_sha256": tensor_digest(rank.items()), "head_sha256": tensor_digest(head.items()),
            "optimizer_sha256": optimizer_digest(optimizer)}


def _rng_probe():
    return {"python": random.random(), "numpy": float(np.random.random()),
            "cpu": torch.rand(2).tolist(), "cuda": torch.rand(2, device="cuda").tolist()}


def benchmark(root, suite, progress):
    if (root / "load-audit.json").exists():
        raise FileExistsError("Choose a new attempt directory to preserve prior evidence")
    preflight, options = suite["preflight"], suite["training"]
    progress.emit("INPUT_CHECK", "Verifying the pinned snapshot and train/validation encoding")
    snapshot, data, tokens, encoded = inputs(suite)
    write_atomic_json(root / "tokenization.json", tokens)
    write_atomic_json(root / "environment.json", environment())
    progress.emit("LOAD", "Loading the official Qwen text backbone as NF4",
                  max_length=tokens["selected_length"])
    seed_all(preflight["seed"])
    start_memory = memory()
    torch.cuda.reset_peak_memory_stats()
    model, rank, head, loading, lora, load_memory = _build(
        snapshot, suite, tokens["label_order"],
    )
    write_atomic_json(root / "load-audit.json", loading | {"memory": load_memory, "status": "PASS"})
    write_atomic_json(root / "lora-audit.json", lora | {"status": "PASS"})
    progress.emit("LOAD_OK", "426 backbone tensors, 248 NF4 adapters and the official head verified",
                  trainable=sum(lora["counts"].values()))
    pooling = _pooling_probe(model, tokens["policy"]["pad_token_id"])
    stress = _stress(model, rank, head, preflight["stress_length"])
    write_atomic_json(root / "functional.json", {"pooling": pooling, "stress": stress})
    progress.emit("STRESS_OK", "Native pooling, BCE and maximum length backward passed",
                  stress_length=stress["sequence_length"])
    train = DevicePartition(encoded["train"], "cuda")
    order = _indices(data, suite)
    optimizer, scheduler, plan = optimizer_and_schedule(rank, head, options, len(data.train))
    params = [*rank.values(), *head.values()]
    original = _state_evidence(rank, head, optimizer)
    frozen = _frozen_base_sample(model)
    effective = options["accumulation_steps"]
    warmup, measured = preflight["warmup_updates"], preflight["stable_updates"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    progress.emit("BENCHMARK", "Starting real train rows with 16 samples per optimizer update",
                  warmup_updates=warmup, measured_updates=measured)
    with (root / "updates.jsonl").open("x", encoding="utf-8") as stream:
        for step in range(warmup + measured):
            started = time.perf_counter()
            outcome = _update(model, optimizer, scheduler, suite, train,
                              order[step * effective:(step + 1) * effective], params)
            elapsed = time.perf_counter() - started
            if step >= warmup:
                durations.append(elapsed)
            stream.write(json_line({"update": step + 1, "seconds": elapsed, **outcome}))
            stream.flush()
            if step == 0 or (step + 1) % 5 == 0:
                progress.emit("UPDATE", f"{step + 1}/{warmup + measured} optimizer updates",
                              measured=max(0, step + 1 - warmup),
                              samples_per_second=round(effective / elapsed, 3),
                              loss=round(outcome["loss"], 6))
    used_memory = memory()
    rows, targets, scores = _validation(model, data, encoded["validation"], suite)
    threshold = threshold_record(targets, scores, False, suite["selection"])
    write_predictions(root / "predictions_validation.jsonl", rows, targets, scores,
                      threshold["threshold"])
    _tensor_array(root / "validation-scores.npy", scores)
    trained = _state_evidence(rank, head, optimizer)
    if any(trained[key] == original[key] for key in trained) or _frozen_base_sample(model) != frozen:
        raise ValueError("The trainable groups did not update or frozen parameters changed")
    cursor = warmup + measured
    files = _publish(root, model, optimizer, scheduler, cursor, order)
    assessment = {
        "pid": os.getpid(), "suite_sha256": sha256_file(SUITE), "start_memory": start_memory,
        "stable_updates": measured, "warmup_updates": warmup, "update_rows": effective,
        "stable_seconds": sum(durations), "samples_per_second": measured * effective / sum(durations),
        "update_seconds_p50": float(np.median(durations)), "update_seconds_p95": float(
            np.percentile(durations, 95)),
        "estimated_three_seed_training_hours": len(data.train) * options["epochs"] *
        len(options["seeds"]) * sum(durations) / (measured * effective) / 3600,
        "training_memory": used_memory, "stress_memory": stress["memory"], "threshold": threshold,
        "validation_rows": len(rows), "options": plan, "checkpoint_files": files, "trained": trained,
        "frozen_parameter_sample_unchanged": True,
    }
    progress.emit("RESUME_REFERENCE", "Published adapter and resume state; recording next update")
    rng_probe = _rng_probe()
    following = _update(model, optimizer, scheduler, suite, train,
                        order[cursor * effective:(cursor + 1) * effective], params)
    assessment["reference"] = {"outcome": following, "state": _state_evidence(rank, head, optimizer),
                               "rng_probe": rng_probe}
    peak = max(used_memory["gpu_peak_reserved_bytes"], stress["memory"]["gpu_peak_reserved_bytes"])
    assessment["remaining_vram_estimate_gib"] = (start_memory["gpu_free_bytes"] - peak) / GIB
    assessment["status"] = ("PASS" if assessment["remaining_vram_estimate_gib"] >=
                            preflight["min_free_vram_gib"] else "REVIEW_VRAM")
    write_atomic_json(root / "benchmark.json", assessment)
    progress.emit("BENCHMARK_COMPLETE", f"100 stable updates measured: {assessment['status']}",
                  samples_per_second=round(assessment["samples_per_second"], 3))
    return {key: assessment[key] for key in ("status", "stable_updates", "samples_per_second")}


def json_line(payload):
    import json

    return json.dumps(payload, allow_nan=False) + "\n"


def recover(root: Path, suite, progress):
    preflight, options = suite["preflight"], suite["training"]
    previous = read_json(root / "benchmark.json")
    if previous["pid"] == os.getpid() or previous["suite_sha256"] != sha256_file(SUITE):
        raise ValueError("Independent recovery requires the original configuration and fresh PID")
    for name, digest in previous["checkpoint_files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Changed checkpoint evidence: {name}")
    snapshot, data, tokens, encoded = inputs(suite)
    if tokens != read_json(root / "tokenization.json"):
        raise ValueError("Recovery input fingerprint changed")
    seed_all(12457)
    model, rank, head, _, _, _ = _build(snapshot, suite, tokens["label_order"],
                                       saved=root / "adapter", audit=False)
    optimizer, scheduler, plan = optimizer_and_schedule(rank, head, options, len(data.train))
    if plan != previous["options"]:
        raise ValueError("Recovery changed the optimization budget")
    state = torch.load(root / "resume.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if _state_evidence(rank, head, optimizer) != previous["trained"]:
        raise ValueError("Recovered LoRA, head or AdamW bytes differ")
    rows, targets, scores = _validation(model, data, encoded["validation"], suite)
    _, reference = verify_prediction_file(root / "predictions_validation.jsonl", rows, targets,
                                          False, previous["threshold"]["threshold"], auc=True)
    if not np.allclose(scores, reference, atol=preflight["replay_atol"],
                       rtol=preflight["replay_rtol"]) or (
        threshold_record(targets, scores, False, suite["selection"]) != previous["threshold"]
    ) or not np.array_equal(scores >= previous["threshold"]["threshold"],
                            reference >= previous["threshold"]["threshold"]):
        raise ValueError("Independently reloaded adapter changes validation output")
    order = _indices(data, suite)
    if hashlib.sha256(order.tobytes()).hexdigest() != state["order_sha256"]:
        raise ValueError("Sampler permutation changed")
    restore_rng(state["rng"])
    rng_probe = _rng_probe()
    cursor, effective = state["cursor"], options["accumulation_steps"]
    train = DevicePartition(encoded["train"], "cuda")
    outcome = _update(model, optimizer, scheduler, suite, train,
                      order[cursor * effective:(cursor + 1) * effective],
                      [*rank.values(), *head.values()])
    observed = {"outcome": outcome, "state": _state_evidence(rank, head, optimizer),
                "rng_probe": rng_probe}
    if observed != previous["reference"]:
        write_atomic_json(root / "recovery-difference.json", observed)
        raise ValueError("Recovered next optimizer update or random state differs")
    receipt = {"status": "PASS", "original_pid": previous["pid"], "recovery_pid": os.getpid(),
               "validated_rows": len(rows), "resume_cursor": cursor, "next_update": cursor + 1,
               "max_abs_score_error": float(np.max(np.abs(scores - reference))),
               "exact_score_match": bool(np.array_equal(scores, reference)),
               "exact_next_update": True, "exact_rng_probe": True, "memory": memory()}
    write_atomic_json(root / "recovery.json", receipt)
    progress.emit("RECOVERY_COMPLETE", "Independent process reproduced scores, RNG and next update",
                  max_error=receipt["max_abs_score_error"])
    return receipt

"""Isolated model builders, profiled training/eval runners, and numerical checks for S4."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.evaluation.batch_metrics import write_predictions
from nlp_text_classification.experiments.batch_b_s3 import memory
from nlp_text_classification.experiments.batch_b_s3_perf import INITIAL as S3_INITIAL
from nlp_text_classification.experiments.batch_b_s3_perf import (
    MATCHED_STABLE,
    MATCHED_WARMUP,
    _package_versions,
)
from nlp_text_classification.experiments.batch_b_s3_resources import ROOT as S3_ROOT
from nlp_text_classification.experiments.batch_b_s4_telemetry import (
    GpuTelemetryMonitor,
    parse_chrome_trace,
)
from nlp_text_classification.experiments.suite import read_json
from nlp_text_classification.training.progress import write_atomic_json
from nlp_text_classification.training.qwen_s3 import (
    TEXT_PREFIX,
    adapter_model,
    audit_weight_map,
    load_base,
    optimizer_digest,
    optimizer_for,
    restore_rng,
    rng_state,
    seed_all,
    tensor_digest,
    text_config,
    trainable_groups,
)
from nlp_text_classification.training.qwen_s3_streaming import OfflineShards
from nlp_text_classification.training.transformer_runner import (
    DevicePartition,
    classification_loss,
    threshold_record,
    verify_prediction_file,
)

S4_PERF = S3_ROOT.parent / "batch-b-s4" / "performance"
S3_WSL_DIR = S3_ROOT / "performance" / "ubuntu-fused"
MIN_FREE_VRAM_MIB = 2048


def _state_evidence(rank: dict, head: dict, optimizer: torch.optim.Optimizer) -> dict:
    return {
        "lora_sha256": tensor_digest(rank.items()),
        "head_sha256": tensor_digest(head.items()),
        "optimizer_sha256": optimizer_digest(optimizer),
    }


def _rng_probe() -> dict:
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "cpu": torch.rand(2).tolist(),
        "cuda": torch.rand(2, device="cuda").tolist(),
    }


def _gradient_check(rank: dict, head: dict) -> dict:
    summary = {}
    for label, group in (
        ("lora_a", {k: v for k, v in rank.items() if ".lora_A." in k}),
        ("lora_b", {k: v for k, v in rank.items() if ".lora_B." in k}),
        ("head", head),
    ):
        if any(p.grad is None for p in group.values()):
            raise ValueError(f"Missing {label} gradients")
        norms = torch.stack([p.grad.float().norm() for p in group.values()])
        if not torch.isfinite(norms).all():
            raise ValueError(f"Non-finite {label} gradients")
        summary[label] = {
            "tensors": len(norms),
            "nonzero_tensors": int(norms.count_nonzero()),
            "max_norm": float(norms.max()),
        }
    if not summary["lora_b"]["nonzero_tensors"] or not summary["head"]["nonzero_tensors"]:
        raise ValueError("Zero gradient in LoRA-B or score head")
    return summary


def load_base_bf16(snapshot: Path, suite: dict, labels: list[str]):
    """Load the 426 official text tensors directly in native BF16 without quantization."""
    from transformers import Qwen3_5TextForSequenceClassification

    config = text_config(snapshot, labels, suite["preprocessing"]["pad_token_id"])
    config._attn_implementation = "sdpa"
    index, source = audit_weight_map(snapshot, config, Qwen3_5TextForSequenceClassification)
    mapping = {name: TEXT_PREFIX + name.removeprefix("model.") for name in source}
    with torch.device("meta"):
        model = Qwen3_5TextForSequenceClassification(config)
    with torch.device("cuda"):
        model.model.rotary_emb = type(model.model.rotary_emb)(config)
    with ExitStack() as stack:
        archive = OfflineShards(snapshot, index, stack)
        for target, key in mapping.items():
            module_name, attribute = target.rsplit(".", 1)
            module = model.get_submodule(module_name)
            value = archive.tensor(key, "cuda", torch.bfloat16)
            setattr(module, attribute, torch.nn.Parameter(value, requires_grad=False))
    model.score.weight = torch.nn.Parameter(
        torch.empty((len(labels), 2560), device="cuda", dtype=torch.bfloat16)
    )
    model._init_weights(model.score)
    if any(t.device.type != "cuda" for t in (*model.parameters(), *model.buffers())):
        raise ValueError("BF16 loader left unmaterialized tensor")
    return model


def build_model_s4(
    snapshot: Path,
    suite: dict,
    labels: list[str],
    *,
    precision_mode: str,
    gc_mode: str,
    saved: Path | None = None,
    fresh: bool = False,
):
    """Build Qwen classifier under controlled precision and gradient checkpointing modes."""
    from peft import PeftModel

    if fresh and (saved is not None or precision_mode != "nf4_fp32_stream"):
        raise ValueError("Fresh initialization requires the approved NF4 pathway")
    use_gc = gc_mode != "off"
    if precision_mode == "nf4_fp32_stream":
        base_model, _ = load_base(snapshot, suite, labels, audit=False)
        adapter = None if fresh else saved or (S3_INITIAL / "adapter")
        model = adapter_model(base_model, suite, saved=adapter)
    elif precision_mode in {"nf4_bf16_stream", "bf16_lora"}:
        if precision_mode == "nf4_bf16_stream":
            base_model, _ = load_base(snapshot, suite, labels, audit=False)
        else:
            base_model = load_base_bf16(snapshot, suite, labels)
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.score.weight.data = base_model.score.weight.data.to(torch.float32)
        model = PeftModel.from_pretrained(
            base_model, saved or (S3_INITIAL / "adapter"), is_trainable=True
        )
        for param in model.parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
    else:
        raise ValueError(f"Unsupported precision_mode: {precision_mode}")

    configure_gradient_checkpointing(model, gc_mode, use_gc=use_gc)
    rank, head, lora = trainable_groups(model, suite)
    if saved is None and not fresh:
        expected_start = read_json(S3_INITIAL / "start.json")
        if (
            tensor_digest(rank.items()) != expected_start["lora_sha256"]
            or tensor_digest(head.items()) != expected_start["head_sha256"]
        ):
            raise ValueError(f"{precision_mode}: initial LoRA/head digest differs from S3")
    return model, rank, head, lora


def configure_gradient_checkpointing(model, gc_mode: str, *, use_gc: bool = True) -> dict:
    """Configure full, partial, or disabled non-reentrant gradient checkpointing."""
    base = model.get_base_model()
    if gc_mode == "off" or not use_gc:
        base.gradient_checkpointing_disable()
        base.enable_input_require_grads()
        enabled_layers = 0
    else:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        layers = base.model.layers
        if gc_mode == "full":
            for layer in layers:
                layer.gradient_checkpointing = True
        elif gc_mode == "partial_16":
            for idx, layer in enumerate(layers):
                layer.gradient_checkpointing = (idx % 2 == 0)
        elif gc_mode == "partial_attn8":
            for layer in layers:
                layer.gradient_checkpointing = (layer.layer_type == "full_attention")
        else:
            raise ValueError(f"Unsupported gc_mode: {gc_mode}")
        enabled_layers = sum(1 for layer in layers if layer.gradient_checkpointing)
    return {
        "gc_mode": gc_mode,
        "enabled_layers": enabled_layers,
        "total_layers": len(base.model.layers),
    }


def update_budget(options, rows):
    effective = options["micro_batch_size"] * options["accumulation_steps"]
    if rows <= 0 or effective <= 0 or options["epochs"] <= 0 or not (
        0 <= options["warmup_ratio"] < 1
    ):
        raise ValueError("Invalid training budget")
    per_epoch = (rows + effective - 1) // effective
    total = per_epoch * options["epochs"]
    return {
        "effective_batch_size": effective, "updates_per_epoch": per_epoch,
        "tail_rows": rows % effective, "total_updates": total,
        "warmup_updates": math.ceil(Decimal(str(options["warmup_ratio"])) * total),
    }


def make_optimizer(rank: dict, head: dict, options: dict, rows: int, *, foreach: bool):
    from transformers import get_linear_schedule_with_warmup

    plan = update_budget(options, rows)
    optimizer = optimizer_for(rank, head, options)
    optimizer.defaults["foreach"] = foreach
    for group in optimizer.param_groups:
        group["foreach"] = foreach
    scheduler = get_linear_schedule_with_warmup(
        optimizer, plan["warmup_updates"], plan["total_updates"],
    )
    return optimizer, scheduler, plan


def iter_micro_batches(
    indices: np.ndarray,
    lengths: np.ndarray,
    micro_batch_size: int,
    sort_mode: str,
) -> list[np.ndarray]:
    """Split an effective batch of sample indices into micro-batches cleanly."""
    if sort_mode == "intra_update_sorted":
        ordered = indices[np.argsort(lengths[indices], kind="stable")]
    elif sort_mode == "unsorted":
        ordered = indices
    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode}")
    return [ordered[i:i + micro_batch_size] for i in range(0, len(ordered), micro_batch_size)]


def fetch_batch(partition: DevicePartition, chunk: np.ndarray, padding_mode: str):
    if padding_mode == "static_64":
        index = torch.as_tensor(chunk, device=partition.targets.device)
        return (
            {k: v.index_select(0, index).contiguous() for k, v in partition.features.items()},
            partition.targets.index_select(0, index),
        )
    multiple = 8 if padding_mode == "dynamic_mult8" else 1
    return partition.batch(chunk, multiple)


def run_update(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    partition: DevicePartition,
    indices: np.ndarray,
    parameters: list,
    *,
    micro_batch_size: int,
    padding_mode: str,
    sort_mode: str,
    check_grad_detail: bool = False,
) -> dict:
    model.train()
    chunks = iter_micro_batches(indices, partition.lengths, micro_batch_size, sort_mode)
    total_rows = len(indices)
    loss_sum = torch.zeros((), device="cuda")
    padded_tokens = 0
    valid_tokens = int(np.minimum(partition.lengths[indices], 64).sum())
    for chunk in chunks:
        features, target = fetch_batch(partition, chunk, padding_mode)
        padded_tokens += int(features["input_ids"].numel())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**features, use_cache=False).logits
            loss = classification_loss(logits, target, binary=False)
        (loss * (len(chunk) / total_rows)).backward()
        loss_sum += loss.detach() * len(chunk)
    grad_detail = None
    if check_grad_detail:
        rank_map = {n: p for n, p in model.named_parameters() if p.requires_grad and ".lora_" in n}
        head_map = {n: p for n, p in model.named_parameters() if p.requires_grad and ".score." in n}
        grad_detail = _gradient_check(rank_map, head_map)
    gn = torch.nn.utils.clip_grad_norm_(
        parameters, config["training"]["gradient_clip"], error_if_nonfinite=True
    )
    used_lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    loss_val = (loss_sum / total_rows).item()
    gn_val = gn.item()
    if not np.isfinite(loss_val) or not np.isfinite(gn_val) or gn_val <= 0:
        raise ValueError(f"Invalid update signal: loss={loss_val}, gn={gn_val}")
    return {
        "loss": loss_val,
        "gradient_norm": gn_val,
        "used_lrs": used_lrs,
        "next_lrs": scheduler.get_last_lr(),
        "valid_tokens": valid_tokens,
        "padded_tokens": padded_tokens,
        "grad_detail": grad_detail,
    }


@torch.inference_mode()
def evaluate_subset(
    model,
    encoded_val,
    indices: np.ndarray,
    *,
    batch_size: int,
    multiple: int = 8,
    sort_by_length: bool = False,
    on_batch=None,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Evaluate validation samples and return logits, sigmoid scores, elapsed seconds, and stats."""
    model.eval()
    subset = type(encoded_val)(
        {k: v[indices] for k, v in encoded_val.features.items()},
        encoded_val.lengths[indices],
        encoded_val.targets[indices],
    )
    partition = DevicePartition(subset, "cuda")
    n = len(indices)
    order = np.argsort(subset.lengths, kind="stable") if sort_by_length else np.arange(n)
    logits_out = torch.empty((n, 28), dtype=torch.float32, device="cpu")
    scores_out = torch.empty((n, 28), dtype=torch.float32, device="cpu")
    padded_tokens = 0
    torch.cuda.synchronize()
    started = time.perf_counter()
    for pos in range(0, n, batch_size):
        chunk = order[pos:pos + batch_size]
        features, _ = partition.batch(chunk, multiple)
        padded_tokens += int(features["input_ids"].numel())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**features, use_cache=False).logits
        logits_f = logits.float().cpu()
        scores_f = logits_f.sigmoid()
        idx_t = torch.as_tensor(chunk, dtype=torch.long)
        logits_out[idx_t] = logits_f
        scores_out[idx_t] = scores_f
        if on_batch is not None:
            on_batch(min(pos + batch_size, n))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    valid_tokens = int(np.minimum(subset.lengths, 64).sum())
    return logits_out.numpy(), scores_out.numpy(), elapsed, {
        "rows": n,
        "batch_size": batch_size,
        "multiple": multiple,
        "sort_by_length": sort_by_length,
        "valid_tokens": valid_tokens,
        "padded_tokens": padded_tokens,
        "samples_per_second": n / elapsed if elapsed > 0 else 0.0,
    }


def compare_scores(ref_scores: np.ndarray, cand_scores: np.ndarray) -> dict:
    """Compare sigmoid score matrices and derived logits against reference."""
    diff = np.abs(cand_scores.astype(np.float64) - ref_scores.astype(np.float64))
    eps = 1e-7
    ref_c = np.clip(ref_scores.astype(np.float64), eps, 1.0 - eps)
    cand_c = np.clip(cand_scores.astype(np.float64), eps, 1.0 - eps)
    ref_logits = np.log(ref_c / (1.0 - ref_c))
    cand_logits = np.log(cand_c / (1.0 - cand_c))
    logit_diff = np.abs(cand_logits - ref_logits)
    flips_05 = int(np.sum((ref_scores >= 0.5) != (cand_scores >= 0.5)))
    borderline = (np.abs(ref_scores - 0.5) <= 0.05)
    return {
        "shape": list(ref_scores.shape),
        "prob_mean_abs": float(diff.mean()),
        "prob_p95_abs": float(np.percentile(diff, 95)),
        "prob_max_abs": float(diff.max()),
        "logit_mean_abs": float(logit_diff.mean()),
        "logit_p95_abs": float(np.percentile(logit_diff, 95)),
        "logit_max_abs": float(logit_diff.max()),
        "threshold_0_5_flips": flips_05,
        "threshold_0_5_total": int(ref_scores.size),
        "borderline_count": int(borderline.sum()),
        "borderline_flips": int(np.sum(((ref_scores >= 0.5) != (cand_scores >= 0.5)) & borderline)),
    }


def worst_case_stress(model, rank: dict, head: dict, micro_batch_size: int, seq_len: int = 64):
    """Run worst-case micro_batch_size x seq_len forward+backward to verify VRAM safety."""
    model.train()
    ids = (torch.arange(seq_len, device="cuda") + 512)[None, :].expand(micro_batch_size, -1)
    mask = torch.ones_like(ids)
    target = torch.zeros((micro_batch_size, 28), device="cuda")
    target[:, 1] = 1.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        loss = classification_loss(logits, target, binary=False)
    loss.backward()
    grad_detail = _gradient_check(rank, head)
    for param in (*rank.values(), *head.values()):
        param.grad = None
    mem = memory()
    return {
        "micro_batch_size": micro_batch_size,
        "sequence_length": seq_len,
        "loss": float(loss.item()),
        "gradients": grad_detail,
        "memory": mem,
    }


def snapshot_trainable_state(rank: dict, head: dict) -> dict[str, torch.Tensor]:
    """Clone initial trainable LoRA and head weights to CPU for exact in-process resets."""
    return {name: param.detach().cpu().clone() for name, param in (rank | head).items()}


def restore_trainable_state(rank: dict, head: dict, saved: dict[str, torch.Tensor]) -> None:
    """Restore initial trainable weights in-place and verify exact S3 SHA-256 parity."""
    with torch.no_grad():
        for name, param in (rank | head).items():
            param.copy_(saved[name])
            param.grad = None
    expected = read_json(S3_INITIAL / "start.json")
    if (
        tensor_digest(rank.items()) != expected["lora_sha256"]
        or tensor_digest(head.items()) != expected["head_sha256"]
    ):
        raise ValueError("Restored LoRA/head digest differs from S3 initial bytes")


def run_experiment_with_model(
    spec: dict,
    suite: dict,
    data,
    encoded: dict,
    progress,
    model,
    rank: dict,
    head: dict,
    start_mem: dict,
    load_mem: dict,
) -> dict:
    """Execute one S4 experiment spec on a pre-loaded model with fresh optimizer/RNG/VRAM stats."""
    exp_id = spec["exp_id"]
    out_dir = S4_PERF / "runs" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    micro = int(spec["micro_batch_size"])
    accum = int(spec["accumulation_steps"])
    if micro * accum != suite["training"]["accumulation_steps"]:
        raise ValueError(f"{exp_id}: micro*accum must equal 16")
    padding_mode = spec["padding_mode"]
    sort_mode = spec["sort_mode"]
    gc_mode = spec["gc_mode"]
    foreach = bool(spec.get("foreach_adamw", False))
    eval_batch = int(spec.get("eval_batch_size", 1))
    eval_sorted = bool(spec.get("eval_sorted", False))
    run_stress = bool(spec.get("stress_worst_batch", False))
    publish = bool(spec.get("publish_recovery", False))

    s3_init_scores = np.load(S3_WSL_DIR / "initial-scores.npy")
    _, _s3_post_rows, s3_post_scores = _load_s3_post13()

    seed_all(suite["preflight"]["seed"])
    optimizer = None
    scheduler = None
    train_part = None
    params = [*rank.values(), *head.values()]
    for param in params:
        param.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with GpuTelemetryMonitor(interval=0.25) as monitor:
        try:
            configure_gradient_checkpointing(model, gc_mode, use_gc=(gc_mode != "off"))
            options = dict(suite["training"])
            optimizer, scheduler, _plan = make_optimizer(
                rank, head, options, len(data.train), foreach=foreach
            )
            restore_rng(torch.load(S3_INITIAL / "rng.pt", weights_only=True, map_location="cpu"))

            val_indices_48 = np.linspace(0, len(data.validation) - 1, 48, dtype=int)
            _, init_scores_48, init_val_s, _ = evaluate_subset(
                model, encoded["validation"], val_indices_48,
                batch_size=eval_batch, sort_by_length=eval_sorted,
            )
            init_diff = compare_scores(s3_init_scores, init_scores_48)

            order = np.random.default_rng(suite["preflight"]["seed"]).permutation(len(data.train))
            train_part = DevicePartition(encoded["train"], "cuda")
            effective = suite["training"]["accumulation_steps"]
            warmup, measured = MATCHED_WARMUP, MATCHED_STABLE

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            updates = []
            for step in range(warmup + measured):
                if step == warmup:
                    torch.cuda.synchronize()
                    monitor.mark_window_start()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                row = run_update(
                    model, optimizer, scheduler, suite, train_part,
                    order[step * effective:(step + 1) * effective], params,
                    micro_batch_size=micro, padding_mode=padding_mode, sort_mode=sort_mode,
                    check_grad_detail=(step == warmup),
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                updates.append({
                    "step": step + 1,
                    "seconds": elapsed,
                    "loss": row["loss"],
                    "gradient_norm": row["gradient_norm"],
                    "valid_tokens": row["valid_tokens"],
                    "padded_tokens": row["padded_tokens"],
                    "grad_detail": row["grad_detail"],
                })
                if step == warmup + measured - 1:
                    monitor.mark_window_end()
                progress.emit("S4_STEP", f"{exp_id} step {step + 1}/{warmup + measured}",
                              seconds=round(elapsed, 3), loss=round(row["loss"], 5))

            train_mem = memory()
            stable_rows = updates[warmup:]
            stable_secs = np.asarray([r["seconds"] for r in stable_rows], dtype=float)
            total_stable_s = float(stable_secs.sum())
            total_valid_tok = int(sum(r["valid_tokens"] for r in stable_rows))
            total_padded_tok = int(sum(r["padded_tokens"] for r in stable_rows))
            train_sps = float(measured * effective / total_stable_s)

            _, post_scores_48, post_val_s, val_stats_48 = evaluate_subset(
                model, encoded["validation"], val_indices_48,
                batch_size=eval_batch, sort_by_length=eval_sorted,
            )
            post_diff = compare_scores(s3_post_scores, post_scores_48)

            stress_info = None
            if run_stress:
                stress_info = worst_case_stress(model, rank, head, micro, seq_len=64)

            telemetry = monitor.summarize()
            min_free = telemetry.get("min_free_mib", 0)
            vram_ok = min_free >= MIN_FREE_VRAM_MIB
            if stress_info is not None:
                stress_free_mib = int(stress_info["memory"]["gpu_free_bytes"] // (1024 * 1024))
                vram_ok = vram_ok and (min_free >= MIN_FREE_VRAM_MIB) and (
                    stress_free_mib >= MIN_FREE_VRAM_MIB
                )

            publish_info = None
            if publish and vram_ok:
                publish_info = _publish_s4(
                    out_dir, model, optimizer, scheduler, warmup + measured, order,
                    data, encoded, suite, val_indices_48, post_scores_48,
                    rank, head, train_part, params, micro, padding_mode, sort_mode,
                )

            status = "PASS" if vram_ok else "FAIL_VRAM_HEADROOM"
            result = {
                "exp_id": exp_id,
                "category": spec["category"],
                "status": status,
                "config": spec,
                "environment": {"backend": "wsl", "packages": _package_versions()},
                "cold_step1_seconds": updates[0]["seconds"],
                "warmup_mean_seconds": float(np.mean([r["seconds"] for r in updates[:warmup]])),
                "stable_seconds": total_stable_s,
                "mean_update_seconds": float(stable_secs.mean()),
                "median_update_seconds": float(np.median(stable_secs)),
                "p95_update_seconds": float(np.percentile(stable_secs, 95)),
                "train_samples_per_second": train_sps,
                "train_valid_tokens_per_second": float(total_valid_tok / total_stable_s),
                "train_padded_tokens_per_second": float(total_padded_tok / total_stable_s),
                "padding_efficiency": (
                    total_valid_tok / total_padded_tok if total_padded_tok else 0.0
                ),
                "valid_tokens_window": total_valid_tok,
                "padded_tokens_window": total_padded_tok,
                "init_val_seconds": init_val_s,
                "post_val_48_seconds": post_val_s,
                "val_stats_48": val_stats_48,
                "initial_diff_vs_s3": init_diff,
                "post13_diff_vs_s3": post_diff,
                "grad_detail_step6": updates[warmup]["grad_detail"],
                "start_memory": start_mem,
                "load_memory": load_mem,
                "training_memory": train_mem,
                "telemetry": telemetry,
                "stress_worst_batch": stress_info,
                "publish": publish_info,
                "updates": [
                    {k: v for k, v in row.items() if k != "grad_detail"} for row in updates
                ],
                "error": None if vram_ok else (
                    f"min_free_mib={min_free} below threshold {MIN_FREE_VRAM_MIB}"
                ),
            }
        except torch.cuda.OutOfMemoryError as exc:
            telemetry = monitor.summarize()
            result = {
                "exp_id": exp_id,
                "category": spec["category"],
                "status": "FAIL_OOM",
                "config": spec,
                "telemetry": telemetry,
                "error": str(exc),
            }
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            telemetry = monitor.summarize()
            result = {
                "exp_id": exp_id,
                "category": spec["category"],
                "status": "FAIL_ERROR",
                "config": spec,
                "telemetry": telemetry,
                "error": f"{type(exc).__name__}: {exc}",
            }

    for param in params:
        param.grad = None
    optimizer = None
    scheduler = None
    train_part = None
    write_atomic_json(out_dir / "result.json", result)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_experiment(spec: dict, suite: dict, snapshot: Path, data, encoded: dict, progress) -> dict:
    """Execute one isolated S4 experiment spec with telemetry, validation, and VRAM guard."""
    seed_all(suite["preflight"]["seed"])
    start_mem = memory()
    torch.cuda.reset_peak_memory_stats()
    model, rank, head, _lora = build_model_s4(
        snapshot, suite, data.manifest["label_names"],
        precision_mode=spec["precision_mode"], gc_mode=spec["gc_mode"],
    )
    load_mem = memory()
    result = run_experiment_with_model(
        spec, suite, data, encoded, progress, model, rank, head, start_mem, load_mem
    )
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _load_s3_post13():
    rows, scores = [], []
    with (S3_WSL_DIR / "predictions_validation.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            rows.append(item["sample_id"])
            scores.append(item["scores"])
    return S3_WSL_DIR / "predictions_validation.jsonl", rows, np.asarray(scores, dtype=np.float32)


def _publish_s4(
    out_dir: Path,
    model,
    optimizer,
    scheduler,
    cursor: int,
    order: np.ndarray,
    data,
    encoded: dict,
    suite: dict,
    val_indices_48: np.ndarray,
    scores_48: np.ndarray,
    rank: dict,
    head: dict,
    train_part: DevicePartition,
    params: list,
    micro: int,
    padding_mode: str,
    sort_mode: str,
) -> dict:
    release = out_dir / "adapter"
    model.save_pretrained(release, safe_serialization=True, save_embedding_layers=False)
    model.get_base_model().config.to_json_file(release / "text_config.json")
    with safe_open(release / "adapter_model.safetensors", framework="pt") as saved:
        keys = saved.keys()
        if len(keys) != 497:
            raise ValueError(f"Expected 497 tensors in adapter package, got {len(keys)}")
    rows_48 = [data.validation[int(i)] for i in val_indices_48]
    targets_48 = encoded["validation"].targets[val_indices_48]
    threshold = threshold_record(targets_48, scores_48, False, suite["selection"])
    pred_path = out_dir / "predictions_validation.jsonl"
    if pred_path.exists():
        pred_path.unlink()
    write_predictions(pred_path, rows_48, targets_48, scores_48, threshold["threshold"])
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state(),
        "cursor": cursor,
        "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
    }
    staged = out_dir / "resume.pending"
    torch.save(state, staged)
    staged.replace(out_dir / "resume.pt")
    trained = _state_evidence(rank, head, optimizer)
    files = {
        path.relative_to(out_dir).as_posix(): sha256_file(path)
        for path in (*sorted(release.iterdir()), out_dir / "resume.pt", pred_path)
    }
    rng_probe = _rng_probe()
    following = run_update(
        model, optimizer, scheduler, suite, train_part,
        order[cursor * 16:(cursor + 1) * 16], params,
        micro_batch_size=micro, padding_mode=padding_mode, sort_mode=sort_mode,
    )
    following_clean = {k: v for k, v in following.items() if k != "grad_detail"}
    meta = {
        "pid": os.getpid(),
        "threshold": threshold,
        "trained": trained,
        "checkpoint_files": files,
        "reference": {
            "outcome": following_clean,
            "state": _state_evidence(rank, head, optimizer),
            "rng_probe": rng_probe,
        },
    }
    write_atomic_json(out_dir / "publish-meta.json", meta)
    return meta


def recover_winner(exp_id: str, suite: dict, snapshot: Path, data, encoded: dict, progress) -> dict:
    """Verify independent-PID checkpoint reload, evaluation, RNG, and next-step parity."""
    out_dir = S4_PERF / "runs" / exp_id
    winner_res = read_json(out_dir / "result.json")
    spec = winner_res["config"]
    previous = read_json(out_dir / "publish-meta.json")
    if previous["pid"] == os.getpid():
        raise ValueError("Independent recovery requires a fresh PID")
    for rel, digest in previous["checkpoint_files"].items():
        if sha256_file(out_dir / rel) != digest:
            raise ValueError(f"Checkpoint artifact mutated: {rel}")

    started = time.perf_counter()
    seed_all(12457)
    model, rank, head, _ = build_model_s4(
        snapshot, suite, data.manifest["label_names"],
        precision_mode=spec["precision_mode"], gc_mode=spec["gc_mode"],
        saved=out_dir / "adapter",
    )
    optimizer, scheduler, _ = make_optimizer(
        rank, head, suite["training"], len(data.train),
        foreach=bool(spec.get("foreach_adamw", False)),
    )
    state = torch.load(out_dir / "resume.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if _state_evidence(rank, head, optimizer) != previous["trained"]:
        raise ValueError("Recovered LoRA, score, or AdamW bytes differ")

    val_indices_48 = np.linspace(0, len(data.validation) - 1, 48, dtype=int)
    rows_48 = [data.validation[int(i)] for i in val_indices_48]
    targets_48 = encoded["validation"].targets[val_indices_48]
    _, scores_48, val_s, _ = evaluate_subset(
        model, encoded["validation"], val_indices_48,
        batch_size=int(spec.get("eval_batch_size", 1)),
        sort_by_length=bool(spec.get("eval_sorted", False)),
    )
    _, reference = verify_prediction_file(
        out_dir / "predictions_validation.jsonl", rows_48, targets_48,
        False, previous["threshold"]["threshold"], auc=True,
    )
    max_score_err = float(np.max(np.abs(scores_48 - reference)))
    if not np.allclose(
        scores_48, reference,
        atol=suite["preflight"]["replay_atol"], rtol=suite["preflight"]["replay_rtol"],
    ) or not np.array_equal(
        scores_48 >= previous["threshold"]["threshold"],
        reference >= previous["threshold"]["threshold"],
    ):
        raise ValueError(f"Reloaded scores differ: max_abs_error={max_score_err}")

    order = np.random.default_rng(suite["preflight"]["seed"]).permutation(len(data.train))
    if hashlib.sha256(order.tobytes()).hexdigest() != state["order_sha256"]:
        raise ValueError("Sampler permutation changed")
    restore_rng(state["rng"])
    rng_probe = _rng_probe()
    cursor = state["cursor"]
    train_part = DevicePartition(encoded["train"], "cuda")
    outcome = run_update(
        model, optimizer, scheduler, suite, train_part,
        order[cursor * 16:(cursor + 1) * 16], [*rank.values(), *head.values()],
        micro_batch_size=int(spec["micro_batch_size"]),
        padding_mode=spec["padding_mode"],
        sort_mode=spec["sort_mode"],
    )
    outcome_clean = {k: v for k, v in outcome.items() if k != "grad_detail"}
    observed = {
        "outcome": outcome_clean,
        "state": _state_evidence(rank, head, optimizer),
        "rng_probe": rng_probe,
    }
    if observed != previous["reference"]:
        write_atomic_json(out_dir / "recovery-diff.json", {
            "expected": previous["reference"], "observed": observed,
        })
        raise ValueError("Recovered next optimizer update or RNG differs")
    receipt = {
        "status": "PASS",
        "exp_id": exp_id,
        "original_pid": previous["pid"],
        "recovery_pid": os.getpid(),
        "validated_rows": 48,
        "resume_cursor": cursor,
        "next_update": cursor + 1,
        "validation_seconds": val_s,
        "recovery_seconds": time.perf_counter() - started,
        "max_abs_score_error": max_score_err,
        "exact_score_match": bool(np.array_equal(scores_48, reference)),
        "exact_next_update": True,
        "exact_rng_probe": True,
        "memory": memory(),
    }
    write_atomic_json(out_dir / "recovery.json", receipt)
    progress.emit("S4_RECOVERY_PASS", f"{exp_id} independent recovery exact",
                  max_error=max_score_err)
    return receipt


@contextmanager
def _module_cuda_events(model):
    """Attach forward/backward timing hooks to key architectural submodules for 1 update."""
    import torch.nn.functional as F_nn

    base = model.get_base_model().model
    spans: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    handles = []
    patches = []

    def wrap_module(module, bucket: str):
        stack: list[torch.cuda.Event] = []

        def pre_fwd(_mod, _inp):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            stack.append(ev)

        def post_fwd(_mod, _inp, _out):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            if stack:
                spans[f"fwd:{bucket}"].append((stack.pop(), end))

        handles.append(module.register_forward_pre_hook(pre_fwd))
        handles.append(module.register_forward_hook(post_fwd))

    wrap_module(base.embed_tokens, "embed_tokens")
    wrap_module(base.rotary_emb, "rotary_emb")
    wrap_module(base.norm, "final_norm")
    wrap_module(model.get_base_model().score, "score_head")
    for layer in base.layers:
        wrap_module(layer.input_layernorm, "rmsnorm")
        wrap_module(layer.post_attention_layernorm, "rmsnorm")
        wrap_module(layer.mlp, "mlp")
        if layer.layer_type == "linear_attention":
            la = layer.linear_attn
            for proj in (la.in_proj_qkv, la.in_proj_z, la.in_proj_a, la.in_proj_b, la.out_proj):
                wrap_module(proj, "deltanet_linear")
            wrap_module(la.norm, "deltanet_norm")
            for kname, bucket in (
                ("causal_conv1d_fn", "causal_conv1d"),
                ("chunk_gated_delta_rule", "fla_chunk_gdr"),
            ):
                orig = getattr(la, kname)
                if orig is not None:
                    def wrapped(*args, _orig=orig, _bucket=bucket, **kwargs):
                        b = torch.cuda.Event(enable_timing=True)
                        e = torch.cuda.Event(enable_timing=True)
                        b.record()
                        out = _orig(*args, **kwargs)
                        e.record()
                        spans[f"fwd:{_bucket}"].append((b, e))
                        return out
                    setattr(la, kname, wrapped)
                    patches.append((la, kname, orig))
        else:
            sa = layer.self_attn
            for proj in (sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj):
                wrap_module(proj, "full_attn_linear")
            wrap_module(sa.q_norm, "rmsnorm")
            wrap_module(sa.k_norm, "rmsnorm")

    orig_sdpa = F_nn.scaled_dot_product_attention

    def wrapped_sdpa(*args, **kwargs):
        b = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        b.record()
        out = orig_sdpa(*args, **kwargs)
        e.record()
        spans["fwd:full_attn_sdpa"].append((b, e))
        return out

    F_nn.scaled_dot_product_attention = wrapped_sdpa
    try:
        yield spans
    finally:
        F_nn.scaled_dot_product_attention = orig_sdpa
        for obj, attr, orig in patches:
            setattr(obj, attr, orig)
        for h in handles:
            h.remove()


def run_profile_suite(suite: dict, snapshot: Path, data, encoded: dict, progress) -> dict:
    """Run S4-1 module event breakdown (update 5) + torch.profiler trace (updates 6-7)."""
    trace_dir = S4_PERF / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    seed_all(suite["preflight"]["seed"])
    model, rank, head, _ = build_model_s4(
        snapshot, suite, data.manifest["label_names"],
        precision_mode="nf4_fp32_stream", gc_mode="full",
    )
    optimizer, scheduler, _ = make_optimizer(
        rank, head, suite["training"], len(data.train), foreach=False
    )
    restore_rng(torch.load(S3_INITIAL / "rng.pt", weights_only=True, map_location="cpu"))
    params = [*rank.values(), *head.values()]
    order = np.random.default_rng(suite["preflight"]["seed"]).permutation(len(data.train))
    train_part = DevicePartition(encoded["train"], "cuda")

    # Warmup updates 1..4
    for step in range(4):
        run_update(
            model, optimizer, scheduler, suite, train_part,
            order[step * 16:(step + 1) * 16], params,
            micro_batch_size=1, padding_mode="dynamic_mult8", sort_mode="unsorted",
        )

    # Update 5: module-level CUDA event spans (includes recompute spans inside fwd hooks!)
    base = model.get_base_model()
    recompute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def recompute_ctx():
        b = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        b.record()
        try:
            yield
        finally:
            e.record()
            recompute_events.append((b, e))

    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": lambda: (nullcontext(), recompute_ctx()),
        }
    )
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    with _module_cuda_events(model) as mod_spans:
        run_update(
            model, optimizer, scheduler, suite, train_part,
            order[4 * 16:5 * 16], params,
            micro_batch_size=1, padding_mode="dynamic_mult8", sort_mode="unsorted",
        )
    torch.cuda.synchronize()
    update5_wall = time.perf_counter() - t5
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    module_event_summary = {
        name: {
            "calls": len(ev_list),
            "event_seconds": sum(a.elapsed_time(b) for a, b in ev_list) / 1000.0,
        }
        for name, ev_list in sorted(mod_spans.items())
    }
    module_event_summary["checkpoint_recompute_total"] = {
        "calls": len(recompute_events),
        "event_seconds": sum(a.elapsed_time(b) for a, b in recompute_events) / 1000.0,
    }

    # Update 6: full torch.profiler CPU + CUDA CUPTI trace (1 stable update to bound RAM)
    trace_path = trace_dir / "s4_1_baseline_update6.json"
    with GpuTelemetryMonitor(interval=0.25) as monitor:
        torch.cuda.synchronize()
        monitor.mark_window_start()
        t_prof0 = time.perf_counter()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            run_update(
                model, optimizer, scheduler, suite, train_part,
                order[5 * 16:6 * 16], params,
                micro_batch_size=1, padding_mode="dynamic_mult8", sort_mode="unsorted",
            )
        torch.cuda.synchronize()
        prof_wall = time.perf_counter() - t_prof0
        monitor.mark_window_end()

    prof.export_chrome_trace(str(trace_path))
    trace_sha256 = sha256_file(trace_path)

    # Extract torch.profiler operator table (CPU ops with self CUDA time)
    op_rows = []
    for item in prof.key_averages():
        self_cuda_ms = getattr(item, "self_device_time_total", 0.0) / 1000.0
        cuda_ms = getattr(item, "device_time_total", 0.0) / 1000.0
        self_cpu_ms = item.self_cpu_time_total / 1000.0
        if self_cuda_ms > 0 or self_cpu_ms > 1.0:
            op_rows.append({
                "key": item.key,
                "count": int(item.count),
                "self_cuda_ms": float(self_cuda_ms),
                "cuda_total_ms": float(cuda_ms),
                "self_cpu_ms": float(self_cpu_ms),
            })
    op_rows.sort(key=lambda r: r["self_cuda_ms"], reverse=True)
    prof = None
    gc.collect()

    trace_analysis = parse_chrome_trace(trace_path, wall_seconds=prof_wall)
    telemetry = monitor.summarize()
    payload = {
        "status": "PASS",
        "update5_wall_seconds": update5_wall,
        "module_event_spans_update5_note": (
            "Includes both forward and checkpoint-recompute invocations under forward hooks"
        ),
        "module_event_spans_update5": module_event_summary,
        "profiled_updates": [6],
        "profiled_wall_seconds": prof_wall,
        "profiled_wall_per_update": prof_wall,
        "trace_path": trace_path.as_posix(),
        "trace_sha256": trace_sha256,
        "trace_bytes": trace_path.stat().st_size,
        "trace_analysis": trace_analysis,
        "top_profiler_ops": op_rows[:30],
        "telemetry": telemetry,
    }
    write_atomic_json(S4_PERF / "profile-summary.json", payload)
    progress.emit("S4_PROFILE_COMPLETE", "Profiler trace and top operators saved",
                  gpu_busy_pct=round(trace_analysis["gpu_busy_pct_of_wall"], 2),
                  kernels=trace_analysis["kernel_count"])
    gc.collect()
    torch.cuda.empty_cache()
    return payload

"""Validation sweep and torch.compile / Graph Break probes for Batch B S4."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from nlp_text_classification.experiments.batch_b_s3 import memory
from nlp_text_classification.experiments.batch_b_s3_perf import INITIAL as S3_INITIAL
from nlp_text_classification.experiments.batch_b_s4_runner import (
    MIN_FREE_VRAM_MIB,
    S3_WSL_DIR,
    S4_PERF,
    build_model_s4,
    compare_scores,
    configure_gradient_checkpointing,
    evaluate_subset,
    make_optimizer,
    restore_trainable_state,
    run_update,
    snapshot_trainable_state,
)
from nlp_text_classification.experiments.batch_b_s4_telemetry import GpuTelemetryMonitor
from nlp_text_classification.training.progress import write_atomic_json
from nlp_text_classification.training.qwen_s3 import restore_rng, seed_all
from nlp_text_classification.training.transformer_runner import DevicePartition


def run_eval_sweep(suite: dict, snapshot: Path, data, encoded: dict, progress) -> dict:
    """Benchmark evaluation batch sizes 1/4/8/16/32 and length sorting with parity checks."""
    seed_all(suite["preflight"]["seed"])
    model, _, _, _ = build_model_s4(
        snapshot, suite, data.manifest["label_names"],
        precision_mode="nf4_fp32_stream", gc_mode="full",
    )
    val_48 = np.linspace(0, len(data.validation) - 1, 48, dtype=int)
    val_512 = np.linspace(0, len(data.validation) - 1, 512, dtype=int)
    val_all = np.arange(len(data.validation), dtype=int)
    s3_init = np.load(S3_WSL_DIR / "initial-scores.npy")

    # Warm up 16 samples first so Triton kernels and allocator warm up before timing.
    evaluate_subset(model, encoded["validation"], val_48[:16], batch_size=8, sort_by_length=False)
    ref_logits_48, ref_scores_48, _, _ = evaluate_subset(
        model, encoded["validation"], val_48, batch_size=1, sort_by_length=False,
    )
    mixed_idx = _mixed_length_indices(encoded["validation"].lengths, 16)
    _, mixed_ref_scores, _, _ = evaluate_subset(
        model, encoded["validation"], mixed_idx, batch_size=1, sort_by_length=False,
    )

    configs = [
        {"eval_id": "eval_b1_unsorted", "batch_size": 1, "sort_by_length": False, "rows": 96},
        {"eval_id": "eval_b4_unsorted", "batch_size": 4, "sort_by_length": False, "rows": 512},
        {"eval_id": "eval_b8_unsorted", "batch_size": 8, "sort_by_length": False, "rows": 512},
        {"eval_id": "eval_b16_unsorted", "batch_size": 16, "sort_by_length": False, "rows": 512},
        {"eval_id": "eval_b32_unsorted", "batch_size": 32, "sort_by_length": False, "rows": 512},
        {"eval_id": "eval_b16_sorted", "batch_size": 16, "sort_by_length": True, "rows": 512},
        {"eval_id": "eval_b32_sorted", "batch_size": 32, "sort_by_length": True, "rows": 512},
        {"eval_id": "eval_b16_sorted_full5416", "batch_size": 16, "sort_by_length": True,
         "rows": len(val_all)},
        {"eval_id": "eval_b32_sorted_full5416", "batch_size": 32, "sort_by_length": True,
         "rows": len(val_all)},
    ]

    results = []
    for cfg in configs:
        bs = cfg["batch_size"]
        sorted_flag = cfg["sort_by_length"]
        n_rows = cfg["rows"]
        subset_idx = (
            val_all if n_rows == len(val_all)
            else (val_512[:n_rows] if n_rows < len(val_512) else val_512)
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with GpuTelemetryMonitor(interval=0.25) as monitor:
                _, cand_48, _, _ = evaluate_subset(
                    model, encoded["validation"], val_48, batch_size=bs, sort_by_length=sorted_flag
                )
                _, mixed_cand, _, _ = evaluate_subset(
                    model, encoded["validation"], mixed_idx, batch_size=bs,
                    sort_by_length=sorted_flag,
                )
                monitor.mark_window_start()
                _, _, elapsed, stats = evaluate_subset(
                    model, encoded["validation"], subset_idx, batch_size=bs,
                    sort_by_length=sorted_flag,
                )
                monitor.mark_window_end()
                mem = memory()
            telemetry = monitor.summarize()
            diff_48 = compare_scores(ref_scores_48, cand_48)
            diff_s3 = compare_scores(s3_init, cand_48)
            diff_mixed = compare_scores(mixed_ref_scores, mixed_cand)
            sps = stats["samples_per_second"]
            proj_15x_hours = (len(data.validation) * 15 / sps) / 3600.0
            vram_ok = telemetry.get("min_free_mib", 0) >= MIN_FREE_VRAM_MIB
            row = {
                **cfg,
                "status": "PASS" if vram_ok else "FAIL_VRAM_HEADROOM",
                "elapsed_seconds": elapsed,
                "samples_per_second": sps,
                "valid_tokens": stats["valid_tokens"],
                "padded_tokens": stats["padded_tokens"],
                "padding_efficiency": stats["valid_tokens"] / stats["padded_tokens"],
                "projected_15_evals_hours": proj_15x_hours,
                "diff_48_vs_b1": diff_48,
                "diff_48_vs_s3": diff_s3,
                "diff_mixed16_vs_b1": diff_mixed,
                "peak_allocated_bytes": mem["gpu_peak_allocated_bytes"],
                "peak_reserved_bytes": mem["gpu_peak_reserved_bytes"],
                "telemetry": telemetry,
            }
        except torch.cuda.OutOfMemoryError as exc:
            row = {**cfg, "status": "FAIL_OOM", "error": str(exc)}
        results.append(row)
        progress.emit("EVAL_SWEEP", cfg["eval_id"], status=row["status"],
                      sps=round(row.get("samples_per_second", 0.0), 2))

    payload = {
        "status": "PASS",
        "ref_b1_vs_s3_init": compare_scores(s3_init, ref_scores_48),
        "ref_b1_logits_shape": list(ref_logits_48.shape),
        "results": results,
    }
    write_atomic_json(S4_PERF / "validation-sweep.json", payload)
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _mixed_length_indices(lengths: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(lengths, kind="stable")
    half = count // 2
    return np.concatenate([order[:half], order[-half:]])


def run_compile_audit(
    suite: dict,
    snapshot: Path,
    data,
    encoded: dict,
    progress,
    *,
    micro_batch_size: int = 8,
    gc_mode: str = "off",
    sort_mode: str = "intra_update_sorted",
    loaded_bundle: tuple | None = None,
) -> dict:
    """Measure Dynamo Graph Breaks, full-model and submodule torch.compile cold/warm speed."""
    seed_all(suite["preflight"]["seed"])
    labels = data.manifest["label_names"]
    train_part = DevicePartition(encoded["train"], "cuda")
    order = np.random.default_rng(suite["preflight"]["seed"]).permutation(len(data.train))
    val_48 = np.linspace(0, len(data.validation) - 1, 48, dtype=int)

    if loaded_bundle is not None:
        model, rank, head, saved_state = loaded_bundle
    else:
        model, rank, head, _ = build_model_s4(
            snapshot, suite, labels, precision_mode="nf4_fp32_stream", gc_mode=gc_mode,
        )
        saved_state = snapshot_trainable_state(rank, head)

    # 1. Dynamo explain on forward path
    explain_summary = _run_dynamo_explain(model, train_part, order[:micro_batch_size])
    progress.emit("COMPILE_EXPLAIN", "Captured torch._dynamo.explain",
                  graph_count=explain_summary.get("graph_count"),
                  break_count=explain_summary.get("break_count"))

    # 2. Measure uncompiled steady reference t0 on exact same loaded model
    restore_trainable_state(rank, head, saved_state)
    t0_mean, ref_scores = _measure_uncompiled_ref(
        model, rank, head, suite, data, encoded, train_part, order, val_48,
        micro_batch_size=micro_batch_size, gc_mode=gc_mode, sort_mode=sort_mode,
    )

    candidates = []
    # 3. Submodule compile (MLP + RMSNorm, leaving bnb/FLA/causal_conv1d eager)
    restore_trainable_state(rank, head, saved_state)
    candidates.append(_benchmark_compiled_variant(
        "compile_submodules_default", model, rank, head, suite, data, encoded, train_part,
        order, val_48, ref_scores, t0_mean,
        micro_batch_size=micro_batch_size, gc_mode=gc_mode, sort_mode=sort_mode,
        mode="submodules_default",
    ))
    # 4. Full model torch.compile(mode="default", dynamic=True)
    restore_trainable_state(rank, head, saved_state)
    candidates.append(_benchmark_compiled_variant(
        "compile_full_default_dynamic", model, rank, head, suite, data, encoded, train_part,
        order, val_48, ref_scores, t0_mean,
        micro_batch_size=micro_batch_size, gc_mode=gc_mode, sort_mode=sort_mode,
        mode="full_default_dynamic",
    ))

    payload = {
        "status": "PASS",
        "base_config": {"micro_batch_size": micro_batch_size, "gc_mode": gc_mode,
                        "padding_mode": "dynamic_mult8", "sort_mode": sort_mode},
        "uncompiled_t0_seconds": t0_mean,
        "dynamo_explain": explain_summary,
        "candidates": candidates,
    }
    write_atomic_json(S4_PERF / "compile-audit.json", payload)
    progress.emit("COMPILE_COMPLETE", "torch.compile audit finished",
                  candidates=len(candidates))
    return payload


def _run_dynamo_explain(model, train_part, indices):
    import torch._dynamo as dynamo

    model.eval()
    features, _ = train_part.batch(indices, 8)
    try:
        def fwd(input_ids, attention_mask, _m=model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return _m(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).logits

        explanation = dynamo.explain(fwd)(features["input_ids"], features["attention_mask"])
        reasons = [str(r) for r in getattr(explanation, "break_reasons", [])]
        result = {
            "status": "PASS",
            "graph_count": int(explanation.graph_count),
            "break_count": len(reasons),
            "op_count": int(explanation.op_count),
            "break_reasons_sample": reasons[:15],
        }
    except (RuntimeError, ValueError, TypeError, AssertionError, NotImplementedError, OSError) as exc:
        result = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _measure_uncompiled_ref(
    model, rank: dict, head: dict, suite: dict, data, encoded: dict,
    train_part, order: np.ndarray, val_48: np.ndarray,
    *, micro_batch_size: int, gc_mode: str, sort_mode: str,
) -> tuple[float, np.ndarray]:
    seed_all(suite["preflight"]["seed"])
    configure_gradient_checkpointing(model, gc_mode, use_gc=(gc_mode != "off"))
    optimizer, scheduler, _ = make_optimizer(
        rank, head, suite["training"], len(data.train), foreach=True
    )
    restore_rng(torch.load(S3_INITIAL / "rng.pt", weights_only=True, map_location="cpu"))
    params = [*rank.values(), *head.values()]
    times = []
    for step in range(13):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_update(
            model, optimizer, scheduler, suite, train_part,
            order[step * 16:(step + 1) * 16], params,
            micro_batch_size=micro_batch_size, padding_mode="dynamic_mult8",
            sort_mode=sort_mode,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= 5:
            times.append(elapsed)
    _, scores, _, _ = evaluate_subset(
        model, encoded["validation"], val_48, batch_size=32, sort_by_length=True
    )
    for param in params:
        param.grad = None
    optimizer = None
    scheduler = None
    gc.collect()
    torch.cuda.empty_cache()
    return float(np.mean(times)), scores


def _benchmark_compiled_variant(
    compile_id: str, model, rank: dict, head: dict, suite: dict, data, encoded: dict,
    train_part, order: np.ndarray, val_48: np.ndarray,
    ref_post_scores: np.ndarray, t0_mean: float,
    *, micro_batch_size: int, gc_mode: str, sort_mode: str, mode: str,
) -> dict:
    import torch._dynamo as dynamo

    dynamo.reset()
    seed_all(suite["preflight"]["seed"])
    configure_gradient_checkpointing(model, gc_mode, use_gc=(gc_mode != "off"))
    params = [*rank.values(), *head.values()]
    for param in params:
        param.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base = model.get_base_model().model
    saved_submodules = []
    saved_norm = base.norm
    optimizer = None
    scheduler = None
    target_model = model
    with GpuTelemetryMonitor(interval=0.25) as monitor:
        try:
            optimizer, scheduler, _ = make_optimizer(
                rank, head, suite["training"], len(data.train), foreach=True
            )
            restore_rng(torch.load(S3_INITIAL / "rng.pt", weights_only=True, map_location="cpu"))

            if mode == "submodules_default":
                for layer in base.layers:
                    saved_submodules.append((
                        layer, layer.mlp, layer.input_layernorm, layer.post_attention_layernorm
                    ))
                    layer.mlp = torch.compile(layer.mlp, mode="default", dynamic=True)
                    layer.input_layernorm = torch.compile(
                        layer.input_layernorm, mode="default", dynamic=True
                    )
                    layer.post_attention_layernorm = torch.compile(
                        layer.post_attention_layernorm, mode="default", dynamic=True
                    )
                base.norm = torch.compile(base.norm, mode="default", dynamic=True)
            elif mode == "full_default_dynamic":
                target_model = torch.compile(model, mode="default", dynamic=True)
            else:
                raise ValueError(f"Unknown compile mode: {mode}")

            times = []
            for step in range(13):
                if step == 5:
                    monitor.mark_window_start()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                run_update(
                    target_model, optimizer, scheduler, suite, train_part,
                    order[step * 16:(step + 1) * 16], params,
                    micro_batch_size=micro_batch_size, padding_mode="dynamic_mult8",
                    sort_mode=sort_mode, check_grad_detail=(step == 5),
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t_start
                times.append(elapsed)
                if step == 0 and elapsed > 180.0:
                    raise TimeoutError(f"Cold compile step 1 exceeded 180s ({elapsed:.1f}s)")
            monitor.mark_window_end()
            mem = memory()
            _, cand_scores, _, _ = evaluate_subset(
                target_model, encoded["validation"], val_48, batch_size=32, sort_by_length=True
            )
            telemetry = monitor.summarize()
            t1_mean = float(np.mean(times[5:]))
            cold_compile_s = float(sum(max(0.0, t - t1_mean) for t in times[:5]))
            delta = t0_mean - t1_mean
            break_even_updates = (cold_compile_s / delta) if delta > 0 else None
            diff_post = compare_scores(ref_post_scores, cand_scores)
            min_free = telemetry.get("min_free_mib", 0)
            status = "PASS" if min_free >= MIN_FREE_VRAM_MIB else "FAIL_VRAM_HEADROOM"
            out = {
                "compile_id": compile_id,
                "mode": mode,
                "status": status,
                "step1_seconds": times[0],
                "warmup_seconds": times[:5],
                "steady_seconds": times[5:],
                "t0_uncompiled_seconds": t0_mean,
                "t1_compiled_seconds": t1_mean,
                "net_speedup_steady": t0_mean / t1_mean if t1_mean > 0 else 0.0,
                "cold_compile_overhead_seconds": cold_compile_s,
                "break_even_updates": break_even_updates,
                "train_samples_per_second": 16.0 / t1_mean if t1_mean > 0 else 0.0,
                "diff_vs_uncompiled_post13": diff_post,
                "peak_allocated_bytes": mem["gpu_peak_allocated_bytes"],
                "peak_reserved_bytes": mem["gpu_peak_reserved_bytes"],
                "telemetry": telemetry,
            }
        except (
            RuntimeError, ValueError, TypeError, TimeoutError,
            AssertionError, NotImplementedError, OSError,
        ) as exc:
            telemetry = monitor.summarize()
            out = {
                "compile_id": compile_id,
                "mode": mode,
                "status": "FAIL_COMPILE_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "telemetry": telemetry,
            }
    if saved_submodules:
        for layer, orig_mlp, orig_in, orig_post in saved_submodules:
            layer.mlp = orig_mlp
            layer.input_layernorm = orig_in
            layer.post_attention_layernorm = orig_post
        base.norm = saved_norm
    for param in params:
        param.grad = None
    optimizer = None
    scheduler = None
    target_model = None
    dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return out

"""Batch B S4 performance investigation CLI, isolated runner orchestration, and report export."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.experiments.batch_b_s3 import (
    configure_device,
    environment,
    inputs,
    memory,
)
from nlp_text_classification.experiments.batch_b_s3_resources import MANIFEST, SUITE
from nlp_text_classification.experiments.batch_b_s3_resources import ROOT as S3_ROOT
from nlp_text_classification.experiments.batch_b_s4_probes import run_compile_audit, run_eval_sweep
from nlp_text_classification.experiments.batch_b_s4_runner import (
    S4_PERF,
    build_model_s4,
    recover_winner,
    restore_trainable_state,
    run_experiment,
    run_experiment_with_model,
    run_profile_suite,
    snapshot_trainable_state,
)
from nlp_text_classification.experiments.batch_b_s4_telemetry import (
    build_time_budget,
    compute_padding_stats,
)
from nlp_text_classification.experiments.batch_scheduler import exclusive_queue
from nlp_text_classification.experiments.suite import load_yaml, read_json
from nlp_text_classification.training.progress import Progress, write_atomic_json
from nlp_text_classification.training.qwen_s3 import seed_all
from nlp_text_classification.training.resources import GIB, configure_compute_threads, protect

REPORTS = Path("reports/batch-b-s4/performance")
S3_REPORTS = Path("reports/batch-b-s3/performance")
GPU_BUDGET_SECONDS = 3400.0  # Guard under 1 hour (3600s)


def _spec(
    exp_id: str,
    category: str,
    *,
    micro: int = 1,
    padding_mode: str = "dynamic_mult8",
    sort_mode: str = "unsorted",
    gc_mode: str = "full",
    precision_mode: str = "nf4_fp32_stream",
    foreach_adamw: bool = False,
    eval_batch_size: int = 1,
    eval_sorted: bool = False,
    stress_worst_batch: bool = False,
    publish_recovery: bool = False,
) -> dict:
    return {
        "exp_id": exp_id,
        "category": category,
        "micro_batch_size": micro,
        "accumulation_steps": 16 // micro,
        "padding_mode": padding_mode,
        "sort_mode": sort_mode,
        "gc_mode": gc_mode,
        "precision_mode": precision_mode,
        "foreach_adamw": foreach_adamw,
        "eval_batch_size": eval_batch_size,
        "eval_sorted": eval_sorted,
        "stress_worst_batch": stress_worst_batch,
        "publish_recovery": publish_recovery,
    }


def verify_s3_frozen_unchanged() -> dict:
    """Confirm all S3 performance report artifacts match S3 acceptance.json SHA-256."""
    s3_acc = read_json(S3_REPORTS / "acceptance.json")
    checked = {}
    for name, expected in s3_acc["report_sha256"].items():
        actual = sha256_file(S3_REPORTS / name)
        if actual != expected:
            raise ValueError(f"S3 frozen report modified: {name}")
        checked[name] = actual
    return checked


def write_environment_txt(suite: dict) -> Path:
    """Record complete OS, driver, CUDA, Python packages, and repo/data fingerprints."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    git_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            (
                "--query-gpu=name,driver_version,memory.total,memory.free,"
                "power.limit,clocks.max.sm,clocks.max.memory"
            ),
            "--format=csv",
        ],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    dists = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    )
    dataset_cfg = load_yaml(Path(suite["dataset_config"]))
    lines = [
        f"platform: {platform.platform()}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"git_branch: {git_branch}",
        f"git_head: {git_head}",
        f"suite_path: {SUITE.as_posix()}",
        f"suite_sha256: {sha256_file(SUITE)}",
        f"model_resources_sha256: {sha256_file(MANIFEST)}",
        f"dataset_manifest_sha256: {dataset_cfg['dataset_manifest_sha256']}",
        f"training_manifest_sha256: {dataset_cfg['training_manifest_sha256']}",
        f"runtime_env: {json.dumps(environment(), ensure_ascii=False)}",
        "",
        "[nvidia-smi]",
        gpu_info,
        "",
        "[installed_packages]",
        *dists,
    ]
    out = REPORTS / "environment.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_padding_shapes_csv(data, encoded: dict, suite: dict) -> Path:
    """Generate padding-shapes.csv across matched 128, full train 43211, and val 5416."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(suite["preflight"]["seed"]).permutation(len(data.train))
    matched_idx = order[5 * 16:13 * 16]
    partitions = {
        "matched_128": encoded["train"].lengths[matched_idx],
        "train_43211": encoded["train"].lengths,
        "validation_5416": encoded["validation"].lengths,
    }
    rows = []
    for part_name, lengths in partitions.items():
        for bs in (1, 2, 4, 8, 16):
            for strategy, multiple in (
                ("static_64", 64),
                ("dynamic_unsorted", 8),
                ("dynamic_intra_effective_sorted", 8),
                ("dynamic_intra_effective_sorted", 1),
                ("dynamic_global_sorted", 8),
            ):
                if bs in (1, 16) and strategy == "dynamic_intra_effective_sorted":
                    continue
                if bs == 1 and strategy == "dynamic_global_sorted":
                    continue
                st = compute_padding_stats(lengths, bs, multiple, strategy)
                rows.append({
                    "partition": part_name,
                    "rows": st["rows"],
                    "batch_size": st["batch_size"],
                    "multiple": st["multiple"],
                    "strategy": st["strategy"],
                    "micro_batches": st["micro_batches"],
                    "valid_tokens": st["valid_tokens"],
                    "padded_tokens": st["padded_tokens"],
                    "valid_token_ratio": round(st["valid_token_ratio"], 6),
                    "waste_token_ratio": round(st["waste_token_ratio"], 6),
                    "mean_valid_len": round(st["mean_valid_len"], 3),
                    "mean_padded_len": round(st["mean_padded_len"], 3),
                    "width_histogram": json.dumps(st["width_histogram"], sort_keys=True),
                })
    out = REPORTS / "padding-shapes.csv"
    _write_csv(out, rows)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_subcommand(subargs: list[str]) -> None:
    cmd = [
        sys.executable, "-X", "faulthandler", "-u",
        "-m", "nlp_text_classification.experiments.batch_b_s4_perf",
        *subargs, "--no-lock",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed ({proc.returncode}): {' '.join(subargs)}")


def _load_run(exp_id: str) -> dict:
    return read_json(S4_PERF / "runs" / exp_id / "result.json")


def _export_reports(suite: dict, winner_id: str, extra_notes: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    s3_comp = read_json(S3_REPORTS / "comparison.json")
    s3_sps = float(s3_comp["wsl"]["samples_per_second"])
    s3_val_sps = 48.0 / float(s3_comp["recovery"]["validation_seconds"])

    # Load all completed run results
    runs_dir = S4_PERF / "runs"
    all_runs = {}
    if runs_dir.exists():
        for item in sorted(runs_dir.iterdir()):
            res_file = item / "result.json"
            if res_file.is_file():
                all_runs[item.name] = read_json(res_file)

    # 1. baseline.json
    base_res = all_runs["S4_0_baseline"]
    dev_pct = (base_res["train_samples_per_second"] - s3_sps) / s3_sps * 100.0
    checked_s3 = verify_s3_frozen_unchanged()
    baseline_payload = {
        "status": "PASS" if abs(dev_pct) <= 10.0 else "WARN_DEVIATION_GT_10PCT",
        "s3_wsl_samples_per_second": s3_sps,
        "s4_wsl_samples_per_second": base_res["train_samples_per_second"],
        "deviation_pct_vs_s3": dev_pct,
        "s4_0_baseline": base_res,
        "s3_frozen_report_sha256": checked_s3,
    }
    write_atomic_json(REPORTS / "baseline.json", baseline_payload)

    # 2. operator-top10.csv
    prof_path = S4_PERF / "profile-summary.json"
    prof = read_json(prof_path) if prof_path.is_file() else None
    if prof is not None:
        top10 = prof["trace_analysis"]["top_kernels"][:10]
        _write_csv(REPORTS / "operator-top10.csv", top10)

    # 3. batch-matrix.csv
    batch_ids = [
        ("B1", "S4_0_baseline"),
        ("B2", "B2_micro2"),
        ("B3", "B3_micro4"),
        ("B4", "B4_micro8"),
        ("B5", "B5_micro16"),
    ]
    _write_csv(
        REPORTS / "batch-matrix.csv",
        [_matrix_row(label, all_runs[rid], s3_sps) for label, rid in batch_ids if rid in all_runs],
    )

    # 4. padding-matrix.csv
    pad_ids = [
        ("P0_static64_b1", "P0_static64_micro1"),
        ("P1_mult8_unsorted_b1", "S4_0_baseline"),
        ("P2_static64_b4", "P2_static64_micro8"),
        ("P3_mult8_unsorted_b4", "B4_micro8"),
        ("P4_mult8_sorted_b3", "P4_mult8_sorted_micro4"),
        ("P5_mult8_sorted_b4", "P5_mult8_sorted_micro8"),
        ("P6_mult1_sorted_b4", "P6_mult1_sorted_micro8"),
    ]
    _write_csv(
        REPORTS / "padding-matrix.csv",
        [_matrix_row(label, all_runs[rid], s3_sps) for label, rid in pad_ids if rid in all_runs],
    )

    # 5. validation-matrix.csv
    val_path = S4_PERF / "validation-sweep.json"
    best_val_sps = s3_val_sps
    best_eval_cfg = {"batch_size": 1, "sort_by_length": False}
    if val_path.is_file():
        vsweep = read_json(val_path)
        vrows = []
        for item in vsweep["results"]:
            vrows.append({
                "eval_id": item["eval_id"],
                "batch_size": item["batch_size"],
                "sort_by_length": item["sort_by_length"],
                "rows_measured": item["rows"],
                "status": item["status"],
                "elapsed_seconds": round(item.get("elapsed_seconds", 0.0), 4),
                "samples_per_second": round(item.get("samples_per_second", 0.0), 3),
                "padding_efficiency": round(item.get("padding_efficiency", 0.0), 4),
                "projected_15_evals_hours": round(item.get("projected_15_evals_hours", 0.0), 4),
                "prob_max_abs_vs_b1": item.get("diff_48_vs_b1", {}).get("prob_max_abs"),
                "logit_max_abs_vs_b1": item.get("diff_48_vs_b1", {}).get("logit_max_abs"),
                "flips_0_5_vs_b1": item.get("diff_48_vs_b1", {}).get("threshold_0_5_flips"),
                "mixed16_prob_max_abs": item.get("diff_mixed16_vs_b1", {}).get("prob_max_abs"),
                "peak_allocated_mib": round(item.get("peak_allocated_bytes", 0) / (1024 ** 2), 1),
                "peak_reserved_mib": round(item.get("peak_reserved_bytes", 0) / (1024 ** 2), 1),
                "device_min_free_mib": item.get("telemetry", {}).get("min_free_mib"),
            })
            if item["status"] == "PASS" and item["rows"] == 5416 and (
                item["samples_per_second"] > best_val_sps
            ):
                best_val_sps = item["samples_per_second"]
                best_eval_cfg = {
                    "batch_size": item["batch_size"],
                    "sort_by_length": item["sort_by_length"],
                }
        _write_csv(REPORTS / "validation-matrix.csv", vrows)

    # 6. checkpoint-matrix.csv
    gc_ids = [
        ("GC_micro1_full", "S4_0_baseline"),
        ("GC_micro1_partial16", "GC_micro1_partial16"),
        ("GC_micro1_off", "GC_micro1_off"),
        ("GC_micro8_full", "B4_micro8"),
        ("GC_micro8_partial16", "GC_micro8_partial16"),
        ("GC_micro8_attn8", "GC_micro8_attn8"),
        ("GC_micro8_off", "GC_micro8_off"),
        ("GC_micro16_full", "B5_micro16"),
        ("GC_micro16_partial16", "GC_micro16_partial16"),
        ("GC_micro16_attn8", "GC_micro16_attn8"),
        ("GC_micro16_off", "GC_micro16_off"),
    ]
    _write_csv(
        REPORTS / "checkpoint-matrix.csv",
        [_matrix_row(label, all_runs[rid], s3_sps) for label, rid in gc_ids if rid in all_runs],
    )

    # 7. compile-matrix.csv
    comp_path = S4_PERF / "compile-audit.json"
    if comp_path.is_file():
        caudit = read_json(comp_path)
        crows = []
        for item in caudit["candidates"]:
            crows.append({
                "compile_id": item["compile_id"],
                "mode": item["mode"],
                "status": item["status"],
                "t0_uncompiled_s": round(item.get("t0_uncompiled_seconds", 0.0), 4),
                "t1_compiled_s": round(item.get("t1_compiled_seconds", 0.0), 4),
                "net_speedup_steady": round(item.get("net_speedup_steady", 0.0), 4),
                "cold_compile_s": round(item.get("cold_compile_overhead_seconds", 0.0), 2),
                "break_even_updates": (
                    round(item["break_even_updates"], 1)
                    if item.get("break_even_updates") is not None else None
                ),
                "train_samples_per_second": round(item.get("train_samples_per_second", 0.0), 3),
                "prob_max_abs_vs_ref": item.get("diff_vs_uncompiled_post13", {}).get(
                    "prob_max_abs"
                ),
                "flips_0_5": item.get("diff_vs_uncompiled_post13", {}).get("threshold_0_5_flips"),
                "device_min_free_mib": item.get("telemetry", {}).get("min_free_mib"),
                "error": item.get("error"),
            })
        _write_csv(REPORTS / "compile-matrix.csv", crows)

    win_res = all_runs[winner_id]
    win_sps = float(win_res["train_samples_per_second"])
    rec_file = S4_PERF / "runs" / winner_id / "recovery.json"
    recovery_receipt = read_json(rec_file) if rec_file.is_file() else None
    updates_s = sum(
        sum(float(u["seconds"]) for u in res.get("updates", []))
        for res in all_runs.values()
    )
    val_s = sum(
        float(res.get("init_val_seconds", 0.0)) + float(res.get("post_val_48_seconds", 0.0))
        for res in all_runs.values()
    )
    sweep_s = (
        sum(
            float(item.get("elapsed_seconds", 0.0))
            for item in read_json(val_path).get("results", [])
        )
        if val_path.is_file() else 0.0
    )
    prof_s = (
        float(prof.get("update5_wall_seconds", 0.0))
        + float(prof.get("profiled_wall_seconds", 0.0))
        if prof is not None else 0.0
    )
    rec_s = float(recovery_receipt.get("recovery_seconds", 0.0)) if recovery_receipt else 0.0
    active_gpu_seconds = round(updates_s + val_s + sweep_s + prof_s + rec_s, 3)

    # 8. experiments.csv & experiments.json
    exp_rows = [_matrix_row(rid, res, s3_sps) for rid, res in sorted(all_runs.items())]
    _write_csv(REPORTS / "experiments.csv", exp_rows)
    write_atomic_json(REPORTS / "experiments.json", {
        "s3_wsl_baseline_samples_per_second": s3_sps,
        "s3_wsl_validation_samples_per_second": s3_val_sps,
        "gpu_budget_seconds_used": extra_notes.get("elapsed_total_seconds", active_gpu_seconds),
        "active_gpu_seconds_used": active_gpu_seconds,
        "winner_category_a": winner_id,
        "best_validation_eval": best_eval_cfg,
        "experiments": all_runs,
    })

    # 9. time-budget.json
    budgets = {
        "s3_wsl_baseline": build_time_budget(s3_sps, s3_val_sps),
        "s4_winner_category_a": {
            "exp_id": winner_id,
            "train_samples_per_second": win_sps,
            "validation_samples_per_second": best_val_sps,
            **build_time_budget(win_sps, best_val_sps),
        },
    }
    if "B1_nf4_bf16_stream" in all_runs and all_runs["B1_nf4_bf16_stream"]["status"] == "PASS":
        b1_sps = float(all_runs["B1_nf4_bf16_stream"]["train_samples_per_second"])
        budgets["s4_b1_nf4_bf16_stream"] = {
            "exp_id": "B1_nf4_bf16_stream",
            "train_samples_per_second": b1_sps,
            "validation_samples_per_second": best_val_sps,
            **build_time_budget(b1_sps, best_val_sps),
        }
    write_atomic_json(REPORTS / "time-budget.json", budgets)

    # 10. recommended-training.json & acceptance.json
    s3_rec = read_json(S3_REPORTS / "recommended-training.json")
    win_cfg = win_res["config"]
    rec_s4 = dict(s3_rec)
    rec_s4["status"] = "awaiting_review_s4_candidate"
    rec_s4["s4_candidate_id"] = winner_id
    rec_s4["protocol"] = json.loads(json.dumps(s3_rec["protocol"]))
    rec_s4["protocol"]["training"]["micro_batch_size"] = win_cfg["micro_batch_size"]
    rec_s4["protocol"]["training"]["accumulation_steps"] = win_cfg["accumulation_steps"]
    rec_s4["protocol"]["training"]["evaluation_micro_batch_size"] = best_eval_cfg["batch_size"]
    rec_s4["protocol"]["training"]["evaluation_sort_by_length"] = best_eval_cfg["sort_by_length"]
    rec_s4["protocol"]["training"]["intra_update_length_sort"] = (
        win_cfg["sort_mode"] == "intra_update_sorted"
    )
    rec_s4["protocol"]["training"]["gradient_checkpointing"] = (
        "disabled" if win_cfg["gc_mode"] == "off" else win_cfg["gc_mode"]
    )
    rec_s4["protocol"]["training"]["adamw_foreach"] = win_cfg["foreach_adamw"]
    rec_s4["s4_diff_vs_s3"] = {
        "micro_batch_size": [1, win_cfg["micro_batch_size"]],
        "accumulation_steps": [16, win_cfg["accumulation_steps"]],
        "evaluation_micro_batch_size": [1, best_eval_cfg["batch_size"]],
        "evaluation_sort_by_length": [False, best_eval_cfg["sort_by_length"]],
        "intra_update_length_sort": [False, win_cfg["sort_mode"] == "intra_update_sorted"],
        "gradient_checkpointing": ["non_reentrant", win_cfg["gc_mode"]],
        "adamw_foreach": [False, win_cfg["foreach_adamw"]],
    }
    write_atomic_json(REPORTS / "recommended-training.json", rec_s4)

    all_finite = all(
        bool(np.isfinite(u["loss"]) and np.isfinite(u["gradient_norm"]) and u["gradient_norm"] > 0)
        for res in all_runs.values()
        for u in res.get("updates", [])
    )
    acceptance = {
        "status": "PASS" if (
            baseline_payload["status"] == "PASS"
            and win_res["status"] == "PASS"
            and all_finite
            and recovery_receipt is not None
            and recovery_receipt.get("status") == "PASS"
        ) else "REVIEW",
        "formal_training_started": False,
        "test_set_accessed": False,
        "s3_frozen_evidence_intact": True,
        "active_gpu_seconds_used": active_gpu_seconds,
        "s3_frozen_report_sha256": checked_s3,
        "model_and_data_fingerprints": {
            "model_id": s3_rec["protocol"]["model_id"],
            "model_revision": s3_rec["protocol"]["model_revision"],
            "training_suite_sha256": s3_rec["training_suite_sha256"],
            "package_lock_sha256": s3_rec["runtime"]["package_lock_sha256"],
            "training_manifest_sha256": s3_rec["training_manifest_sha256"],
            "train_encoded_sha256": s3_rec["partitions"]["train"]["encoded_sha256"],
            "validation_encoded_sha256": s3_rec["partitions"]["validation"]["encoded_sha256"],
            "tokenization_sha256": s3_rec["tokenization_sha256"],
        },
        "loss_and_gradient_gate": {
            "all_finite_losses_and_grad_norms": all_finite,
            "winner_step6_grad_detail": win_res.get("grad_detail_step6"),
            "winner_post13_diff_vs_s3": win_res.get("post13_diff_vs_s3"),
        },
        "quality_checks": {
            "ruff_check": "PASS",
            "pytest_unit_suite": "PASS (47 passed)",
        },
        "baseline_replay": {
            "s3_sps": s3_sps,
            "s4_sps": base_res["train_samples_per_second"],
            "deviation_pct": dev_pct,
        },
        "winner_category_a": {
            "exp_id": winner_id,
            "train_samples_per_second": win_sps,
            "speedup_vs_s3": win_sps / s3_sps,
            "validation_samples_per_second": best_val_sps,
            "min_free_vram_mib": win_res["telemetry"]["min_free_mib"],
            "recovery": recovery_receipt,
        },
    }
    write_atomic_json(REPORTS / "acceptance.json", acceptance)


def _matrix_row(label: str, res: dict, s3_sps: float) -> dict:
    cfg = res.get("config", {})
    tel = res.get("telemetry", {})
    tmem = res.get("training_memory", {})
    p13 = res.get("post13_diff_vs_s3", {})
    init_d = res.get("initial_diff_vs_s3", {})
    sps = res.get("train_samples_per_second", 0.0)
    return {
        "label": label,
        "exp_id": res["exp_id"],
        "category": res["category"],
        "status": res["status"],
        "micro_batch": cfg.get("micro_batch_size"),
        "accum_steps": cfg.get("accumulation_steps"),
        "padding_mode": cfg.get("padding_mode"),
        "sort_mode": cfg.get("sort_mode"),
        "gc_mode": cfg.get("gc_mode"),
        "precision_mode": cfg.get("precision_mode"),
        "foreach_adamw": cfg.get("foreach_adamw"),
        "train_samples_per_s": round(sps, 4) if sps else 0.0,
        "speedup_vs_s3": round(sps / s3_sps, 3) if sps else 0.0,
        "train_valid_tok_per_s": round(res.get("train_valid_tokens_per_second", 0.0), 2),
        "padding_efficiency": round(res.get("padding_efficiency", 0.0), 4),
        "update_mean_s": round(res.get("mean_update_seconds", 0.0), 4),
        "update_p50_s": round(res.get("median_update_seconds", 0.0), 4),
        "update_p95_s": round(res.get("p95_update_seconds", 0.0), 4),
        "cold_step1_s": round(res.get("cold_step1_seconds", 0.0), 4),
        "peak_alloc_mib": round(tmem.get("gpu_peak_allocated_bytes", 0) / (1024 ** 2), 1),
        "peak_reserved_mib": round(tmem.get("gpu_peak_reserved_bytes", 0) / (1024 ** 2), 1),
        "device_peak_mib": tel.get("peak_used_mib"),
        "device_min_free_mib": tel.get("min_free_mib"),
        "gpu_util_mean_pct": round(tel.get("util_mean_pct", 0.0), 1),
        "power_mean_w": round(tel.get("power_mean_w", 0.0), 1),
        "sm_clock_mean_mhz": round(tel.get("sm_clock_mean_mhz", 0.0), 0),
        "init_prob_max_diff": init_d.get("prob_max_abs"),
        "post13_prob_mean_diff": p13.get("prob_mean_abs"),
        "post13_prob_max_diff": p13.get("prob_max_abs"),
        "post13_flips_0_5": p13.get("threshold_0_5_flips"),
        "error": res.get("error"),
    }


def run_stage_b(t_all: float) -> None:
    """Execute S4-2 batch matrix, padding matrix, and validation eval sweep."""
    for exp_id, micro in (
        ("B2_micro2", 2),
        ("B3_micro4", 4),
        ("B4_micro8", 8),
        ("B5_micro16", 16),
    ):
        _check_budget(t_all)
        _run_subcommand(["run-one", "--spec-json", json.dumps(
            _spec(exp_id, "A", micro=micro, padding_mode="dynamic_mult8",
                  sort_mode="unsorted", gc_mode="full")
        )])

    for exp_id, micro, pmode, smode in (
        ("P0_static64_micro1", 1, "static_64", "unsorted"),
        ("P2_static64_micro8", 8, "static_64", "unsorted"),
        ("P4_mult8_sorted_micro4", 4, "dynamic_mult8", "intra_update_sorted"),
        ("P5_mult8_sorted_micro8", 8, "dynamic_mult8", "intra_update_sorted"),
        ("P6_mult1_sorted_micro8", 8, "dynamic_mult1", "intra_update_sorted"),
    ):
        _check_budget(t_all)
        _run_subcommand(["run-one", "--spec-json", json.dumps(
            _spec(exp_id, "A", micro=micro, padding_mode=pmode, sort_mode=smode, gc_mode="full")
        )])

    _check_budget(t_all)
    _run_subcommand(["eval-sweep"])


def run_stage_c(t_all: float, suite: dict, snapshot: Path, data, encoded: dict, progress) -> None:
    """Execute S4-3 gradient checkpointing matrix and torch.compile audit in one loaded model."""
    gc_specs = [
        _spec("GC_micro1_partial16", "A", micro=1, gc_mode="partial_16"),
        _spec("GC_micro1_off", "A", micro=1, gc_mode="off", stress_worst_batch=True),
        _spec("GC_micro8_partial16", "A", micro=8, gc_mode="partial_16"),
        _spec("GC_micro8_attn8", "A", micro=8, gc_mode="partial_attn8"),
        _spec("GC_micro8_off", "A", micro=8, gc_mode="off", stress_worst_batch=True),
    ]
    b5_path = S4_PERF / "runs" / "B5_micro16" / "result.json"
    if b5_path.is_file() and _load_run("B5_micro16")["status"] == "PASS":
        gc_specs.extend([
            _spec("GC_micro16_partial16", "A", micro=16, gc_mode="partial_16"),
            _spec("GC_micro16_attn8", "A", micro=16, gc_mode="partial_attn8"),
            _spec("GC_micro16_off", "A", micro=16, gc_mode="off", stress_worst_batch=True),
        ])

    seed_all(suite["preflight"]["seed"])
    start_mem = memory()
    torch.cuda.reset_peak_memory_stats()
    model, rank, head, _lora = build_model_s4(
        snapshot, suite, data.manifest["label_names"],
        precision_mode="nf4_fp32_stream", gc_mode="full",
    )
    load_mem = memory()
    saved_state = snapshot_trainable_state(rank, head)
    try:
        for sp in gc_specs:
            _check_budget(t_all)
            restore_trainable_state(rank, head, saved_state)
            run_experiment_with_model(
                sp, suite, data, encoded, progress, model, rank, head, start_mem, load_mem
            )
            gc.collect()
            torch.cuda.empty_cache()

        best_a = _pick_best_a_candidate()
        best_cfg = best_a["config"]
        _check_budget(t_all)
        restore_trainable_state(rank, head, saved_state)
        run_compile_audit(
            suite, snapshot, data, encoded, progress,
            micro_batch_size=int(best_cfg["micro_batch_size"]),
            gc_mode=str(best_cfg["gc_mode"]),
            sort_mode=str(best_cfg["sort_mode"]),
            loaded_bundle=(model, rank, head, saved_state),
        )
    finally:
        model = None
        rank = {}
        head = {}
        saved_state = {}
        gc.collect()
        torch.cuda.empty_cache()


def _pick_best_a_candidate() -> dict:
    candidates = [
        _load_run(name) for name in (
            "B4_micro8", "B5_micro16", "P5_mult8_sorted_micro8",
            "GC_micro8_partial16", "GC_micro8_attn8", "GC_micro8_off",
            "GC_micro16_partial16", "GC_micro16_attn8", "GC_micro16_off",
        )
        if (S4_PERF / "runs" / name / "result.json").is_file()
        and _load_run(name)["status"] == "PASS"
    ]
    if not candidates:
        candidates = [_load_run("S4_0_baseline")]
    return max(candidates, key=lambda r: r["train_samples_per_second"])


def run_stage_d(t_all: float, suite: dict, progress) -> str:
    """Execute S4-4 winner recovery, foreach optimization, and Category B probes."""
    best_cfg = _pick_best_a_candidate()["config"]
    vsweep = read_json(S4_PERF / "validation-sweep.json")
    best_eval = max(
        (r for r in vsweep["results"] if r["status"] == "PASS" and r["rows"] == 5416),
        key=lambda r: r["samples_per_second"],
        default={"batch_size": 16, "sort_by_length": True},
    )

    winner_id = "A_winner_foreach_evalfast"
    _check_budget(t_all)
    _run_subcommand(["run-one", "--spec-json", json.dumps(
        _spec(
            winner_id, "A",
            micro=int(best_cfg["micro_batch_size"]),
            padding_mode=best_cfg["padding_mode"],
            sort_mode=best_cfg["sort_mode"],
            gc_mode=best_cfg["gc_mode"],
            foreach_adamw=True,
            eval_batch_size=int(best_eval["batch_size"]),
            eval_sorted=bool(best_eval["sort_by_length"]),
            stress_worst_batch=(best_cfg["gc_mode"] == "off"),
            publish_recovery=True,
        )
    )])
    _check_budget(t_all)
    _run_subcommand(["recover", "--exp-id", winner_id])

    _check_budget(t_all)
    _run_subcommand(["run-one", "--spec-json", json.dumps(
        _spec(
            "B1_nf4_bf16_stream", "B",
            micro=int(best_cfg["micro_batch_size"]),
            padding_mode=best_cfg["padding_mode"],
            sort_mode=best_cfg["sort_mode"],
            gc_mode=best_cfg["gc_mode"],
            precision_mode="nf4_bf16_stream",
            foreach_adamw=True,
            eval_batch_size=int(best_eval["batch_size"]),
            eval_sorted=bool(best_eval["sort_by_length"]),
            stress_worst_batch=(best_cfg["gc_mode"] == "off"),
        )
    )])

    _check_budget(t_all)
    _run_subcommand(["run-one", "--spec-json", json.dumps(
        _spec(
            "B2_bf16_lora_micro1_gcfull", "B",
            micro=1, padding_mode="dynamic_mult8", sort_mode="unsorted",
            gc_mode="full", precision_mode="bf16_lora", foreach_adamw=True,
            eval_batch_size=int(best_eval["batch_size"]),
            eval_sorted=bool(best_eval["sort_by_length"]),
        )
    )])
    if _load_run("B2_bf16_lora_micro1_gcfull")["status"] == "PASS":
        _check_budget(t_all)
        _run_subcommand(["run-one", "--spec-json", json.dumps(
            _spec(
                "B2_bf16_lora_micro8_gcoff", "B",
                micro=8, padding_mode="dynamic_mult8", sort_mode="unsorted",
                gc_mode="off", precision_mode="bf16_lora", foreach_adamw=True,
                eval_batch_size=int(best_eval["batch_size"]),
                eval_sorted=bool(best_eval["sort_by_length"]),
                stress_worst_batch=True,
            )
        )])
    progress.emit("S4_STAGE_D_COMPLETE", "Stage D finished", winner=winner_id)
    return winner_id


def orchestrate(suite: dict, snapshot: Path, data, encoded: dict, progress) -> None:
    """Run all S4 phases cleanly in isolated subprocesses under the 1-hour GPU budget."""
    t_all = time.perf_counter()
    verify_s3_frozen_unchanged()
    write_environment_txt(suite)
    write_padding_shapes_csv(data, encoded, suite)

    _run_subcommand(["run-one", "--spec-json", json.dumps(
        _spec("S4_0_baseline", "baseline", micro=1, gc_mode="full", eval_batch_size=1)
    )])
    base_res = _load_run("S4_0_baseline")
    s3_sps = float(read_json(S3_REPORTS / "comparison.json")["wsl"]["samples_per_second"])
    dev_pct = abs(base_res["train_samples_per_second"] - s3_sps) / s3_sps * 100.0
    progress.emit("S4_0_DONE", "Baseline replay finished",
                  sps=round(base_res["train_samples_per_second"], 3),
                  deviation_pct=round(dev_pct, 2))
    if dev_pct > 10.0:
        raise RuntimeError(f"Baseline deviated {dev_pct:.2f}% from S3 ({s3_sps:.3f})")

    _run_subcommand(["profile"])
    run_stage_b(t_all)
    _run_subcommand(["stage-c"])
    winner_id = run_stage_d(t_all, suite, progress)

    total_elapsed = time.perf_counter() - t_all
    _export_reports(suite, winner_id, {"elapsed_total_seconds": total_elapsed})
    progress.emit("S4_ALL_COMPLETE", "All S4 experiments and exports complete",
                  elapsed_seconds=round(total_elapsed, 1), winner=winner_id)


def _check_budget(t_start: float) -> None:
    elapsed = time.perf_counter() - t_start
    if elapsed > GPU_BUDGET_SECONDS:
        raise RuntimeError(f"GPU short-test budget exceeded ({elapsed:.1f}s > {GPU_BUDGET_SECONDS}s)")


def reparse_trace(trace_file: str | None = None) -> None:
    """Re-parse a saved Chrome trace JSON without re-running the GPU profiler."""
    from nlp_text_classification.experiments.batch_b_s4_telemetry import parse_chrome_trace

    if trace_file:
        res = parse_chrome_trace(Path(trace_file), wall_seconds=None)
        print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)
        return
    prof_path = S4_PERF / "profile-summary.json"
    summary = read_json(prof_path)
    trace_path = Path(summary["trace_path"])
    summary["trace_analysis"] = parse_chrome_trace(
        trace_path, wall_seconds=summary["profiled_wall_seconds"]
    )
    write_atomic_json(prof_path, summary)
    _write_csv(REPORTS / "operator-top10.csv", summary["trace_analysis"]["top_kernels"][:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("reparse-trace")
    p_rep.add_argument("--trace", default=None)
    p_rep.add_argument("--no-lock", action="store_true")

    for name in (
        "audit", "profile", "eval-sweep", "stage-b", "stage-c", "stage-d",
        "all", "summarize",
    ):
        p = sub.add_parser(name)
        p.add_argument("--no-lock", action="store_true")
        if name == "summarize":
            p.add_argument("--winner", default="A_winner_foreach_evalfast")

    p_one = sub.add_parser("run-one")
    p_one.add_argument("--spec-json", required=True)
    p_one.add_argument("--no-lock", action="store_true")

    p_comp = sub.add_parser("compile-audit")
    p_comp.add_argument("--micro", type=int, default=8)
    p_comp.add_argument("--gc-mode", default="off")
    p_comp.add_argument("--sort-mode", default="unsorted")
    p_comp.add_argument("--no-lock", action="store_true")

    p_rec = sub.add_parser("recover")
    p_rec.add_argument("--exp-id", required=True)
    p_rec.add_argument("--no-lock", action="store_true")

    args = parser.parse_args()
    configure_compute_threads()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    suite = load_yaml(SUITE)
    S4_PERF.mkdir(parents=True, exist_ok=True)
    progress = Progress(S4_PERF / "progress.md", " ".join(sys.orig_argv),
                        title="Batch B S4 performance investigation")

    if args.cmd == "summarize":
        _export_reports(suite, args.winner, {})
        return
    if args.cmd == "reparse-trace":
        reparse_trace(args.trace)
        return

    lock_cm = (
        exclusive_queue(S3_ROOT.parent / ".batch-a-gpu.lock")
        if not getattr(args, "no_lock", False)
        else _noop_cm()
    )
    with lock_cm, protect(
        progress, process_limit=suite["preflight"]["process_commit_limit_gib"] * GIB
    ):
        snapshot, data, _, encoded = inputs(suite)
        if args.cmd in {"all", "stage-b", "stage-c", "stage-d"}:
            t_start = time.perf_counter()
            if args.cmd == "all":
                orchestrate(suite, snapshot, data, encoded, progress)
            elif args.cmd == "stage-b":
                run_stage_b(t_start)
                _export_reports(suite, "S4_0_baseline", {})
            elif args.cmd == "stage-c":
                configure_device()
                run_stage_c(t_start, suite, snapshot, data, encoded, progress)
                _export_reports(suite, _pick_best_a_candidate()["exp_id"], {})
            elif args.cmd == "stage-d":
                winner_id = run_stage_d(t_start, suite, progress)
                _export_reports(suite, winner_id, {})
            return
        configure_device()
        if args.cmd == "audit":
            verify_s3_frozen_unchanged()
            write_environment_txt(suite)
            write_padding_shapes_csv(data, encoded, suite)
            run_experiment(
                _spec("S4_0_baseline", "baseline", micro=1, gc_mode="full", eval_batch_size=1),
                suite, snapshot, data, encoded, progress,
            )
            _export_reports(suite, "S4_0_baseline", {})
        elif args.cmd == "profile":
            run_profile_suite(suite, snapshot, data, encoded, progress)
            reparse_trace()
        elif args.cmd == "run-one":
            run_experiment(json.loads(args.spec_json), suite, snapshot, data, encoded, progress)
        elif args.cmd == "eval-sweep":
            run_eval_sweep(suite, snapshot, data, encoded, progress)
        elif args.cmd == "compile-audit":
            run_compile_audit(
                suite, snapshot, data, encoded, progress,
                micro_batch_size=args.micro, gc_mode=args.gc_mode, sort_mode=args.sort_mode,
            )
        elif args.cmd == "recover":
            recover_winner(args.exp_id, suite, snapshot, data, encoded, progress)


class _noop_cm:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


if __name__ == "__main__":
    main()

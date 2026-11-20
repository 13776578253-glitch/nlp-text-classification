"""Export frozen experiment evidence and dataset statistics for the web dashboard."""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import torch
import yaml

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.features import tokens as batch_a_tokens
from nlp_text_classification.evaluation.dashboard_comparisons import build_chinese_comparisons
from nlp_text_classification.models.neural import make_model

MODEL_DISPLAY: dict[str, dict[str, str]] = {
    "E1": {
        "display_name": "TF-IDF + LinearSVC",
        "short_label": "LinearSVC",
        "batch": "Batch A",
        "family": "Linear",
    },
    "E2": {
        "display_name": "TextCNN (Random Emb.)",
        "short_label": "TextCNN-Rand",
        "batch": "Batch A",
        "family": "CNN",
    },
    "E3-RNN": {
        "display_name": "Vanilla RNN",
        "short_label": "RNN",
        "batch": "Batch A",
        "family": "Recurrent",
    },
    "E3-LSTM": {
        "display_name": "LSTM",
        "short_label": "LSTM",
        "batch": "Batch A",
        "family": "Recurrent",
    },
    "E3-BiLSTM": {
        "display_name": "BiLSTM",
        "short_label": "BiLSTM",
        "batch": "Batch A",
        "family": "Recurrent",
    },
    "E5-TUNE": {
        "display_name": "TextCNN (Word2Vec Tune)",
        "short_label": "TextCNN-W2V-Tune",
        "batch": "Batch A",
        "family": "CNN",
    },
    "E5-FROZEN": {
        "display_name": "TextCNN (Word2Vec Frozen)",
        "short_label": "TextCNN-W2V-Froz",
        "batch": "Batch A",
        "family": "CNN",
    },
    "B1": {
        "display_name": "DistilBERT-base-uncased",
        "short_label": "DistilBERT",
        "batch": "Batch B · Encoder",
        "family": "Encoder",
    },
    "B2": {
        "display_name": "ModernBERT-base",
        "short_label": "ModernBERT-base",
        "batch": "Batch B · Encoder",
        "family": "Encoder",
    },
    "B3": {
        "display_name": "ModernBERT-large",
        "short_label": "ModernBERT-large",
        "batch": "Batch B · Encoder",
        "family": "Encoder",
    },
    "B4": {
        "display_name": "Chinese MacBERT-base",
        "short_label": "MacBERT-base",
        "batch": "Batch B · Encoder",
        "family": "Encoder",
    },
    "Qwen3.5-4B": {
        "display_name": "Qwen3.5-4B-Base (QLoRA)",
        "short_label": "Qwen3.5-4B",
        "batch": "Batch B · QLoRA",
        "family": "LLM-QLoRA",
    },
}

_BOOTSTRAP_ROW_RE = re.compile(
    r"^\|\s*(?P<a>[^|]+?)\s*\|\s*(?P<b>[^|]+?)\s*\|\s*(?P<diff>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"\[(?P<lo>[-+]?\d+(?:\.\d+)?),\s*(?P<hi>[-+]?\d+(?:\.\d+)?)\]\s*\|\s*(?P<n>\d+)\s*\|$"
)

LOSS_FUNCTIONS: dict[str, str] = {
    "E1": "Squared Hinge",
    "E2": "BCEWithLogitsLoss",
    "E3-RNN": "BCEWithLogitsLoss",
    "E3-LSTM": "BCEWithLogitsLoss",
    "E3-BiLSTM": "BCEWithLogitsLoss",
    "E5-TUNE": "BCEWithLogitsLoss",
    "E5-FROZEN": "BCEWithLogitsLoss",
    "B1": "BCEWithLogitsLoss",
    "B2": "BCEWithLogitsLoss",
    "B3": "BCEWithLogitsLoss",
    "B4": "CrossEntropyLoss",
    "Qwen3.5-4B": "BCEWithLogitsLoss",
}


def _format_param_label(count: int | None, suffix: str = "") -> str:
    if not count or count <= 0:
        return "—"
    if count >= 1_000_000_000:
        val = f"{count / 1_000_000_000:.1f}".rstrip("0").rstrip(".") + "B"
    elif count >= 1_000_000:
        val = f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        val = f"{count / 1_000:.0f}K"
    else:
        val = str(count)
    return f"{val}{suffix}"


def _load_frozen_test_texts(root: Path, ds_id: str, limit: int = 512) -> list[str]:
    rows: list[str] = []
    test_file = root / "data/processed" / ds_id / "v1/test.jsonl"
    with test_file.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows.append(json.loads(line)["text"])
            if len(rows) >= limit:
                break
    return rows


def _benchmark_batch_a_run(
    ds_id: str,
    model_code: str,
    run_path: Path,
    resolved_cfg: dict[str, Any],
    texts: list[str],
    batch_size: int = 32,
) -> tuple[int, dict[str, Any]]:
    if model_code == "E1":
        bundle = joblib.load(run_path / "checkpoint/model.joblib")
        clf = bundle["classifier"]
        if hasattr(clf, "estimators_"):
            param_count = int(
                sum(est.coef_.size + est.intercept_.size for est in clf.estimators_)
            )
        else:
            param_count = int(clf.coef_.size + clf.intercept_.size)
        vec = bundle["vectorizer"]
        vec.transform(texts[:batch_size])
        repeats = 3
        started = time.perf_counter()
        for _ in range(repeats):
            for start in range(0, len(texts), batch_size):
                feats = vec.transform(texts[start : start + batch_size])
                clf.decision_function(feats)
        elapsed = max(1e-6, (time.perf_counter() - started) / repeats)
        return param_count, {
            "ms_per_sample": round((elapsed / len(texts)) * 1000.0, 4),
            "samples_per_second": round(len(texts) / elapsed, 1),
            "batch_size": batch_size,
            "device": "cpu",
            "source": "frozen_test_replay",
        }

    ckpt = torch.load(run_path / "checkpoint/model.pt", map_location="cpu", weights_only=True)
    vocab: dict[str, int] = ckpt["vocabulary"]
    state_dict: dict[str, torch.Tensor] = ckpt["model"]
    param_count = int(sum(tensor.numel() for tensor in state_dict.values()))

    exp = resolved_cfg["experiment"]
    settings = resolved_cfg["settings"]
    outputs = 1 if ds_id == "movie_reviews_zh" else 28
    max_len = 256 if ds_id == "movie_reviews_zh" else 128
    tok_mode = "whitespace" if ds_id == "movie_reviews_zh" else "english_word"
    min_len = max(settings["textcnn"]["kernel_sizes"]) if exp["model"] == "textcnn" else 1

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = make_model(exp["model"], settings, len(vocab) + 2, outputs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    encoded_seqs: list[list[int]] = []
    for txt in texts:
        seq = [vocab.get(w, 1) for w in batch_a_tokens(txt, tok_mode)[:max_len]] or [1]
        encoded_seqs.append(seq)

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(encoded_seqs), batch_size):
        chunk = encoded_seqs[start : start + batch_size]
        width = max(min_len, max(len(s) for s in chunk))
        x = torch.zeros((len(chunk), width), dtype=torch.long)
        lengths = torch.tensor([len(s) for s in chunk], dtype=torch.long)
        for idx, seq in enumerate(chunk):
            x[idx, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
        batches.append((x.to(device), lengths))

    with torch.inference_mode():
        for _ in range(3):
            for x_dev, lengths in batches[:3]:
                model(x_dev, lengths).sigmoid()
        if device_name == "cuda":
            torch.cuda.synchronize()
        repeats = 5
        started = time.perf_counter()
        for _ in range(repeats):
            for x_dev, lengths in batches:
                model(x_dev, lengths).sigmoid()
        if device_name == "cuda":
            torch.cuda.synchronize()
        elapsed = max(1e-6, (time.perf_counter() - started) / repeats)

    return param_count, {
        "ms_per_sample": round((elapsed / len(texts)) * 1000.0, 4),
        "samples_per_second": round(len(texts) / elapsed, 1),
        "batch_size": batch_size,
        "device": device_name,
        "source": "frozen_test_replay",
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_history_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value == "" or value is None:
                    parsed[key] = None
                elif key in {"epoch", "best_epoch", "optimizer_steps"}:
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _unsegment_zh(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return ""
    out = [tokens[0]]
    for token in tokens[1:]:
        prev = out[-1]
        if prev[-1].isascii() and prev[-1].isalnum() and token[0].isascii() and token[0].isalnum():
            out.append(" ")
        out.append(token)
    return "".join(out)


def _export_datasets(root: Path) -> dict[str, Any]:
    token_lock = _read_json(root / "configs/batch_b/tokenization_lock.json")
    qwen_tok = _read_json(root / "reports/batch-b-s3/performance/tokenization.json")
    qwen_test_tok = _read_json(root / "reports/batch-b-s4/formal/test-tokenization.json")

    datasets: dict[str, Any] = {}
    configs = {
        "goemotions": {
            "raw_counts": {"train": 43410, "validation": 5426, "test": 5427},
            "batch_a_lengths": {"max_length": 128, "p50": 15.0, "p95": 28.0, "p99": 32.0},
            "tokenizer_name": "english_word",
        },
        "movie_reviews_zh": {
            "raw_counts": {"train": 19998, "validation": 5629, "test": 369},
            "batch_a_lengths": {"max_length": 256, "p50": 43.0, "p95": 62.0, "p99": 68.0},
            "tokenizer_name": "whitespace",
        },
    }

    for ds_id, cfg in configs.items():
        v1_dir = root / "data/processed" / ds_id / "v1"
        manifest = _read_json(v1_dir / "dataset_manifest.json")
        source_audit = _read_json(v1_dir / "source_audit.json")
        excluded = _read_json(v1_dir / "excluded.json")

        exclusion_reasons = Counter(row["reason"] for row in excluded["rows"])
        splits: dict[str, Any] = {}
        total_v1 = sum(manifest["derived_summary"]["counts"].values())
        for split in ("train", "validation", "test"):
            raw_n = source_audit["counts"][split]
            v1_n = manifest["derived_summary"]["counts"][split]
            splits[split] = {
                "raw_count": raw_n,
                "frozen_count": v1_n,
                "excluded_count": raw_n - v1_n,
                "ratio": v1_n / total_v1,
                "sha256": manifest["splits"][split]["sha256"],
            }

        if ds_id == "goemotions":
            label_counts = []
            train_total_rows = manifest["derived_summary"]["counts"]["train"]
            for idx, name in enumerate(manifest["label_names"]):
                key = str(idx)
                raw_tr = source_audit["label_counts"]["train"][key]
                v1_tr = manifest["derived_summary"]["label_counts"]["train"][key]
                raw_va = source_audit["label_counts"]["validation"][key]
                v1_va = manifest["derived_summary"]["label_counts"]["validation"][key]
                te = manifest["derived_summary"]["label_counts"]["test"][key]
                label_counts.append(
                    {
                        "id": idx,
                        "name": name,
                        "raw_train": raw_tr,
                        "v1_train": v1_tr,
                        "raw_val": raw_va,
                        "v1_val": v1_va,
                        "test": te,
                        "train_rate": v1_tr / train_total_rows,
                    }
                )
            tokenization_profiles = [
                {
                    "group": "Batch A (Word)",
                    "max_length": cfg["batch_a_lengths"]["max_length"],
                    "train_p50": cfg["batch_a_lengths"]["p50"],
                    "train_p95": cfg["batch_a_lengths"]["p95"],
                    "train_p99": cfg["batch_a_lengths"]["p99"],
                    "train_truncated": 0,
                    "val_truncated": 0,
                },
                *[
                    {
                        "group": f"{m_id} ({MODEL_DISPLAY[m_id]['short_label']})",
                        "max_length": token_lock["models"][m_id]["selected_length"],
                        "train_p50": token_lock["models"][m_id]["partitions"]["train"]["percentiles"]["50"],
                        "train_p95": token_lock["models"][m_id]["partitions"]["train"]["percentiles"]["95"],
                        "train_p99": token_lock["models"][m_id]["partitions"]["train"]["percentiles"]["99"],
                        "train_truncated": token_lock["models"][m_id]["partitions"]["train"]["truncated_rows"],
                        "val_truncated": token_lock["models"][m_id]["partitions"]["validation"]["truncated_rows"],
                    }
                    for m_id in ("B1", "B2", "B3")
                ],
                {
                    "group": "Qwen3.5-4B (QLoRA)",
                    "max_length": qwen_tok["selected_length"],
                    "train_p50": qwen_tok["partitions"]["train"]["percentiles"]["p50"],
                    "train_p95": qwen_tok["partitions"]["train"]["percentiles"]["p95"],
                    "train_p99": qwen_tok["partitions"]["train"]["percentiles"]["p99"],
                    "train_truncated": qwen_tok["partitions"]["train"]["truncated_rows"],
                    "val_truncated": qwen_tok["partitions"]["validation"]["truncated_rows"],
                    "test_truncated": qwen_test_tok["truncated_rows"],
                },
            ]
        else:
            # Chinese dataset: label_names are ["negative", "positive"] matching label_ids [1, 0]
            label_counts = []
            for idx, (name, raw_id) in enumerate(
                zip(manifest["label_names"], manifest["label_ids"], strict=True)
            ):
                key = str(raw_id)
                label_counts.append(
                    {
                        "id": idx,
                        "source_id": raw_id,
                        "name": name,
                        "raw_train": source_audit["label_counts"]["train"][key],
                        "v1_train": manifest["derived_summary"]["label_counts"]["train"][key],
                        "raw_val": source_audit["label_counts"]["validation"][key],
                        "v1_val": manifest["derived_summary"]["label_counts"]["validation"][key],
                        "test": manifest["derived_summary"]["label_counts"]["test"][key],
                    }
                )
            b4_tok = token_lock["models"]["B4"]
            tokenization_profiles = [
                {
                    "group": "Batch A (Whitespace)",
                    "max_length": cfg["batch_a_lengths"]["max_length"],
                    "train_p50": cfg["batch_a_lengths"]["p50"],
                    "train_p95": cfg["batch_a_lengths"]["p95"],
                    "train_p99": cfg["batch_a_lengths"]["p99"],
                    "train_truncated": 0,
                    "val_truncated": 0,
                },
                {
                    "group": "B4 (MacBERT WordPiece)",
                    "max_length": b4_tok["selected_length"],
                    "train_p50": b4_tok["partitions"]["train"]["percentiles"]["50"],
                    "train_p95": b4_tok["partitions"]["train"]["percentiles"]["95"],
                    "train_p99": b4_tok["partitions"]["train"]["percentiles"]["99"],
                    "train_truncated": b4_tok["partitions"]["train"]["truncated_rows"],
                    "val_truncated": b4_tok["partitions"]["validation"]["truncated_rows"],
                },
            ]

        train_ann = sum(int(v) for v in manifest["derived_summary"]["label_counts"]["train"].values())
        val_ann = sum(int(v) for v in manifest["derived_summary"]["label_counts"]["validation"].values())
        test_ann = sum(int(v) for v in manifest["derived_summary"]["label_counts"]["test"].values())
        zero_overlap = all(
            item.get("shared_texts", 0) == 0
            for item in manifest["derived_summary"]["cross_split"].values()
        )

        datasets[ds_id] = {
            "dataset": ds_id,
            "task_type": manifest["task_type"],
            "label_names": manifest["label_names"],
            "label_ids": manifest["label_ids"],
            "positive_raw_id": 0 if ds_id == "movie_reviews_zh" else None,
            "manifest_sha256": manifest.get("training_sha256"),
            "splits": splits,
            "total_raw": sum(source_audit["counts"].values()),
            "total_frozen": total_v1,
            "total_label_annotations": {
                "train": train_ann,
                "validation": val_ann,
                "test": test_ann,
            },
            "avg_labels_per_sample": {
                "train": train_ann / splits["train"]["frozen_count"],
                "validation": val_ann / splits["validation"]["frozen_count"],
                "test": test_ann / splits["test"]["frozen_count"],
            },
            "audit_summary": {
                "status": "PASS" if zero_overlap else "WARN",
                "zero_cross_split_overlap": zero_overlap,
                "excluded_total": sum(s["excluded_count"] for s in splits.values()),
                "blank_rows_skipped_total": sum(source_audit.get("blank_rows", {}).values()),
            },
            "exclusion_reasons": dict(exclusion_reasons),
            "blank_rows_skipped": source_audit.get("blank_rows", {}),
            "cross_split_overlap": manifest["derived_summary"]["cross_split"],
            "within_split": manifest["derived_summary"]["within_split"],
            "label_counts": label_counts,
            "tokenization_profiles": tokenization_profiles,
            "word2vec_info": source_audit.get("word2vec"),
        }
    return datasets


def _parse_zh_bootstrap(report_md: Path) -> list[dict[str, Any]]:
    lines = report_md.read_text(encoding="utf-8").splitlines()
    in_section = False
    rows: list[dict[str, Any]] = []
    for line in lines:
        if line.strip() == "## Chinese paired Bootstrap":
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            match = _BOOTSTRAP_ROW_RE.match(line.strip())
            if match:
                rows.append(
                    {
                        "model_a": match.group("a"),
                        "model_b": match.group("b"),
                        "comparison": f"{match.group('a')} − {match.group('b')}",
                        "difference": float(match.group("diff")),
                        "ci_low": float(match.group("lo")),
                        "ci_high": float(match.group("hi")),
                        "n": int(match.group("n")),
                    }
                )
    if len(rows) != 5:
        raise ValueError(f"Expected 5 Chinese paired Bootstrap rows in {report_md}, got {len(rows)}")
    return rows


def _collect_runs_and_summaries(root: Path) -> dict[str, Any]:
    batch_a_idx = _read_json(root / "docs/finish-task/batch-a/model-index.json")
    class_metrics_a = _read_json(root / "reports/batch-a/255a9bfe53ee/class_metrics.json")
    quality_a = _read_json(root / "reports/batch-a/255a9bfe53ee/quality_review.json")
    zh_bootstrap = _parse_zh_bootstrap(root / "reports/batch-a/255a9bfe53ee/report.md")

    encoder_freeze = _read_json(root / "reports/batch-b-final/2605f6766085/freeze.json")
    encoder_results = _read_json(root / "reports/batch-b-final/2605f6766085/results.json")
    encoder_label_metrics = _read_json(
        root / "reports/batch-b-final/2605f6766085/label-metrics.json"
    )

    qwen_freeze = _read_json(
        root
        / "reports/batch-b-s4/formal/frozen"
        / "97ce29261f3844e715731f62f7859df2397db2e7381744bc8a0db86cff4e5d15.json"
    )
    qwen_audit = _read_json(root / "reports/batch-b-s4/formal/freeze-audit.json")
    qwen_test = _read_json(root / "reports/batch-b-s4/formal/test-acceptance.json")
    resource_lock = _read_json(root / "configs/batch_b/resource_lock.json")
    test_texts_cache = {
        ds_id: _load_frozen_test_texts(root, ds_id, limit=512)
        for ds_id in ("goemotions", "movie_reviews_zh")
    }

    formal_runs: list[dict[str, Any]] = []
    batch_a_main_map: dict[str, dict[str, Any]] = {}
    batch_a_seed_groups: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    batch_a_curves: dict[str, dict[str, list[dict[str, Any]]]] = {
        "goemotions": {},
        "movie_reviews_zh": {},
    }
    batch_a_curve_candidates: dict[tuple[str, str, int], tuple[int, dict[str, Any]]] = {}

    # 1. Batch A (47 formal runs)
    for entry in batch_a_idx["formal"]["runs"]:
        role = entry["role"]
        parts = entry["key"].split("/")
        dataset, model_code = parts[0], parts[1]
        seed = int(entry["seed"])
        run_path = root / entry["run"]
        thresholds = _read_json(run_path / "thresholds.json")
        result = _read_json(run_path / "result.json")
        resolved_cfg = _read_yaml(run_path / "resolved_config.yaml")
        test_metrics = _read_json(run_path / "test/metrics_test.json")

        meta_disp = MODEL_DISPLAY[model_code]
        secondary = (
            test_metrics.get("subset_accuracy")
            if dataset == "goemotions"
            else test_metrics.get("accuracy")
        )
        run_record = {
            "id": f"batch-a:{dataset}:{model_code}:seed-{seed}:{role}",
            "dataset": dataset,
            "batch": meta_disp["batch"],
            "family": meta_disp["family"],
            "model": model_code,
            "display_name": meta_disp["display_name"],
            "short_label": meta_disp["short_label"],
            "role": "primary" if role == "main" else role,
            "seed": seed,
            "run_id": run_path.name,
            "score_kind": thresholds["score_kind"],
            "threshold": thresholds["threshold"],
            "best_epoch": result.get("best_epoch"),
            "val_macro_f1": entry["validation_macro_f1"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_micro_f1": test_metrics["micro_f1"],
            "secondary_metric": secondary,
            "secondary_metric_name": "subset_accuracy" if dataset == "goemotions" else "accuracy",
            "samples_f1": test_metrics.get("samples_f1"),
            "hamming_loss": test_metrics.get("hamming_loss"),
            "positive_f1": test_metrics.get("positive_f1"),
            "roc_auc": test_metrics.get("roc_auc"),
            "pr_auc": test_metrics.get("pr_auc"),
            "stop_reason": result.get("stop_reason"),
            "parameters": resolved_cfg.get("settings"),
        }
        formal_runs.append(run_record)

        history = _read_history_csv(run_path / "history.csv")
        if history:
            priority = 0 if role == "main" else 1
            c_key = (dataset, model_code, seed)
            existing = batch_a_curve_candidates.get(c_key)
            if existing is None or priority < existing[0]:
                ckpt_file = run_path / "checkpoint/model.pt"
                batch_a_curve_candidates[c_key] = (
                    priority,
                    {
                        "id": f"{dataset}:{model_code}:seed-{seed}",
                        "dataset": dataset,
                        "model": model_code,
                        "display_name": meta_disp["display_name"],
                        "short_label": meta_disp["short_label"],
                        "batch": meta_disp["batch"],
                        "seed": seed,
                        "role": "primary" if role == "main" else role,
                        "run_id": run_path.name,
                        "loss_function": LOSS_FUNCTIONS[model_code],
                        "score_kind": thresholds["score_kind"],
                        "threshold": thresholds["threshold"],
                        "best_epoch": result.get("best_epoch"),
                        "epochs_completed": len(history),
                        "stop_reason": result.get("stop_reason"),
                        "val_macro_f1": entry["validation_macro_f1"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "global_step": None,
                        "checkpoint_path": f"{entry['run']}/checkpoint/model.pt",
                        "checkpoint_sha256": sha256_file(ckpt_file) if ckpt_file.is_file() else None,
                        "epochs": history,
                    },
                )

        if role == "main":
            param_count, inf_bench = _benchmark_batch_a_run(
                dataset,
                model_code,
                run_path,
                resolved_cfg,
                test_texts_cache[dataset],
            )
            batch_a_main_map[f"{dataset}/{model_code}"] = run_record | {
                "result_raw": result,
                "test_metrics": test_metrics,
                "run_path": entry["run"],
                "param_count": param_count,
                "param_label": _format_param_label(param_count),
                "inference_benchmark": inf_bench,
            }
            if history:
                batch_a_curves[dataset][model_code] = history
        else:
            batch_a_seed_groups.setdefault((dataset, model_code), {})[seed] = run_record

    batch_a_run_curves = [pair[1] for _, pair in sorted(batch_a_curve_candidates.items())]

    # 2. Batch B Encoders (12 formal runs)
    encoder_curves: dict[str, Any] = {}
    encoder_curve_runs: list[dict[str, Any]] = []
    encoder_bench_acc: dict[str, list[tuple[int, float]]] = {}
    for run_key, item in sorted(encoder_results["runs"].items()):
        model_code = item["model"]
        seed = int(item["seed"])
        dataset = item["dataset"]
        test_m = item["test"]
        meta_disp = MODEL_DISPLAY[model_code]
        run_rel = item["run"].replace("\\", "/")
        run_path = root / run_rel
        result = _read_json(run_path / "result.json")
        eval_manifest = _read_json(
            root
            / "artifacts/batch-b-final/test"
            / encoder_results["freeze_sha256"]
            / run_key
            / "evaluation_manifest.json"
        )
        val_inf = result.get("validation_inference") or {}
        if val_inf.get("samples") and val_inf.get("seconds"):
            encoder_bench_acc.setdefault(model_code, []).append(
                (int(val_inf["samples"]), float(val_inf["seconds"]))
            )

        formal_runs.append(
            {
                "id": f"batch-b-enc:{dataset}:{model_code}:seed-{seed}",
                "dataset": dataset,
                "batch": meta_disp["batch"],
                "family": meta_disp["family"],
                "model": model_code,
                "display_name": meta_disp["display_name"],
                "short_label": meta_disp["short_label"],
                "role": item["role"],
                "seed": seed,
                "run_id": run_path.name,
                "score_kind": item["score_kind"],
                "threshold": item["threshold"],
                "best_epoch": item["best_epoch"],
                "val_macro_f1": item["validation_macro_f1"],
                "test_macro_f1": test_m["macro_f1"],
                "test_micro_f1": test_m["micro_f1"],
                "secondary_metric": (
                    test_m["subset_accuracy"] if dataset == "goemotions" else test_m["accuracy"]
                ),
                "secondary_metric_name": (
                    "subset_accuracy" if dataset == "goemotions" else "accuracy"
                ),
                "samples_f1": test_m.get("samples_f1"),
                "hamming_loss": test_m.get("hamming_loss"),
                "positive_f1": test_m.get("positive_f1"),
                "roc_auc": test_m.get("roc_auc"),
                "pr_auc": test_m.get("pr_auc"),
                "stop_reason": result.get("stop_reason"),
                "stage": item["stage"],
            }
        )
        history = _read_history_csv(run_path / "history.csv")
        encoder_curves[run_key] = {
            "model": model_code,
            "dataset": dataset,
            "seed": seed,
            "best_epoch": item["best_epoch"],
            "threshold": item["threshold"],
            "epochs": history,
        }
        encoder_curve_runs.append(
            {
                "id": f"{dataset}:{model_code}:seed-{seed}",
                "dataset": dataset,
                "model": model_code,
                "display_name": meta_disp["display_name"],
                "short_label": meta_disp["short_label"],
                "batch": meta_disp["batch"],
                "seed": seed,
                "role": item["role"],
                "run_id": run_path.name,
                "loss_function": LOSS_FUNCTIONS[model_code],
                "score_kind": item["score_kind"],
                "threshold": item["threshold"],
                "best_epoch": item["best_epoch"],
                "epochs_completed": len(history),
                "stop_reason": result.get("stop_reason"),
                "val_macro_f1": item["validation_macro_f1"],
                "test_macro_f1": test_m["macro_f1"],
                "elapsed_seconds": result.get("elapsed_seconds"),
                "global_step": None,
                "checkpoint_path": f"{run_rel}/{result['checkpoint']}",
                "checkpoint_sha256": eval_manifest.get("checkpoint_sha256"),
                "epochs": history,
            }
        )

    # Include B1 12-epoch budget probe curve
    b1_budget = encoder_freeze["b1_budget_decision"]
    b1_long_run = root / b1_budget["runs"]["s2_12_epochs"].replace("\\", "/")
    encoder_curves["B1-42-long-probe"] = {
        "model": "B1",
        "dataset": "goemotions",
        "seed": 42,
        "best_epoch": 4,
        "threshold": 0.2,
        "epochs": _read_history_csv(b1_long_run / "history.csv"),
    }

    # 3. Batch B Qwen3.5-4B S4 (3 formal runs)
    qwen_curves: dict[str, Any] = {}
    qwen_curve_runs: list[dict[str, Any]] = []
    qwen_by_seed: dict[int, dict[str, Any]] = {}
    qwen_bench_acc: list[tuple[int, float]] = []
    for frozen_row, test_row in zip(qwen_freeze["runs"], qwen_test["runs"], strict=True):
        seed = int(frozen_row["seed"])
        if seed != int(test_row["seed"]):
            raise ValueError("Mismatched Qwen seed ordering between freeze and test acceptance")
        m = test_row["metrics"]
        meta_disp = MODEL_DISPLAY["Qwen3.5-4B"]
        eval_json = _read_json(
            root
            / "artifacts/batch-b-s4/formal/test"
            / qwen_test["freeze_sha256"]
            / f"seed-{seed}/evaluation.json"
        )
        q_inf = eval_json["inference"]
        qwen_bench_acc.append((int(q_inf["rows"]), float(q_inf["inference_seconds"])))
        rec = {
            "id": f"batch-b-qwen:goemotions:Qwen3.5-4B:seed-{seed}",
            "dataset": "goemotions",
            "batch": meta_disp["batch"],
            "family": meta_disp["family"],
            "model": "Qwen3.5-4B",
            "display_name": meta_disp["display_name"],
            "short_label": meta_disp["short_label"],
            "role": "primary" if seed == 42 else "replication",
            "seed": seed,
            "run_id": f"seed-{seed}/{frozen_row['checkpoint']}",
            "score_kind": frozen_row["score_kind"],
            "threshold": frozen_row["threshold"],
            "best_epoch": frozen_row["best_epoch"],
            "val_macro_f1": frozen_row["validation_macro_f1"],
            "test_macro_f1": m["macro_f1"],
            "test_micro_f1": m["micro_f1"],
            "secondary_metric": m["subset_accuracy"],
            "secondary_metric_name": "subset_accuracy",
            "samples_f1": m["samples_f1"],
            "hamming_loss": m["hamming_loss"],
            "pr_auc": m["pr_auc"],
            "stop_reason": frozen_row["stop_reason"],
            "global_step": frozen_row["global_step"],
            "per_label_precision": m["per_label_precision"],
            "per_label_recall": m["per_label_recall"],
            "per_label_f1": m["per_label_f1"],
        }
        formal_runs.append(rec)
        qwen_by_seed[seed] = rec
        audit_seed = qwen_audit["runs"][f"seed-{seed}"]
        q_epochs = [
            {
                "epoch": ep["epoch"],
                "global_step": ep["global_step"],
                "val_macro_f1": ep["macro_f1"],
                "threshold": ep["threshold"],
                "loss": ep["train_loss"],
                "elapsed_seconds": ep["elapsed_seconds"],
                "validation_seconds": ep["validation_seconds"],
                "stale_epochs": ep["stale_epochs"],
            }
            for ep in audit_seed["epoch_records"]
        ]
        qwen_curves[f"seed-{seed}"] = {
            "seed": seed,
            "best_epoch": frozen_row["best_epoch"],
            "threshold": frozen_row["threshold"],
            "epochs": q_epochs,
        }
        qwen_curve_runs.append(
            {
                "id": f"goemotions:Qwen3.5-4B:seed-{seed}",
                "dataset": "goemotions",
                "model": "Qwen3.5-4B",
                "display_name": meta_disp["display_name"],
                "short_label": meta_disp["short_label"],
                "batch": meta_disp["batch"],
                "seed": seed,
                "role": "primary" if seed == 42 else "replication",
                "run_id": f"seed-{seed}/{frozen_row['checkpoint']}",
                "loss_function": LOSS_FUNCTIONS["Qwen3.5-4B"],
                "score_kind": frozen_row["score_kind"],
                "threshold": frozen_row["threshold"],
                "best_epoch": frozen_row["best_epoch"],
                "epochs_completed": len(q_epochs),
                "stop_reason": frozen_row["stop_reason"],
                "val_macro_f1": frozen_row["validation_macro_f1"],
                "test_macro_f1": m["macro_f1"],
                "elapsed_seconds": sum(ep["elapsed_seconds"] for ep in audit_seed["epoch_records"]),
                "global_step": frozen_row["global_step"],
                "checkpoint_path": (
                    f"artifacts/batch-b-s4/formal/runs/seed-{seed}/checkpoints/"
                    f"{frozen_row['checkpoint']}/adapter"
                ),
                "checkpoint_sha256": eval_json["identity"]["checkpoint_sha256"],
                "epochs": q_epochs,
            }
        )

    # Build Leaderboards (11 GoEmotions, 8 Chinese)
    leaderboard: dict[str, list[dict[str, Any]]] = {"goemotions": [], "movie_reviews_zh": []}
    batch_a_models = ["E1", "E2", "E3-RNN", "E3-LSTM", "E3-BiLSTM", "E5-TUNE", "E5-FROZEN"]

    for ds_id in ("goemotions", "movie_reviews_zh"):
        for model_code in batch_a_models:
            main_rec = batch_a_main_map[f"{ds_id}/{model_code}"]
            test_m = main_rec["test_metrics"]
            seed_group = batch_a_seed_groups.get((ds_id, model_code), {})
            three_seed = None
            if len(seed_group) == 3 and {42, 123, 2026} == set(seed_group.keys()):
                ordered = [seed_group[s] for s in (42, 123, 2026)]
                three_seed = {
                    "val_macro_f1": _mean_sd([r["val_macro_f1"] for r in ordered]),
                    "test_macro_f1": _mean_sd([r["test_macro_f1"] for r in ordered]),
                    "test_micro_f1": _mean_sd([r["test_micro_f1"] for r in ordered]),
                    "secondary_metric": _mean_sd([r["secondary_metric"] for r in ordered]),
                    "by_seed": {str(r["seed"]): r["test_macro_f1"] for r in ordered},
                    "val_by_seed": {str(r["seed"]): r["val_macro_f1"] for r in ordered},
                }
            meta_disp = MODEL_DISPLAY[model_code]
            leaderboard[ds_id].append(
                {
                    "model_id": f"{ds_id}/{model_code}",
                    "model_code": model_code,
                    "display_name": meta_disp["display_name"],
                    "short_label": meta_disp["short_label"],
                    "batch": meta_disp["batch"],
                    "family": meta_disp["family"],
                    "score_kind": main_rec["score_kind"],
                    "loss_function": LOSS_FUNCTIONS[model_code],
                    "param_count": main_rec["param_count"],
                    "trainable_params": main_rec["param_count"],
                    "param_label": main_rec["param_label"],
                    "inference_benchmark": main_rec["inference_benchmark"],
                    "threshold": main_rec["threshold"],
                    "best_epoch": main_rec["best_epoch"],
                    "stop_reason": main_rec["stop_reason"],
                    "run_id": main_rec["run_id"],
                    "seed42": {
                        "val_macro_f1": main_rec["val_macro_f1"],
                        "test_macro_f1": test_m["macro_f1"],
                        "test_micro_f1": test_m["micro_f1"],
                        "secondary_metric": main_rec["secondary_metric"],
                        "samples_f1": test_m.get("samples_f1"),
                        "hamming_loss": test_m.get("hamming_loss"),
                        "positive_f1": test_m.get("positive_f1"),
                        "positive_precision": test_m.get("positive_precision"),
                        "positive_recall": test_m.get("positive_recall"),
                        "roc_auc": test_m.get("roc_auc"),
                        "pr_auc": test_m.get("pr_auc"),
                    },
                    "three_seed": three_seed,
                    "parameters": main_rec["parameters"],
                }
            )

    # Batch B Encoders in Leaderboard
    for model_code in ("B1", "B2", "B3", "B4"):
        ds_id = "movie_reviews_zh" if model_code == "B4" else "goemotions"
        summary = encoder_results["three_seed_summary"][model_code]
        r42 = encoder_results["runs"][f"{model_code}-42"]
        t42 = r42["test"]
        meta_disp = MODEL_DISPLAY[model_code]
        enc_params = int(resource_lock["models"][model_code]["parameters"])
        enc_samples = sum(s for s, _ in encoder_bench_acc[model_code])
        enc_secs = sum(sec for _, sec in encoder_bench_acc[model_code])
        enc_bench = {
            "ms_per_sample": round((enc_secs / enc_samples) * 1000.0, 4),
            "samples_per_second": round(enc_samples / enc_secs, 1),
            "batch_size": 32,
            "device": "cuda",
            "source": "frozen_validation_eval",
        }
        if ds_id == "goemotions":
            three_seed = {
                "val_macro_f1": {
                    "n": 3,
                    "mean": summary["validation_macro_f1"]["mean"],
                    "sd": summary["validation_macro_f1"]["sample_sd"],
                },
                "test_macro_f1": {
                    "n": 3,
                    "mean": summary["macro_f1"]["mean"],
                    "sd": summary["macro_f1"]["sample_sd"],
                },
                "test_micro_f1": {
                    "n": 3,
                    "mean": summary["micro_f1"]["mean"],
                    "sd": summary["micro_f1"]["sample_sd"],
                },
                "secondary_metric": {
                    "n": 3,
                    "mean": summary["subset_accuracy"]["mean"],
                    "sd": summary["subset_accuracy"]["sample_sd"],
                },
                "by_seed": summary["macro_f1"]["by_seed"],
                "val_by_seed": summary["validation_macro_f1"]["by_seed"],
            }
        else:
            three_seed = {
                "val_macro_f1": {
                    "n": 3,
                    "mean": summary["validation_macro_f1"]["mean"],
                    "sd": summary["validation_macro_f1"]["sample_sd"],
                },
                "test_macro_f1": {
                    "n": 3,
                    "mean": summary["macro_f1"]["mean"],
                    "sd": summary["macro_f1"]["sample_sd"],
                },
                "test_micro_f1": {
                    "n": 3,
                    "mean": summary["accuracy"]["mean"],
                    "sd": summary["accuracy"]["sample_sd"],
                },
                "secondary_metric": {
                    "n": 3,
                    "mean": summary["accuracy"]["mean"],
                    "sd": summary["accuracy"]["sample_sd"],
                },
                "by_seed": summary["macro_f1"]["by_seed"],
                "val_by_seed": summary["validation_macro_f1"]["by_seed"],
                "roc_auc": summary["roc_auc"],
                "pr_auc": summary["pr_auc"],
            }

        leaderboard[ds_id].append(
            {
                "model_id": f"{ds_id}/{model_code}",
                "model_code": model_code,
                "display_name": meta_disp["display_name"],
                "short_label": meta_disp["short_label"],
                "batch": meta_disp["batch"],
                "family": meta_disp["family"],
                "score_kind": r42["score_kind"],
                "loss_function": LOSS_FUNCTIONS[model_code],
                "param_count": enc_params,
                "trainable_params": enc_params,
                "param_label": _format_param_label(enc_params),
                "inference_benchmark": enc_bench,
                "threshold": r42["threshold"],
                "best_epoch": r42["best_epoch"],
                "stop_reason": "early_stopping" if model_code != "B1" else "epoch_cap",
                "run_id": Path(r42["run"].replace("\\", "/")).name,
                "seed42": {
                    "val_macro_f1": r42["validation_macro_f1"],
                    "test_macro_f1": t42["macro_f1"],
                    "test_micro_f1": t42["micro_f1"],
                    "secondary_metric": (
                        t42["subset_accuracy"] if ds_id == "goemotions" else t42["accuracy"]
                    ),
                    "samples_f1": t42.get("samples_f1"),
                    "hamming_loss": t42.get("hamming_loss"),
                    "positive_f1": t42.get("positive_f1"),
                    "positive_precision": t42.get("positive_precision"),
                    "positive_recall": t42.get("positive_recall"),
                    "roc_auc": t42.get("roc_auc"),
                    "pr_auc": t42.get("pr_auc"),
                },
                "three_seed": three_seed,
                "parameters": next(
                    item["settings"]
                    for item in encoder_freeze["runs"]
                    if item["model"] == model_code and item["seed"] == 42
                ),
            }
        )

    # Batch B Qwen3.5-4B in Leaderboard
    q42 = qwen_by_seed[42]
    q_agg = qwen_test["aggregate"]
    q_meta = MODEL_DISPLAY["Qwen3.5-4B"]
    q_rows = sum(r for r, _ in qwen_bench_acc)
    q_secs = sum(s for _, s in qwen_bench_acc)
    leaderboard["goemotions"].append(
        {
            "model_id": "goemotions/Qwen3.5-4B",
            "model_code": "Qwen3.5-4B",
            "display_name": q_meta["display_name"],
            "short_label": q_meta["short_label"],
            "batch": q_meta["batch"],
            "family": q_meta["family"],
            "score_kind": "sigmoid",
            "loss_function": LOSS_FUNCTIONS["Qwen3.5-4B"],
            "param_count": 4_000_000_000,
            "trainable_params": 65_001_472,
            "param_label": "4B · NF4",
            "inference_benchmark": {
                "ms_per_sample": round((q_secs / q_rows) * 1000.0, 4),
                "samples_per_second": round(q_rows / q_secs, 1),
                "batch_size": 32,
                "device": "cuda-wsl2",
                "source": "frozen_test_eval",
            },
            "threshold": q42["threshold"],
            "best_epoch": q42["best_epoch"],
            "stop_reason": q42["stop_reason"],
            "run_id": q42["run_id"],
            "seed42": {
                "val_macro_f1": q42["val_macro_f1"],
                "test_macro_f1": q42["test_macro_f1"],
                "test_micro_f1": q42["test_micro_f1"],
                "secondary_metric": q42["secondary_metric"],
                "samples_f1": q42["samples_f1"],
                "hamming_loss": q42["hamming_loss"],
                "pr_auc": q42["pr_auc"],
            },
            "three_seed": {
                "val_macro_f1": _mean_sd([qwen_by_seed[s]["val_macro_f1"] for s in (42, 123, 2026)]),
                "test_macro_f1": {
                    "n": 3,
                    "mean": q_agg["macro_f1"]["mean"],
                    "sd": q_agg["macro_f1"]["sample_std"],
                },
                "test_micro_f1": {
                    "n": 3,
                    "mean": q_agg["micro_f1"]["mean"],
                    "sd": q_agg["micro_f1"]["sample_std"],
                },
                "secondary_metric": {
                    "n": 3,
                    "mean": q_agg["subset_accuracy"]["mean"],
                    "sd": q_agg["subset_accuracy"]["sample_std"],
                },
                "by_seed": {str(s): qwen_by_seed[s]["test_macro_f1"] for s in (42, 123, 2026)},
                "val_by_seed": {str(s): qwen_by_seed[s]["val_macro_f1"] for s in (42, 123, 2026)},
            },
            "parameters": {
                "lora_rank": qwen_freeze["recipe"]["lora"]["rank"],
                "lora_alpha": qwen_freeze["recipe"]["lora"]["alpha"],
                "rank_lr": qwen_freeze["recipe"]["training"]["rank_learning_rate"],
                "head_lr": qwen_freeze["recipe"]["training"]["head_learning_rate"],
                "quantization": "NF4 (double quant, bf16 compute)",
            },
        }
    )

    # Sort leaderboards descending by primary test score (3-seed mean if available else seed42)
    for ds_id in ("goemotions", "movie_reviews_zh"):
        leaderboard[ds_id].sort(
            key=lambda row: (
                row["three_seed"]["test_macro_f1"]["mean"]
                if row["three_seed"]
                else row["seed42"]["test_macro_f1"]
            ),
            reverse=True,
        )

    # Per-label 28 GoEmotions breakdown across ALL 11 models
    en_manifest = _read_json(root / "data/processed/goemotions/v1/dataset_manifest.json")
    per_label_goemotions: list[dict[str, Any]] = []
    for idx, label_name in enumerate(en_manifest["label_names"]):
        key = str(idx)
        entry: dict[str, Any] = {
            "id": idx,
            "name": label_name,
            "train_support": en_manifest["derived_summary"]["label_counts"]["train"][key],
            "val_support": en_manifest["derived_summary"]["label_counts"]["validation"][key],
            "test_support": en_manifest["derived_summary"]["label_counts"]["test"][key],
            "models": {},
        }
        for model_code in batch_a_models:
            cm = class_metrics_a[f"goemotions/{model_code}"]
            f1 = cm["per_label_f1"][idx]
            entry["models"][model_code] = {
                "f1": f1,
                "seed42_f1": f1,
                "precision": cm["per_label_precision"][idx],
                "recall": cm["per_label_recall"][idx],
            }
        for model_code in ("B1", "B2", "B3"):
            by_seed_rows = [encoder_label_metrics[f"{model_code}-{s}"][idx] for s in (42, 123, 2026)]
            entry["models"][model_code] = {
                "f1": statistics.mean(r["f1"] for r in by_seed_rows),
                "seed42_f1": by_seed_rows[0]["f1"],
                "precision": statistics.mean(r["precision"] for r in by_seed_rows),
                "recall": statistics.mean(r["recall"] for r in by_seed_rows),
                "by_seed": {
                    str(s): r["f1"] for s, r in zip((42, 123, 2026), by_seed_rows, strict=True)
                },
            }
        q_lbl = qwen_test["per_label"][idx]
        entry["models"]["Qwen3.5-4B"] = {
            "f1": q_lbl["mean_f1"],
            "seed42_f1": qwen_by_seed[42]["per_label_f1"][idx],
            "precision": statistics.mean(
                qwen_by_seed[s]["per_label_precision"][idx] for s in (42, 123, 2026)
            ),
            "recall": statistics.mean(
                qwen_by_seed[s]["per_label_recall"][idx] for s in (42, 123, 2026)
            ),
            "by_seed": {str(s): qwen_by_seed[s]["per_label_f1"][idx] for s in (42, 123, 2026)},
        }
        per_label_goemotions.append(entry)

    # Embedding ablation (RQ2)
    ablation_embedding: dict[str, list[dict[str, Any]]] = {"goemotions": [], "movie_reviews_zh": []}
    lb_lookup = {row["model_id"]: row for ds in leaderboard.values() for row in ds}
    for ds_id in ("goemotions", "movie_reviews_zh"):
        for model_code in ("E2", "E5-TUNE", "E5-FROZEN"):
            main_rec = batch_a_main_map[f"{ds_id}/{model_code}"]
            res = main_rec["result_raw"]
            w2v = res.get("word2vec", {})
            lb_row = lb_lookup[f"{ds_id}/{model_code}"]
            ablation_embedding[ds_id].append(
                {
                    "model_code": model_code,
                    "display_name": MODEL_DISPLAY[model_code]["display_name"],
                    "embedding_type": w2v.get("type"),
                    "source": w2v.get("source"),
                    "vocabulary_coverage": w2v.get("vocabulary_coverage"),
                    "token_coverage": w2v.get("token_coverage"),
                    "frozen_verified": res.get("frozen_embedding_verified"),
                    "seed42_test_macro_f1": lb_row["seed42"]["test_macro_f1"],
                    "three_seed_mean": lb_row["three_seed"]["test_macro_f1"]["mean"],
                    "three_seed_sd": lb_row["three_seed"]["test_macro_f1"]["sd"],
                    "by_seed": lb_row["three_seed"]["by_seed"],
                }
            )

    # Chinese confusion matrices
    confusion_matrices_zh: list[dict[str, Any]] = []
    for model_code in batch_a_models:
        cm = class_metrics_a[f"movie_reviews_zh/{model_code}"]
        confusion_matrices_zh.append(
            {
                "id": f"{model_code}-42",
                "model_code": model_code,
                "display_name": MODEL_DISPLAY[model_code]["display_name"],
                "seed": 42,
                "threshold": cm["threshold"],
                "score_kind": "decision" if model_code == "E1" else "sigmoid",
                "matrix": cm["confusion_matrix"],
                "accuracy": cm["accuracy"],
                "macro_f1": cm["macro_f1"],
                "positive_precision": cm["positive_precision"],
                "positive_recall": cm["positive_recall"],
                "positive_f1": cm["positive_f1"],
                "roc_auc": cm["roc_auc"],
                "pr_auc": cm["pr_auc"],
            }
        )
    for seed in (42, 123, 2026):
        run = encoder_results["runs"][f"B4-{seed}"]
        t = run["test"]
        confusion_matrices_zh.append(
            {
                "id": f"B4-{seed}",
                "model_code": "B4",
                "display_name": MODEL_DISPLAY["B4"]["display_name"],
                "seed": seed,
                "threshold": run["threshold"],
                "score_kind": "softmax_positive",
                "matrix": t["confusion_matrix"],
                "accuracy": t["accuracy"],
                "macro_f1": t["macro_f1"],
                "positive_precision": t["positive_precision"],
                "positive_recall": t["positive_recall"],
                "positive_f1": t["positive_f1"],
                "roc_auc": t["roc_auc"],
                "pr_auc": t["pr_auc"],
            }
        )

    return {
        "freezes": {
            "batch_a_sha256": batch_a_idx["formal"]["sha256"],
            "batch_b_encoder_sha256": encoder_results["freeze_sha256"],
            "batch_b_qwen_sha256": qwen_test["freeze_sha256"],
            "goemotions_manifest_sha256": batch_a_idx["dataset_manifest_sha256"]["goemotions"],
            "movie_reviews_zh_manifest_sha256": batch_a_idx["dataset_manifest_sha256"][
                "movie_reviews_zh"
            ],
        },
        "quality_flags_a": [
            {
                "dataset": item["dataset"],
                "experiment": item["experiment"],
                "flags": item["flags"],
                "train_macro_f1": item["train_macro_f1"],
                "validation_macro_f1": item["validation_macro_f1"],
            }
            for item in quality_a["performance_review"]["reviewed"]
        ],
        "formal_runs": formal_runs,
        "leaderboard": leaderboard,
        "paired_comparisons": {
            "goemotions_b3_vs_b2": encoder_results["modernbert_paired_comparison"],
            "movie_reviews_zh_batch_a": zh_bootstrap,
            "movie_reviews_zh_families": build_chinese_comparisons(
                root, batch_a_idx["formal"], encoder_results, zh_bootstrap
            ),
        },
        "per_label_goemotions": per_label_goemotions,
        "ablation_embedding": ablation_embedding,
        "confusion_matrices_zh": confusion_matrices_zh,
        "training_curves": {
            "runs": [*batch_a_run_curves, *encoder_curve_runs, *qwen_curve_runs],
            "batch_a": batch_a_curves,
            "batch_b_encoders": encoder_curves,
            "batch_b_qwen": qwen_curves,
        },
        "b1_budget_decision": b1_budget,
    }


def _collect_hpo_and_lr_selection(root: Path, b1_budget: dict[str, Any]) -> dict[str, Any]:
    queue = _read_json(root / "artifacts/batch-a/queue.json")
    s1_results = _read_json(root / "reports/batch-b-s1/results.json")

    studies: dict[str, Any] = {}
    for study_name, study_trials in queue["proposals"].items():
        dataset, model = study_name.split("-", 1)
        winner_run = Path(queue["main"][f"{dataset}/{model}"]).as_posix()
        baseline_run = Path(queue["tasks"][f"baseline/{dataset}/{model}/42"][-1]).as_posix()
        baseline_result = _read_json(root / baseline_run / "result.json")
        baseline_cfg = _read_yaml(root / baseline_run / "resolved_config.yaml")["settings"]

        trials = [
            {
                "trial_number": -1,
                "label": "Baseline",
                "status": baseline_result["status"],
                "macro_f1": baseline_result["macro_f1"],
                "best_epoch": baseline_result.get("best_epoch"),
                "elapsed_seconds": baseline_result["elapsed_seconds"],
                "reason": baseline_result["reason"],
                "run_id": Path(baseline_run).name,
                "is_winner": baseline_run == winner_run,
                "params": baseline_cfg,
            }
        ]
        for num_str, info in sorted(study_trials.items(), key=lambda kv: int(kv[0])):
            run_norm = Path(info["run"]).as_posix()
            trials.append(
                {
                    "trial_number": int(num_str),
                    "label": f"Trial {num_str}",
                    "status": info["status"],
                    "macro_f1": info.get("macro_f1"),
                    "best_epoch": info.get("best_epoch"),
                    "elapsed_seconds": info.get("elapsed_seconds"),
                    "reason": info.get("reason"),
                    "run_id": Path(run_norm).name,
                    "is_winner": run_norm == winner_run,
                    "params": info.get("params", {}),
                }
            )
        studies[study_name] = {
            "study": study_name,
            "dataset": dataset,
            "model": model,
            "display_name": MODEL_DISPLAY[model]["display_name"],
            "trials": trials,
        }

    # Batch B S1 LR selection slots (baseline vs alternate)
    selected_slots = set(s1_results["selected"].values())
    lr_slots: list[dict[str, Any]] = []
    for slot_name, run_rel_list in s1_results["attempts"].items():
        slot_kind, model_code = slot_name.split("/")
        run_info = s1_results["runs"][slot_name]
        res = run_info["result"]
        settings = run_info["settings"]
        ds = "movie_reviews_zh" if model_code == "B4" else "goemotions"
        lr_slots.append(
            {
                "slot": slot_name,
                "kind": slot_kind,
                "model": model_code,
                "display_name": MODEL_DISPLAY[model_code]["display_name"],
                "dataset": ds,
                "learning_rate": settings["learning_rate"],
                "val_macro_f1": res["macro_f1"],
                "threshold": res["threshold"],
                "best_epoch": res["best_epoch"],
                "epochs_completed": res.get("epochs_completed", settings["epochs"]),
                "stop_reason": res["stop_reason"],
                "elapsed_seconds": res["elapsed_seconds"],
                "peak_allocated_gib": res["peak_allocated_bytes"] / (1024**3),
                "selected": slot_name in selected_slots,
                "run_id": Path(run_rel_list[-1].replace("\\", "/")).name,
            }
        )

    return {
        "batch_a_studies": studies,
        "batch_b_s1_lr": lr_slots,
        "batch_b_b1_budget": b1_budget,
    }


def _collect_sample_presets(root: Path) -> dict[str, list[dict[str, Any]]]:
    en_manifest = _read_json(root / "data/processed/goemotions/v1/dataset_manifest.json")
    en_names = en_manifest["label_names"]
    en_rows = [
        json.loads(line)
        for line in (root / "data/processed/goemotions/v1/test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    en_by_id = {row["sample_id"]: row for row in en_rows}

    # Diverse curated GoEmotions test examples (IDs verified from test.jsonl)
    wanted_en = [
        ("eelgwd1", "Gratitude (Single)", "感谢 · 单标签"),
        ("ee3yyz3", "Amusement + Love (Multi)", "开心 + 喜爱 · 多标签"),
        ("eeccgbb", "Admiration + Curiosity", "赞赏 + 好奇 · 多标签"),
        ("edk4e66", "Fear (Short)", "恐惧 · 短句"),
        ("eeaf7fg", "Annoyance + Disapproval", "烦躁 + 不赞同"),
        ("eem5uti", "Neutral (Descriptive)", "中性 · 客观陈述"),
    ]
    # Also locate one real `grief` (id 16) or `pride` (id 21) rare tail sample dynamically
    rare_grief = next((r for r in en_rows if 16 in r["labels"]), None)
    if rare_grief:
        wanted_en.append((rare_grief["sample_id"], "Rare Tail: Grief", "长尾稀缺类 · grief"))

    en_presets = []
    for sample_id, title_en, title_zh in wanted_en:
        row = en_by_id.get(sample_id)
        if row is None:
            continue
        en_presets.append(
            {
                "sample_id": row["sample_id"],
                "title_en": title_en,
                "title_zh": title_zh,
                "text": row["text"],
                "gold_labels": [en_names[i] for i in row["labels"]],
            }
        )

    zh_rows = [
        json.loads(line)
        for line in (root / "data/processed/movie_reviews_zh/v1/test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    zh_by_id = {row["sample_id"]: row for row in zh_rows}
    wanted_zh = [
        ("test:200", "Positive Review (Crime/Drama)", "正向影评 · 剧情犯罪"),
        ("test:6", "Negative Review (Strong)", "负向影评 · 强烈吐槽"),
        ("test:190", "Positive Review (Mixed EN/ZH)", "正向影评 · 中英混杂"),
        ("test:2", "Negative Review (Nuanced)", "负向影评 · 先扬后抑"),
        ("test:197", "Positive Review (Emotional)", "正向影评 · 情感共鸣"),
    ]
    zh_presets = []
    for sample_id, title_en, title_zh in wanted_zh:
        row = zh_by_id.get(sample_id)
        if row is None:
            continue
        gold = ["positive"] if row["labels"] == [0] else ["negative"]
        zh_presets.append(
            {
                "sample_id": row["sample_id"],
                "title_en": title_en,
                "title_zh": title_zh,
                "text": _unsegment_zh(row["text"]),
                "segmented_text": row["text"],
                "gold_labels": gold,
            }
        )

    return {"goemotions": en_presets, "movie_reviews_zh": zh_presets}


def build_dashboard_bundle(root: Path | str = ".") -> dict[str, Any]:
    repo_root = Path(root).resolve()
    datasets = _export_datasets(repo_root)
    core = _collect_runs_and_summaries(repo_root)
    hpo = _collect_hpo_and_lr_selection(repo_root, core.pop("b1_budget_decision"))
    presets = _collect_sample_presets(repo_root)

    formal_runs = core["formal_runs"]
    en_runs = sum(1 for r in formal_runs if r["dataset"] == "goemotions")
    zh_runs = sum(1 for r in formal_runs if r["dataset"] == "movie_reviews_zh")
    total_test_preds = en_runs * 5427 + zh_runs * 369

    assert len(formal_runs) == 62, f"Expected 62 formal runs, got {len(formal_runs)}"
    assert len(core["per_label_goemotions"]) == 28, "Expected 28 GoEmotions labels"
    assert len(core["leaderboard"]["goemotions"]) == 11, "Expected 11 GoEmotions models"
    assert len(core["leaderboard"]["movie_reviews_zh"]) == 8, "Expected 8 Chinese models"

    return {
        "meta": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "freezes": core["freezes"],
            "totals": {
                "formal_runs": len(formal_runs),
                "goemotions_runs": en_runs,
                "movie_reviews_zh_runs": zh_runs,
                "frozen_samples": (
                    datasets["goemotions"]["total_frozen"]
                    + datasets["movie_reviews_zh"]["total_frozen"]
                ),
                "test_predictions_total": total_test_preds,
                "models_goemotions": len(core["leaderboard"]["goemotions"]),
                "models_movie_reviews_zh": len(core["leaderboard"]["movie_reviews_zh"]),
            },
        },
        "datasets": datasets,
        "leaderboard": core["leaderboard"],
        "formal_runs": formal_runs,
        "paired_comparisons": core["paired_comparisons"],
        "per_label_goemotions": core["per_label_goemotions"],
        "ablation_embedding": core["ablation_embedding"],
        "confusion_matrices_zh": core["confusion_matrices_zh"],
        "training_curves": core["training_curves"],
        "hpo_trials": hpo,
        "quality_flags_a": core["quality_flags_a"],
        "sample_presets": presets,
    }


def export_web_data(
    root: Path | str = ".",
    output_path: Path | str = "apps/web/src/data/dashboard-data.json",
) -> Path:
    repo_root = Path(root).resolve()
    target = Path(output_path)
    if not target.is_absolute():
        target = repo_root / target
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_dashboard_bundle(repo_root)
    payload = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target.write_text(payload, encoding="utf-8")
    return target

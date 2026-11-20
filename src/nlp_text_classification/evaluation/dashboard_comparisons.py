"""Paired follow-up comparisons using published test predictions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from nlp_text_classification.evaluation.batch_report import paired_macro_f1_interval
from nlp_text_classification.experiments.suite import checked_file, read_json

DATASET = "movie_reviews_zh"
FAMILIES = {
    "architecture": (
        ("E2", "E1"),
        ("E3-LSTM", "E2"),
        ("E3-BiLSTM", "E3-LSTM"),
        ("B4", "E3-BiLSTM"),
    ),
    "ablation": (("E5-TUNE", "E2"), ("E5-FROZEN", "E2"), ("E5-TUNE", "E5-FROZEN")),
}


def build_chinese_comparisons(
    root: Path,
    batch_a: dict[str, Any],
    encoders: dict[str, Any],
    originally_planned: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute two-sided, Bonferroni-adjusted intervals for one family of seven."""
    macbert = encoders["runs"]["B4-42"]
    paths = {
        entry["key"].split("/")[1]: (
            root / entry["test_manifest"],
            batch_a["sha256"],
            entry["test_macro_f1"],
        )
        for entry in batch_a["runs"]
        if entry["role"] == "main"
        and entry["seed"] == 42
        and entry["key"].startswith(f"{DATASET}/")
    }
    paths["B4"] = (
        checked_file(
            root / macbert["test_evidence"] / "evaluation_manifest.json",
            macbert["test_manifest_sha256"],
        ),
        encoders["freeze_sha256"],
        macbert["test"]["macro_f1"],
    )
    labels: np.ndarray | None = None
    sample_ids: list[str] = []
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    for code in dict.fromkeys(
        code for pairs in FAMILIES.values() for pair in pairs for code in pair
    ):
        manifest_file, freeze, expected_score = paths[code]
        manifest = read_json(manifest_file)
        if manifest["freeze_sha256"] != freeze:
            raise ValueError(f"Prediction freeze differs for {code}")
        name = "predictions_test.jsonl"
        file = checked_file(manifest_file.parent / name, manifest["files"][name])
        with file.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        ids = [item["sample_id"] for item in records]
        truth = np.asarray([item["target"] for item in records])
        pred = np.asarray([item["predicted"] for item in records])
        if (
            len(ids) != 369
            or len(set(ids)) != len(ids)
            or truth.shape != (len(ids), 1)
            or pred.shape != truth.shape
            or not np.isin(truth, [0, 1]).all()
            or not np.isin(pred, [0, 1]).all()
        ):
            raise ValueError(f"Expected unique binary test predictions for {code}")
        if labels is None:
            sample_ids, labels = ids, truth
        elif ids != sample_ids or not np.array_equal(labels, truth):
            raise ValueError(f"Test IDs or labels do not align for {code}")
        score = float(f1_score(truth.ravel(), pred.ravel(), average="macro", labels=[0, 1]))
        if not np.isclose(score, expected_score, atol=1e-12, rtol=0):
            raise ValueError(f"Test Macro-F1 differs from the published result for {code}")
        predictions[code], scores[code] = pred, score

    count = sum(len(pairs) for pairs in FAMILIES.values())
    alpha = 0.05
    bootstrap = {"confidence_level": 1 - alpha / count, "n_resamples": 100_000, "seed": 20261117}
    planned = {(row["model_a"], row["model_b"]) for row in originally_planned}
    if not planned.issubset({pair for pairs in FAMILIES.values() for pair in pairs}):
        raise ValueError("The dashboard family must cover the original planned Chinese comparisons")
    if labels is None:
        raise ValueError("No aligned test predictions")
    result: dict[str, Any] = {
        "dataset": DATASET,
        "seed": 42,
        "n": len(sample_ids),
        "correction": "bonferroni",
        "familywise_alpha": alpha,
        "comparison_count": count,
        "confidence_level": bootstrap["confidence_level"],
        "n_resamples": bootstrap["n_resamples"],
        "resampling_seed": bootstrap["seed"],
        "groups": {},
    }
    for group, pairs in FAMILIES.items():
        rows = []
        for model, reference in pairs:
            bounds = paired_macro_f1_interval(
                labels, predictions[model], predictions[reference], bootstrap
            )
            rows.append(
                {
                    "model_a": model,
                    "model_b": reference,
                    "difference": scores[model] - scores[reference],
                    "ci_low": bounds["lower"],
                    "ci_high": bounds["upper"],
                    "originally_planned": (model, reference) in planned,
                }
            )
        result["groups"][group] = rows
    return result

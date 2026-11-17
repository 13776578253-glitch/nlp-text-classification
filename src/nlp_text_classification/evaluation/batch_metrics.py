"""Shared Batch A threshold search, metrics and prediction evidence."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from nlp_text_classification.data.contracts import Example


def _divide(numerator, denominator):
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0
    )


def _counts(targets: np.ndarray, predicted: np.ndarray, binary: bool) -> tuple:
    if binary:
        targets = np.column_stack((~targets[:, 0], targets[:, 0]))
        predicted = np.column_stack((~predicted[:, 0], predicted[:, 0]))
    tp = np.sum(targets & predicted, axis=0)
    fp = np.sum(~targets & predicted, axis=0)
    fn = np.sum(targets & ~predicted, axis=0)
    return tp, fp, fn


def _validate(targets: np.ndarray, scores: np.ndarray, binary: bool) -> np.ndarray:
    if targets.shape != scores.shape or targets.ndim != 2 or not len(targets):
        raise ValueError("Targets and scores must share a nonempty two-dimensional shape")
    if binary != (targets.shape[1] == 1):
        raise ValueError("Wrong score width for the task")
    if not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Non-finite scores or invalid target encoding")
    return targets.astype(bool)


def select_threshold(targets: np.ndarray, scores: np.ndarray, binary: bool,
                     selection: dict, kind: str) -> dict:
    truth = _validate(targets, scores, binary)
    grid = selection["threshold_grids"][kind]
    start, stop = (Decimal(str(value)) for value in grid["range_inclusive"])
    step = Decimal(str(grid["step"]))
    candidates = []
    default = Decimal(str(grid["default"]))
    for offset in range(int((stop - start) / step) + 1):
        threshold = start + offset * step
        tp, fp, fn = _counts(truth, scores >= float(threshold), binary)
        macro = float(_divide(2 * tp, 2 * tp + fp + fn).mean())
        candidates.append((threshold, macro))
    best_score = max(macro for _, macro in candidates)
    tied = (item for item in candidates if best_score - item[1] <= selection["score_tie_atol"])
    chosen, macro = min(tied, key=lambda item: (abs(item[0] - default), item[0]))
    return {
        "threshold": float(chosen), "macro_f1": macro, "score_kind": kind,
        "candidates": [{"threshold": float(t), "macro_f1": f} for t, f in candidates],
    }


def measure(targets: np.ndarray, scores: np.ndarray, threshold: float,
            binary: bool, *, auc: bool = True) -> dict:
    truth = _validate(targets, scores, binary)
    predicted = scores >= threshold
    tp, fp, fn = _counts(truth, predicted, binary)
    label_f1 = _divide(2 * tp, 2 * tp + fp + fn)
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    _, frequencies = np.unique(predicted, axis=0, return_counts=True)
    result = {
        "macro_f1": float(label_f1.mean()),
        "micro_f1": float(_divide(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum())),
        "per_label_f1": label_f1.tolist(),
        "per_label_precision": precision.tolist(),
        "per_label_recall": recall.tolist(),
        "per_label_support": (tp + fn).tolist(),
        "threshold": float(threshold),
        "modal_prediction_share": float(frequencies.max() / len(predicted)),
        "predicted_label_fraction": (1.0 if binary else float(predicted.any(axis=0).mean())),
    }
    if binary:
        result.update({
            "accuracy": float((truth == predicted).mean()),
            "per_class_f1": result["per_label_f1"],
            "per_class_support": result["per_label_support"],
            "positive_f1": float(label_f1[1]),
            "positive_precision": float(precision[1]),
            "positive_recall": float(recall[1]),
            "confusion_matrix": [[int(tp[0]), int(fp[1])], [int(fn[1]), int(tp[1])]],
        })
        if auc:
            valid = bool(truth.any() and not truth.all())
            result["roc_auc"] = float(roc_auc_score(truth[:, 0], scores[:, 0])) if valid else None
    else:
        result.update({
            "samples_f1": float(_divide(2 * (truth & predicted).sum(1),
                                        truth.sum(1) + predicted.sum(1)).mean()),
            "hamming_loss": float((truth != predicted).mean()),
            "subset_accuracy": float((truth == predicted).all(1).mean()),
        })
    if auc:
        valid = np.flatnonzero(truth.any(0) & ~truth.all(0))
        per_label = {int(k): float(average_precision_score(truth[:, k], scores[:, k]))
                     for k in valid}
        result["pr_auc"] = float(np.mean(list(per_label.values()))) if per_label else None
        result["auc_undefined_score_columns"] = sorted(set(range(truth.shape[1])) - per_label.keys())
    return result


def write_predictions(path: Path, rows: list[Example], targets: np.ndarray,
                      scores: np.ndarray, threshold: float) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row, target, score in zip(rows, targets, scores, strict=True):
            payload = {"sample_id": row.sample_id, "raw_labels": row.labels,
                       "target": target.tolist(), "scores": score.tolist(),
                       "predicted": (score >= threshold).astype(np.int8).tolist()}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_predictions(
    path: Path, *, score_kind: str
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        items = [json.loads(line) for line in stream]
    keys = [row["sample_id"] for row in items]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate prediction ID in {path}")
    scores = np.asarray([row["scores"] for row in items], dtype=np.float64)
    if score_kind == "sigmoid":
        # JSON preserves float32 score values but has no dtype. Restore it so
        # NumPy rounds scalar threshold comparisons just as it did at inference.
        native_scores = scores.astype(np.float32)
        if not np.array_equal(scores, native_scores):
            raise ValueError(f"Neural scores do not round-trip as float32 in {path}")
        scores = native_scores
    elif score_kind != "decision":
        raise ValueError(f"Unknown score kind: {score_kind}")
    return (keys, np.asarray([row["target"] for row in items]), scores,
            np.asarray([row["predicted"] for row in items]))

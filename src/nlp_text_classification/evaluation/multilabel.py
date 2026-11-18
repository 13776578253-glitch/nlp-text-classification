"""Shared evaluation protocol: thresholds are selected ONLY on validation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score


def indicator_matrix(rows: list, n_labels: int) -> np.ndarray:
    matrix = np.zeros((len(rows), n_labels), dtype=np.int8)
    for i, row in enumerate(rows):
        matrix[i, list(row.labels)] = 1
    return matrix


def as_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("Expected scores of shape [examples, labels]")
    return (scores >= threshold).astype(np.int8)


def select_threshold(
    y_true: np.ndarray, scores: np.ndarray, *, score_kind: str
) -> dict:
    if y_true.shape != scores.shape or y_true.ndim != 2:
        raise ValueError("Validation labels and scores must have identical 2D shapes")
    if score_kind == "probability":
        candidates = np.round(np.arange(0.05, 0.951, 0.05), 3)
    elif score_kind == "decision":
        candidates = np.round(np.arange(-1.5, 1.501, 0.1), 3)
    else:
        raise ValueError(f"Unknown score kind: {score_kind}")
    ranks = [(float(f1_score(y_true, as_predictions(scores, float(t)), average="micro", zero_division=0)), float(t)) for t in candidates]
    # Deterministic tie break: choose the lower threshold.
    best = min(ranks, key=lambda item: (-item[0], item[1]))
    return {"threshold": best[1], "validation_micro_f1": best[0], "score_kind": score_kind}


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    if y_true.shape != scores.shape or y_true.ndim != 2:
        raise ValueError("Labels and score shape mismatch")
    predicted = as_predictions(scores, threshold)
    per_label_f1 = np.asarray(f1_score(y_true, predicted, average=None, zero_division=0))
    return {
        "micro_f1": float(f1_score(y_true, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, predicted, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, predicted, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, predicted, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, predicted)),
        "subset_accuracy": float(np.mean(np.all(predicted == y_true, axis=1))),
        "support_per_label": [int(x) for x in y_true.sum(axis=0)],
        "f1_per_label": [float(x) for x in per_label_f1],
        "threshold": float(threshold),
    }

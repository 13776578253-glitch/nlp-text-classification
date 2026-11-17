import numpy as np
from sklearn.metrics import f1_score

from nlp_text_classification.data.contracts import Example
from nlp_text_classification.evaluation.batch_metrics import (
    measure,
    read_predictions,
    select_threshold,
    write_predictions,
)


def test_batch_macro_f1_uses_both_binary_classes_and_default_tiebreak():
    y = np.array([[1], [0], [1], [0]])
    scores = np.array([[0.9], [0.1], [0.8], [0.2]])
    grid = {"threshold_grids": {"sigmoid": {
        "range_inclusive": [0.1, 0.9], "step": 0.1, "default": 0.5
    }}, "score_tie_atol": 1e-12}
    tuning = select_threshold(y, scores, True, grid, "sigmoid")
    assert tuning["threshold"] == 0.5
    metrics = measure(y, scores, 0.85, True)
    assert metrics["macro_f1"] == f1_score(y, scores >= 0.85, average="macro")
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_batch_multilabel_macro_f1_follows_frozen_column_order():
    y = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    scores = np.array([[0.9, 0.6, 0.3], [0.1, 0.9, 0.1],
                       [0.4, 0.2, 0.1], [0.2, 0.8, 0.1]])
    metrics = measure(y, scores, 0.5, False)
    assert metrics["macro_f1"] == f1_score(y, scores >= 0.5, average="macro")
    assert metrics["per_label_support"] == [2, 3, 1]


def test_neural_predictions_replay_float32_threshold_boundaries(tmp_path):
    y = np.array([[1], [0]])
    scores = np.array([[0.95], [0.1]], dtype=np.float32)
    assert float(scores[0, 0]) < 0.95
    selection = {"threshold_grids": {"sigmoid": {
        "range_inclusive": [0.9, 0.95], "step": 0.05, "default": 0.95,
    }}, "score_tie_atol": 1e-12}
    rows = [Example("positive", "", (1,), "validation"),
            Example("negative", "", (0,), "validation")]
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, rows, y, scores, 0.95)
    _, targets, restored, predicted = read_predictions(path, score_kind="sigmoid")
    assert restored.dtype == np.float32
    assert np.array_equal(restored >= 0.95, predicted)
    assert select_threshold(targets, restored, True, selection, "sigmoid") == (
        select_threshold(y, scores, True, selection, "sigmoid")
    )

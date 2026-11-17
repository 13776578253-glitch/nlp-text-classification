import numpy as np
from sklearn.metrics import f1_score

from nlp_text_classification.evaluation.batch_metrics import measure, select_threshold


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

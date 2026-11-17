import numpy as np
import pytest

from nlp_text_classification.evaluation.multilabel import as_predictions, evaluate, select_threshold


def test_validation_threshold_selection_and_f1():
    truth = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.8], [0.1, 0.2]])
    tuning = select_threshold(truth, scores, score_kind="probability")
    metrics = evaluate(truth, scores, tuning["threshold"])
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_wrong_shapes_fail():
    with pytest.raises(ValueError, match="identical 2D"):
        select_threshold(np.zeros((3, 2)), np.zeros((3, 1)), score_kind="decision")
    with pytest.raises(ValueError, match="Expected scores"):
        as_predictions(np.zeros((3,)), 0.5)

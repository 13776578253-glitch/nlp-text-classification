import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from nlp_text_classification.evaluation.batch_b_final_report import paired_bootstrap
from nlp_text_classification.experiments.batch_b_final import (
    MODELS,
    SEEDS,
    publish_once,
    require_matrix,
)


def test_paired_bootstrap_matches_sklearn_and_uses_shared_rows():
    truth = np.array([[0, 1], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=bool)
    first = truth.copy()
    second = np.array([[1, 0], [0, 1], [0, 0], [1, 1], [0, 1]], dtype=bool)
    pairs = [(first, second), (second, first), (second, second)]
    config = {"metrics": ["macro_f1", "micro_f1", "subset_accuracy"], "models": ["B3", "B2"],
              "unit": "aligned_test_row", "reuse_rows_across_fixed_seeds": True,
              "n_resamples": 30, "seed": 42}
    points, draws = paired_bootstrap(truth, pairs, config)
    expected = [[f1_score(truth, pred, average=average) for average in ("macro", "micro")]
                + [accuracy_score(truth, pred)] for pred in (first, second)]
    np.testing.assert_allclose(points[0], expected)
    np.testing.assert_array_equal(draws[:, 0], -draws[:, 1])
    np.testing.assert_array_equal(draws[:, 2], 0)


def test_freeze_matrix_rejects_duplicate_runs_and_reassigned_primary():
    rows = [
        {"model": model, "seed": seed, "run": f"{model}/{seed}",
         "role": "primary" if seed == 42 else "replication"}
        for model in MODELS for seed in SEEDS
    ]
    require_matrix(rows)
    with pytest.raises(ValueError, match="distinct Run"):
        require_matrix(rows[:-1] + [rows[-1] | {"run": rows[0]["run"]}])
    with pytest.raises(ValueError, match="roles changed"):
        require_matrix([rows[0] | {"role": "replication"}, *rows[1:]])


def test_published_freeze_cannot_be_overwritten(tmp_path):
    path = tmp_path / "freeze.json"
    publish_once(path, b"original")
    publish_once(path, b"original")
    with pytest.raises(ValueError, match="Published bytes differ"):
        publish_once(path, b"changed")
    assert path.read_bytes() == b"original"

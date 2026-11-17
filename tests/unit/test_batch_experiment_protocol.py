from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
pytest.importorskip("optuna")

from nlp_text_classification.evaluation.batch_report import paired_macro_f1_interval
from nlp_text_classification.experiments.batch_scheduler import Batch
from nlp_text_classification.experiments.performance import flags
from nlp_text_classification.experiments.suite import load_suite
from nlp_text_classification.training.progress import Progress


def test_optuna_extension_only_until_four_accepted(tmp_path, monkeypatch):
    batch = Batch.__new__(Batch)
    batch.config, _ = load_suite(Path("configs/experiment_suites/batch_a.yaml"))
    batch.output = tmp_path
    batch.state_path = tmp_path / "queue.json"
    batch.state = {"proposals": {}}
    batch.progress = Progress(tmp_path / "progress.md", "unit test")
    attempts = []

    def attempt(dataset, experiment, cohort, seed, params, pruning, *, suffix):
        slot = int(suffix.removeprefix("/"))
        attempts.append(slot)
        status = "PASS" if slot in {0, 1, 5, 6, 7} else "PRUNED"
        if slot >= 6:
            assert not pruning(8, 0.0)
        return tmp_path / f"trial{slot}", {
            "status": status, "macro_f1": slot / 10, "reason": status,
        }

    monkeypatch.setattr(batch, "attempt", attempt)
    base = tmp_path / "base"
    result = batch.search("goemotions", "E2", (base, {"status": "PASS", "macro_f1": 0.2}))
    assert attempts == list(range(7))
    assert result.name == "trial6"


def test_paired_bootstrap_uses_same_indices_for_identical_predictions():
    y = np.array([1, 0, 1, 0, 0, 1])
    predicted = np.array([1, 0, 0, 1, 0, 1])
    params = {"seed": 42, "n_resamples": 120, "confidence_level": 0.95}
    interval = paired_macro_f1_interval(y, predicted, predicted, params)
    assert interval["lower"] == interval["upper"] == 0


def test_performance_review_distinguishes_technical_success_from_poor_results():
    config, _ = load_suite(Path("configs/experiment_suites/batch_a.yaml"))
    quality_result = {"macro_f1": 0.15, "modal_prediction_share": 0.999,
                      "predicted_label_fraction": 0.05}
    result = flags(quality_result, quality_result, 0.04, config["performance_review"])
    assert result == ["degenerate_predictions", "underfitting", "unstable_across_seeds"]

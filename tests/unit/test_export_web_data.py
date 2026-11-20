"""Tests for the static dashboard data exporter."""

from __future__ import annotations

import json
import math
from pathlib import Path

from nlp_text_classification.evaluation.export_web_data import build_dashboard_bundle


def test_build_dashboard_bundle_counts_and_integrity(frozen_experiment_root: Path):
    bundle = build_dashboard_bundle(frozen_experiment_root)

    assert bundle["meta"]["totals"] == {
        "formal_runs": 62,
        "goemotions_runs": 34,
        "movie_reviews_zh_runs": 28,
        "frozen_samples": 80046,
        "test_predictions_total": 194850,
        "models_goemotions": 11,
        "models_movie_reviews_zh": 8,
    }
    assert len(bundle["formal_runs"]) == 62
    assert len(bundle["per_label_goemotions"]) == 28
    assert len(bundle["paired_comparisons"]["movie_reviews_zh_batch_a"]) == 5
    assert "mean_of_fixed_seeds" in bundle["paired_comparisons"]["goemotions_b3_vs_b2"]
    assert len(bundle["ablation_embedding"]["goemotions"]) == 3
    assert len(bundle["ablation_embedding"]["movie_reviews_zh"]) == 3
    assert len(bundle["confusion_matrices_zh"]) == 10  # 7 Batch A + 3 B4 seeds
    assert len(bundle["hpo_trials"]["batch_a_studies"]) == 10
    assert len(bundle["hpo_trials"]["batch_b_s1_lr"]) == 8
    assert len(bundle["sample_presets"]["goemotions"]) >= 6
    assert len(bundle["sample_presets"]["movie_reviews_zh"]) == 5

    # Verify serialized JSON has zero NaN/Infinity
    text = json.dumps(bundle, allow_nan=False)
    assert len(text) > 10_000
    for row in bundle["per_label_goemotions"]:
        assert len(row["models"]) == 11
        for metrics in row["models"].values():
            assert math.isfinite(metrics["f1"])

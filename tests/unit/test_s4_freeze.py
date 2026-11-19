"""Tests for the S4 validation-only freeze contract."""

from pathlib import Path

import pytest

from nlp_text_classification.experiments import batch_b_s4_freeze as freeze


def test_freeze_never_references_the_test_partition():
    source = Path(freeze.__file__).read_text(encoding="utf-8")
    assert "test.jsonl" not in source
    assert "load_partition" not in source
    assert "splits\"][\"test\"]" in source or 'splits"]["test"]' in source


def test_freeze_sources_cover_every_bound_file():
    bound = set(freeze.BOUND_SOURCES)
    assert "batch_b_s4_formal.py" in bound
    assert "transformer_runner.py" in bound
    assert "batch_metrics.py" in bound
    assert len(bound) == 13
    assert set(freeze.FREEZE_SOURCES) == {"batch_b_s4_freeze.py"}


def test_seeds_are_the_approved_three():
    assert freeze.SEEDS == (42, 123, 2026)


def test_publish_once_is_idempotent(tmp_path):
    target = tmp_path / "frozen" / "manifest.json"
    payload = freeze._bytes({"status": "frozen", "value": 1})
    freeze._publish_once(target, payload)
    assert target.read_bytes() == payload
    freeze._publish_once(target, payload)


def test_publish_once_rejects_changed_bytes(tmp_path):
    target = tmp_path / "manifest.json"
    freeze._publish_once(target, freeze._bytes({"value": 1}))
    with pytest.raises(ValueError, match="Published bytes differ"):
        freeze._publish_once(target, freeze._bytes({"value": 2}))


def test_bytes_are_canonical_and_reject_nan():
    first = freeze._bytes({"b": 1, "a": [2, 3]})
    second = freeze._bytes({"a": [2, 3], "b": 1})
    assert first == second
    assert first.endswith(b"\n")
    with pytest.raises(ValueError):
        freeze._bytes({"value": float("nan")})


def test_summary_uses_sample_standard_deviation():
    audits = {"seed-42": {"macro_f1": 0.5, "micro_f1": 0.6, "pr_auc": 0.7, "hamming_loss": 0.1},
              "seed-123": {"macro_f1": 0.6, "micro_f1": 0.7, "pr_auc": 0.8, "hamming_loss": 0.2}}
    summary = freeze._summary(audits)
    assert summary["macro_f1_mean"] == pytest.approx(0.55)
    assert summary["macro_f1_std"] == pytest.approx(0.07071067811865475)
    assert summary["seeds"] == ["123", "42"]

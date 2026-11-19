import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nlp_text_classification.data.contracts import Example
from nlp_text_classification.experiments import batch_b_s4_test as s4
from nlp_text_classification.experiments.suite import TrainingData
from nlp_text_classification.training.progress import Progress


def make_partition(path):
    lines = [{"sample_id": "t1", "text": "a", "labels": [0], "split": "test"},
             {"sample_id": "t2", "text": "b", "labels": [1], "split": "test"},
             {"sample_id": "t3", "text": "c", "labels": [0, 1], "split": "test"}]
    raw = ("\n".join(json.dumps(row) for row in lines) + "\n").encode()
    path.write_bytes(raw)
    return {"count": 3, "sha256": hashlib.sha256(raw).hexdigest()}


def test_one_source_read_and_only_cache_on_replay(tmp_path, monkeypatch):
    source, cache = tmp_path / "test.jsonl", tmp_path / "cache.jsonl"
    record = make_partition(source)
    read_bytes = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(FileNotFoundError):
        s4.test_rows(source, cache, record, set(), allow_source=False)
    assert reads == []
    rows = s4.test_rows(source, cache, record, set(), allow_source=True)
    assert [row.sample_id for row in rows] == ["t1", "t2", "t3"]
    assert s4.test_rows(source, cache, record, set(), allow_source=False) == rows
    assert s4.test_rows(source, cache, record, set(), allow_source=True) == rows
    assert reads.count(source) == 1
    cache.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        s4.test_rows(source, cache, record, set(), allow_source=True)
    assert reads.count(source) == 1


def test_partition_overlap_prevents_cache(tmp_path):
    source, cache = tmp_path / "test.jsonl", tmp_path / "cache.jsonl"
    record = make_partition(source)
    with pytest.raises(ValueError, match="sample IDs"):
        s4.test_rows(source, cache, record, {"t2"}, allow_source=True)
    assert not cache.exists()


def test_failed_freeze_stops_before_opening_test(tmp_path, monkeypatch):
    def reject():
        raise ValueError("freeze failed")

    monkeypatch.setattr(s4, "ROOT", tmp_path)
    monkeypatch.setattr(s4, "verified_freeze", reject)
    with pytest.raises(ValueError, match="freeze failed"):
        s4.run("test")
    assert not list(tmp_path.rglob("test-input.jsonl"))


def test_published_runs_are_only_replayed_and_missing_runs_can_complete(tmp_path):
    rows = [Example("t1", "a", (0,), "test"), Example("t2", "b", (1,), "test"),
            Example("t3", "c", (0, 1), "test")]
    data = TrainingData({}, {"task_type": "multilabel", "label_ids": [0, 1]}, [], [])
    entries = [{"seed": seed, "global_step": 5402, "checkpoint_sha256": str(seed),
                "adapter_sha256": str(seed), "threshold": cutoff, "validation_macro_f1": 0.5}
               for seed, cutoff in zip(s4.SEEDS, (0.2, 0.3, 0.15), strict=True)]
    frozen = {"runs": entries, "label_order": ["a", "b"]}
    (tmp_path / "contract.json").write_bytes(s4._bytes({"encoder_comparison": {}}))
    progress = Progress(tmp_path / "progress.md", "fixture")
    scores = np.asarray([[0.2, 0.02], [0.03, 0.8], [0.99, 0.65]], dtype=np.float32)
    invoked = []

    def infer(entry):
        invoked.append(entry["seed"])
        if entry["seed"] == 123 and invoked == [42, 123]:
            raise RuntimeError("synthetic failure")
        return scores, {"rows": 3}

    with pytest.raises(RuntimeError, match="synthetic failure"):
        s4.evaluate(frozen, data, tmp_path, rows, progress, infer)
    published = tmp_path / "seed-42" / "predictions_test.jsonl"
    prior = published.stat().st_mtime_ns
    assert not (tmp_path / "acceptance.json").exists()
    s4.evaluate(frozen, data, tmp_path, rows, progress, infer)
    receipt = (tmp_path / "acceptance.json").read_bytes()
    s4.evaluate(frozen, data, tmp_path, rows, progress)
    assert invoked == [42, 123, 123, 2026]
    assert prior == published.stat().st_mtime_ns
    assert receipt == (tmp_path / "acceptance.json").read_bytes()
    result = json.loads(receipt)
    assert result["status"] == "PASS" and result["test_read"]
    assert result["runs"][0]["metrics"]["macro_f1"] == 1.0
    assert result["runs"][1]["metrics"]["macro_f1"] < 1.0

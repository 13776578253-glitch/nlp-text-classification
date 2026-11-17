from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")

from nlp_text_classification.experiments.batch_cli import synthetic_data
from nlp_text_classification.experiments.suite import load_suite, read_json
from nlp_text_classification.training.batch_run import resolve, run_attempt
from nlp_text_classification.training.progress import Progress


def test_epoch_checkpoint_resume_reproduces_uninterrupted_predictions(tmp_path):
    config, datasets = load_suite(Path("configs/experiment_suites/batch_a.yaml"))
    data = synthetic_data("movie_reviews_zh", datasets["movie_reviews_zh"])
    resolved = resolve(config, "E3-BiLSTM", {"epoch_cap": 3})
    progress = Progress(tmp_path / "progress.md", "checkpoint recovery test")
    started = []

    def interrupt(epoch, score):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_attempt(data, resolved, tmp_path, progress, cohort="replication",
                    seed=42, device="cpu", pruning=interrupt, on_start=started.append)
    original = started[0]
    assert read_json(original / "result.json")["status"] == "INCOMPLETE"
    resumed = run_attempt(data, resolved, tmp_path, progress, cohort="replication",
                          seed=42, device="cpu", previous=str(original),
                          resume=original / "checkpoint")
    control = run_attempt(data, resolved, tmp_path, progress, cohort="replication",
                          seed=42, device="cpu")
    assert read_json(resumed / "result.json")["status"] == "PASS"
    assert (resumed / "predictions_validation.jsonl").read_bytes() == (
        control / "predictions_validation.jsonl"
    ).read_bytes()
    assert len((resumed / "history.csv").read_text(encoding="utf-8").splitlines()) == 4
    assert read_json(original / "result.json")["status"] == "INCOMPLETE"

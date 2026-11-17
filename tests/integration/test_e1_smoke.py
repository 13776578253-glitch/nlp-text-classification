import json
from pathlib import Path

import pytest

from nlp_text_classification.cli import main
from nlp_text_classification.experiments.e1_svm import predict, train


def synthetic_dataset(path: Path) -> Path:
    path.mkdir()
    (path / "emotions.txt").write_text("\n".join([f"emotion_{i}" for i in range(27)] + ["neutral"]) + "\n", encoding="utf-8")
    # SVM sees positive and negative examples of every label.
    (path / "train.tsv").write_text("".join(
        f"uniqueemotion{i} alpha context\t{i}\ttrain-{i}\n"
        f"another emotion {i} with beta context\t{i}\ttrain-b-{i}\n"
        for i in range(28)
    ), encoding="utf-8")
    (path / "dev.tsv").write_text("".join(
        f"uniqueemotion{i} context\t{i}\tvalidation-{i}\n" for i in range(28)
    ), encoding="utf-8")
    (path / "test.tsv").write_text("".join(
        f"another emotion {i}\t{i}\ttest-{i}\n" for i in range(28)
    ), encoding="utf-8")
    return path


def test_e1_end_to_end_without_test_peeking(tmp_path):
    path = synthetic_dataset(tmp_path / "data")
    (path / "test.tsv").unlink()
    run = train(path, tmp_path / "out", strict_data=False, min_df=1, max_features=2000)
    assert (run / "metrics_validation.json").is_file()
    assert not (run / "metrics_test.json").exists()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["data_sha256"]) == {"train", "validation", "labels"}
    assert manifest["run_scope"] == "standalone_validation"
    assert (run / "config.json").exists()
    assert (run / "predictions_validation.jsonl").exists()
    result = predict(run, ["uniqueemotion1 alpha context"])
    assert len(result[0]["scores"]) == 28
    assert result[0]["score_kind"] == "decision"


def test_legacy_single_run_final_test_entrypoint_is_removed(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["nlp-text-classification", "evaluate-final-e1"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err

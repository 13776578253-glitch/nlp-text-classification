from pathlib import Path

from nlp_text_classification.experiments.e1_svm import evaluate_final, predict, train


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
    run = train(path, tmp_path / "out", strict_data=False, min_df=1, max_features=2000)
    assert (run / "metrics_validation.json").is_file()
    assert not (run / "metrics_test.json").exists()
    assert (run / "config.json").exists()
    assert (run / "predictions_validation.jsonl").exists()
    result = predict(run, ["uniqueemotion1 alpha context"])
    assert len(result[0]["scores"]) == 28
    assert result[0]["score_kind"] == "decision"


def test_frozen_final_eval_is_one_time(tmp_path):
    import pytest
    path = synthetic_dataset(tmp_path / "data")
    run = train(path, tmp_path / "out", strict_data=False, min_df=1, max_features=2000)
    # Synthetic fixture size differs from official release, so strict validation
    # must refuse to call it the official test split.
    with pytest.raises(ValueError, match="rows"):
        evaluate_final(run, path)
    assert not (run / "metrics_test.json").exists()
    metrics = evaluate_final(run, path, strict_data=False)
    assert "macro_f1" in metrics
    assert (run / "metrics_test.json").exists()
    with pytest.raises(FileExistsError):
        evaluate_final(run, path, strict_data=False)

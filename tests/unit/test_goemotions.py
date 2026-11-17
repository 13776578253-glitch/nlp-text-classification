from pathlib import Path

import pytest

from nlp_text_classification.data.goemotions import load_split, validate_dataset


def fixture_dataset(tmp_path: Path) -> Path:
    dest = tmp_path / "goemotions"
    dest.mkdir()
    labels = [f"label_{i}" for i in range(27)] + ["neutral"]
    (dest / "emotions.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    (dest / "train.tsv").write_text('happy now\t0,17\ttrain1\n', encoding="utf-8")
    (dest / "dev.tsv").write_text('sad now\t25\tdev1\n', encoding="utf-8")
    (dest / "test.tsv").write_text('neutral\t27\ttest1\n', encoding="utf-8")
    return dest


def test_load_official_tsv_shape(tmp_path):
    dest = fixture_dataset(tmp_path)
    rows = load_split(dest, "train")
    assert rows[0].labels == (0, 17)
    assert validate_dataset(dest, strict=False)["counts"] == {"train": 1, "validation": 1, "test": 1}


def test_sample_overlap_is_rejected(tmp_path):
    dest = fixture_dataset(tmp_path)
    (dest / "dev.tsv").write_text('sad now\t25\ttrain1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset(dest, strict=False)


def test_out_of_range_label_is_rejected(tmp_path):
    dest = fixture_dataset(tmp_path)
    (dest / "train.tsv").write_text('bad label\t28\ttrain1\n', encoding="utf-8")
    with pytest.raises(ValueError, match="out of range"):
        load_split(dest, "train")

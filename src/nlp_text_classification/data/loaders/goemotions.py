"""Official GoEmotions agreement-filtered TSV loader; no dataset re-splitting."""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file, summarize_splits
from nlp_text_classification.data.contracts import Example

SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
EXPECTED_ROWS = {"train": 43410, "validation": 5426, "test": 5427}
EXPECTED_LABELS = 28
DEFAULT_ROOT = Path("data/raw/goemotions")


def load_labels(data_dir: str | Path) -> list[str]:
    labels = (Path(data_dir) / "emotions.txt").read_text(encoding="utf-8").splitlines()
    if len(labels) != EXPECTED_LABELS or len(set(labels)) != len(labels):
        raise ValueError(f"Expected {EXPECTED_LABELS} unique labels, got {len(labels)}")
    if labels[-1] != "neutral":
        raise ValueError("Official GoEmotions label order must end with neutral")
    return labels


def load_split(data_dir: str | Path, split: str) -> list[Example]:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split: {split}")
    path = Path(data_dir) / SPLIT_FILES[split]
    examples: list[Example] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for line_no, row in enumerate(reader, 1):
            if len(row) != 3:
                raise ValueError(f"{path}:{line_no}: expected three TSV fields, got {len(row)}")
            text, label_ids, sample_id = row
            if not text.strip() or not sample_id.strip() or not label_ids.strip():
                raise ValueError(f"{path}:{line_no}: empty required field")
            labels = tuple(sorted({int(x) for x in label_ids.split(",")}))
            if any(label < 0 or label >= EXPECTED_LABELS for label in labels):
                raise ValueError(f"{path}:{line_no}: label out of range")
            if sample_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicated sample id {sample_id}")
            seen.add(sample_id)
            examples.append(Example(sample_id, text, labels, split))
    return examples


def _check_splits(splits: dict[str, list[Example]], *, strict: bool) -> None:
    for name, examples in splits.items():
        if strict and len(examples) != EXPECTED_ROWS[name]:
            raise ValueError(f"{name}: {len(examples)} rows, expected {EXPECTED_ROWS[name]}")
    sets = {name: {row.sample_id for row in rows} for name, rows in splits.items()}
    for a, b in combinations(sets, 2):
        collisions = sets[a] & sets[b]
        if collisions:
            raise ValueError(f"Sample ID leakage: {a} vs {b}: {len(collisions)} collisions")


def load_training_splits(data_dir: str | Path, *, strict: bool) -> dict[str, list[Example]]:
    splits = {name: load_split(data_dir, name) for name in ("train", "validation")}
    _check_splits(splits, strict=strict)
    return splits


def validate_dataset(data_dir: str | Path, *, strict: bool = True) -> dict:
    root = Path(data_dir)
    label_names = load_labels(root)
    splits = {name: load_split(root, name) for name in SPLIT_FILES}
    _check_splits(splits, strict=strict)
    files = [*SPLIT_FILES.values(), "emotions.txt"]
    return {
        "dataset": "goemotions-simplified-official",
        "labels": label_names,
        "num_labels": len(label_names),
        "split_source": "official Google Research train/dev/test TSV; unchanged",
        "sha256": {filename: sha256_file(root / filename) for filename in files},
        **summarize_splits(splits, range(EXPECTED_LABELS)),
    }

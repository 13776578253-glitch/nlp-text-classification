"""Official GoEmotions agreement-filtered TSV loader; no dataset re-splitting."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

SPLIT_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}
EXPECTED_ROWS = {"train": 43410, "validation": 5426, "test": 5427}
EXPECTED_LABELS = 28


@dataclass(frozen=True)
class Example:
    sample_id: str
    text: str
    labels: tuple[int, ...]
    split: str


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


def validate_dataset(data_dir: str | Path, *, strict: bool = True) -> dict:
    label_names = load_labels(data_dir)
    splits = {name: load_split(data_dir, name) for name in SPLIT_FILES}
    for name, examples in splits.items():
        if strict and len(examples) != EXPECTED_ROWS[name]:
            raise ValueError(f"{name}: {len(examples)} rows, expected {EXPECTED_ROWS[name]}")
    sets = {name: {row.sample_id for row in rows} for name, rows in splits.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        collisions = sets[a] & sets[b]
        if collisions:
            raise ValueError(f"Sample ID leakage: {a} vs {b}: {len(collisions)} collisions")
    return {
        "dataset": "goemotions-simplified-official",
        "labels": label_names,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "num_labels": len(label_names),
        "split_source": "official Google Research train/dev/test TSV; unchanged",
    }

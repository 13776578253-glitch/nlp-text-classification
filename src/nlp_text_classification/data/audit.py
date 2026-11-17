"""Aggregate file and split fingerprints for the data audit."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

from nlp_text_classification.data.contracts import Example


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_splits(splits: dict[str, list[Example]], label_ids: Iterable[int]) -> dict:
    label_ids = tuple(label_ids)
    text_labels: dict[str, dict[str, set[tuple[int, ...]]]] = {}
    label_counts = {}
    within_split = {}
    for split, rows in splits.items():
        counts = Counter(label for row in rows for label in row.labels)
        label_counts[split] = {str(label): counts[label] for label in label_ids}
        assignments = defaultdict(set)
        for row in rows:
            text_key = " ".join(row.text.split())
            assignments[text_key].add(row.labels)
        text_labels[split] = assignments
        within_split[split] = {
            "extra_duplicate_rows": len(rows) - len(assignments),
            "texts_with_conflicting_label_sets": sum(
                len(label_sets) > 1 for label_sets in assignments.values()
            ),
        }

    cross_split = {}
    for left, right in combinations(splits, 2):
        shared = text_labels[left].keys() & text_labels[right].keys()
        cross_split[f"{left}:{right}"] = {
            "shared_texts": len(shared),
            "texts_with_conflicting_label_sets": sum(
                len(text_labels[left][key] | text_labels[right][key]) > 1 for key in shared
            ),
        }

    return {
        "counts": {split: len(rows) for split, rows in splits.items()},
        "label_counts": label_counts,
        "text_comparison": "exact match after whitespace normalization",
        "within_split": within_split,
        "cross_split": cross_split,
    }

"""Build versioned, disjoint experiment splits from the audited source files."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file, summarize_splits
from nlp_text_classification.data.contracts import Example
from nlp_text_classification.data.loaders import goemotions, movie_reviews_zh
from nlp_text_classification.training.run import write_json


def _text_key(row: Example) -> str:
    return " ".join(row.text.split())


def isolate_splits(splits: dict[str, list[Example]]) -> tuple[dict, list]:
    """Keep the complete test split and prefer evaluation rows over fitting rows."""
    test_keys = {_text_key(row) for row in splits["test"]}
    validation_keys = {_text_key(row) for row in splits["validation"]}
    result = {"train": [], "validation": [], "test": splits["test"]}
    excluded = []
    seen = set()
    for split in ("train", "validation"):
        for row in splits[split]:
            key = _text_key(row)
            pair = (key, row.labels)
            reason = None
            if key in test_keys:
                reason = "shared_with_test"
            elif split == "train" and key in validation_keys:
                reason = "shared_with_validation"
            elif split == "train" and pair in seen:
                reason = "train_identical_text_and_labels"
            if reason:
                excluded.append({"sample_id": row.sample_id, "split": split, "reason": reason})
            else:
                result[split].append(row)
                if split == "train":
                    seen.add(pair)
    return result, excluded


def freeze_dataset(
    dataset: str, raw_dir: Path, output: Path, revision: str, *, positive_id: int | None = None
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", revision):
        raise ValueError("Invalid dataset revision")
    if dataset == "goemotions":
        audit = goemotions.validate_dataset(raw_dir)
        splits = {key: goemotions.load_split(raw_dir, key) for key in goemotions.SPLIT_FILES}
        label_names = audit["labels"]
        task = "multilabel"
        label_ids = list(range(len(label_names)))
    elif dataset == "movie_reviews_zh":
        if positive_id not in (0, 1):
            raise ValueError("Stage 0 requires the audited raw positive ID for Chinese reviews")
        audit = movie_reviews_zh.audit_dataset(raw_dir)
        splits = {key: movie_reviews_zh.load_split(raw_dir, key)[0]
                  for key in movie_reviews_zh.SPLIT_FILES}
        label_names = ["negative", "positive"]
        task = "binary"
        label_ids = [1 - positive_id, positive_id]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    derived, excluded = isolate_splits(splits)
    report = summarize_splits(derived, label_ids)
    if any(item["shared_texts"] for item in report["cross_split"].values()):
        raise ValueError("Derived partitions still contain cross-split duplicate texts")
    if any(not rows for rows in derived.values()):
        raise ValueError("Derived partitions cannot be empty")
    for filename, digest in audit["sha256"].items():
        if sha256_file(raw_dir / filename) != digest:
            raise ValueError(f"Source changed while freezing: {filename}")

    output.mkdir(parents=True, exist_ok=True)
    target = output / revision
    if target.exists():
        raise FileExistsError(f"Dataset revision already exists: {target}")
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=output) as temp:
        stage = Path(temp) / revision
        stage.mkdir()
        records = {}
        for split, rows in derived.items():
            name = f"{split}.jsonl"
            with (stage / name).open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            records[split] = {"file": name, "sha256": sha256_file(stage / name), "count": len(rows)}
        write_json(stage / "excluded.json", {"rows": excluded})
        write_json(stage / "source_audit.json", audit)
        train_view = {
            "dataset": dataset,
            "revision": revision,
            "task_type": task,
            "label_names": label_names,
            "label_ids": label_ids,
            "splits": {name: records[name] for name in ("train", "validation")},
        }
        if task == "binary":
            vector_name = movie_reviews_zh.VECTOR_FILE
            train_view["word2vec"] = {
                "file": str((raw_dir / vector_name).resolve()),
                "sha256": audit["sha256"][vector_name],
            }
        write_json(stage / "training.json", train_view)
        manifest = {
            **train_view,
            "splits": records,
            "training_sha256": sha256_file(stage / "training.json"),
            "source_audit_sha256": sha256_file(stage / "source_audit.json"),
            "exclusions_sha256": sha256_file(stage / "excluded.json"),
            "derived_summary": report,
        }
        write_json(stage / "dataset_manifest.json", manifest)
        stage.rename(target)
    return {"path": str(target), "counts": report["counts"], "excluded": len(excluded),
            "manifest_sha256": sha256_file(target / "dataset_manifest.json")}

"""Load the Batch A matrix and the hash-checked train/validation view."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.data.contracts import Example


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_file(path: Path, digest: str) -> Path:
    if sha256_file(path) != digest:
        raise ValueError(f"Fingerprint mismatch: {path}")
    return path


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping: {path}")
    return data


def load_suite(path: Path) -> tuple[dict, dict]:
    config = load_yaml(path)
    experiments = [item["id"] for item in config["experiments"]]
    datasets = {key: load_yaml(Path(ref)) for key, ref in config["dataset_configs"].items()}
    expected = {"E1", "E2", "E3-RNN", "E3-LSTM", "E3-BiLSTM", "E5-TUNE", "E5-FROZEN"}
    if set(experiments) != expected or len(experiments) != len(expected):
        raise ValueError("Batch A must contain each of its seven experiment IDs once")
    if set(datasets) != {"goemotions", "movie_reviews_zh"}:
        raise ValueError("Batch A requires both frozen datasets")
    if config["report_aliases"] != {"E5-RANDOM": "E2"}:
        raise ValueError("E5-RANDOM must reference E2")
    if config["execution"]["gpu_concurrency"] != 1:
        raise ValueError("The Batch A GPU queue has exactly one worker")
    for key, item in datasets.items():
        if item["dataset_id"] != key or item["selection_metric"] != config["selection"]["metric"]:
            raise ValueError(f"Dataset settings disagree with the suite: {key}")
    return config, datasets


def load_partition(root: Path, record: dict, split: str) -> list[Example]:
    if record["file"] != f"{split}.jsonl":
        raise ValueError(f"Invalid partition path: {record['file']}")
    path = checked_file(root / record["file"], record["sha256"])
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["split"] != split:
                raise ValueError(f"Unexpected split in {path}")
            rows.append(Example(item["sample_id"], item["text"], tuple(item["labels"]), split))
    if len(rows) != record["count"] or len({row.sample_id for row in rows}) != len(rows):
        raise ValueError(f"Incorrect count or repeated ID: {path}")
    return rows


@dataclass
class TrainingData:
    settings: dict
    manifest: dict
    train: list[Example]
    validation: list[Example]

    @property
    def binary(self) -> bool:
        return self.manifest["task_type"] == "binary"

    @property
    def outputs(self) -> int:
        return 1 if self.binary else len(self.manifest["label_ids"])

    def encode(self, rows: list[Example]) -> np.ndarray:
        targets = np.zeros((len(rows), self.outputs), dtype=np.int8)
        if self.binary:
            positive = self.settings["positive_raw_id"]
            for index, row in enumerate(rows):
                if len(row.labels) != 1 or row.labels[0] not in self.manifest["label_ids"]:
                    raise ValueError(f"Invalid binary target for {row.sample_id}")
                targets[index, 0] = row.labels[0] == positive
        else:
            for index, row in enumerate(rows):
                if any(label not in self.manifest["label_ids"] for label in row.labels):
                    raise ValueError(f"Invalid multilabel target for {row.sample_id}")
                targets[index, list(row.labels)] = 1
        return targets


def load_training_data(settings: dict) -> TrainingData:
    root = Path(settings["frozen_root"])
    manifest = read_json(checked_file(root / "training.json", settings["training_manifest_sha256"]))
    if set(manifest["splits"]) != {"train", "validation"}:
        raise ValueError("The training manifest may only contain train and validation")
    if settings["dataset_id"] != manifest["dataset"] or settings["task_type"] != manifest["task_type"]:
        raise ValueError("Training manifest identity does not match dataset configuration")
    if settings["num_labels"] != len(manifest["label_ids"]):
        raise ValueError("Training manifest label count does not match dataset configuration")
    if manifest["task_type"] == "binary" and manifest["label_ids"][-1] != settings["positive_raw_id"]:
        raise ValueError("Binary positive class does not match the frozen label mapping")
    train = load_partition(root, manifest["splits"]["train"], "train")
    validation = load_partition(root, manifest["splits"]["validation"], "validation")
    if {row.sample_id for row in train} & {row.sample_id for row in validation}:
        raise ValueError("Sample ID overlaps train and validation")
    return TrainingData(settings, manifest, train, validation)

"""Pin and verify the official Batch B encoder snapshots."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from nlp_text_classification.data.audit import sha256_file
from nlp_text_classification.experiments.suite import checked_file, load_yaml, read_json
from nlp_text_classification.training.progress import write_atomic_json

SUITE = Path("configs/experiment_suites/batch_b_s1.yaml")
OUTPUT = Path("artifacts/batch-b-s1")
CACHE = OUTPUT / "hf-cache"
RESOURCE_LOCK = Path("configs/batch_b/resource_lock.json")
TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    "added_tokens.json",
}
HEAD_PREFIXES = ("classifier.", "pre_classifier.", "head.", "bert.pooler.")


def load_config():
    suite = load_yaml(SUITE)
    datasets = {key: load_yaml(Path(path)) for key, path in suite["dataset_configs"].items()}
    models = {}
    for path in suite["model_configs"]:
        model = load_yaml(Path(path))
        models[model["experiment_id"]] = model
    if tuple(models) != ("B1", "B2", "B3", "B4"):
        raise ValueError("Batch B S1 requires the four ordered encoder configurations")
    return suite, datasets, models


def environment():
    return {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "accelerate",
    )}


def _hub_metadata(model):
    from huggingface_hub import HfApi
    from requests.exceptions import ConnectionError, Timeout

    repo = model["model_id"]
    for attempt in range(4):
        try:
            info = HfApi().model_info(repo, files_metadata=True, token=False, timeout=25)
            break
        except (ConnectionError, Timeout) as exc:
            if attempt == 3:
                raise
            print(f"{repo} metadata {type(exc).__name__}, retry {attempt + 1}/3", flush=True)
            time.sleep(2 ** attempt)
    if info.id != repo or not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise ValueError(f"Unexpected identity or revision for {repo}")
    license_id = info.card_data.to_dict().get("license")
    if license_id != "apache-2.0":
        raise ValueError(f"Verify the changed license for {repo}: {license_id}")
    weight = "model.safetensors" if model["weight_format"] == "safetensors" else "pytorch_model.bin"
    allowed = TOKENIZER_FILES | {"README.md", "LICENSE", "config.json", weight}
    files = {item.rfilename: {
        "bytes": item.size,
        **({"lfs_sha256": item.lfs.sha256} if item.lfs else {"git_blob": item.blob_id}),
    } for item in info.siblings if item.rfilename in allowed}
    if not {weight, "tokenizer.json", "tokenizer_config.json", "config.json", "README.md"} <= files.keys():
        raise ValueError(f"Incomplete upstream snapshot for {repo}")
    if not files[weight].get("lfs_sha256"):
        raise ValueError(f"Expected a verified LFS weight object for {repo}")
    return {"model_id": repo, "revision": info.sha, "license": license_id,
            "weight_format": model["weight_format"], "weight_file": weight, "files": files}


def _verify_download(directory, metadata):
    entries = {}
    for name, identity in metadata["files"].items():
        path = directory / name
        if path.stat().st_size != identity["bytes"]:
            raise ValueError(f"File size differs from Hub metadata: {path}")
        digest = sha256_file(path)
        if "lfs_sha256" in identity:
            if digest != identity["lfs_sha256"]:
                raise ValueError(f"Hub LFS identity mismatch: {path}")
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            if blob != identity["git_blob"]:
                raise ValueError(f"Git blob identity mismatch: {path}")
        entries[name] = {**identity, "sha256": digest}
    return entries


def load_tokenizer(snapshot):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError(f"Expected official fast tokenizer with padding: {snapshot}")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def load_encoder(snapshot, model, labels):
    from transformers import AutoConfig, AutoModelForSequenceClassification

    options = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(snapshot, **options)
    if config.model_type != model["model_type"]:
        raise ValueError(f"Unexpected model type at {snapshot}: {config.model_type}")
    config.num_labels = len(labels)
    config.id2label = dict(enumerate(labels))
    config.label2id = {label: index for index, label in config.id2label.items()}
    config.problem_type = (
        "single_label_classification" if len(labels) == 2 else "multi_label_classification"
    )
    network, loading = AutoModelForSequenceClassification.from_pretrained(
        snapshot, config=config, use_safetensors=model["weight_format"] == "safetensors",
        weights_only=True, output_loading_info=True, attn_implementation="sdpa", **options,
    )
    backbone_missing = [key for key in loading["missing_keys"] if not key.startswith(HEAD_PREFIXES)]
    if backbone_missing or loading["mismatched_keys"] or loading.get("error_msgs"):
        raise ValueError(f"Pretrained backbone did not load cleanly: {loading}")
    if not all(parameter.requires_grad for parameter in network.parameters()):
        raise ValueError("All encoder and head parameters must be trainable")
    return network, loading


def label_names(data):
    if data.binary:
        return ["negative", "positive"]
    return list(data.manifest["label_names"])


def read_resources(models, *, verify_weights=True):
    lock = read_json(RESOURCE_LOCK)
    if lock["environment"] != environment() or set(lock["models"]) != set(models):
        raise ValueError("Batch B resource lock or dependency version changed")
    for key, model in models.items():
        entry = lock["models"][key]
        if (entry["model_id"], entry["weight_format"]) != (model["model_id"], model["weight_format"]):
            raise ValueError(f"Unexpected resource for {key}")
        directory = CACHE / key / entry["revision"]
        for name, record in entry["files"].items():
            if verify_weights or name != entry["weight_file"]:
                checked_file(directory / name, record["sha256"])
    return lock["models"]


def prepare_resources():
    import torch
    from huggingface_hub import snapshot_download

    from nlp_text_classification.experiments.suite import load_training_data

    _, datasets, models = load_config()
    labels = {key: label_names(load_training_data(item)) for key, item in datasets.items()}
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pin_path = OUTPUT / "hub_revisions.json"
    pinned = read_json(pin_path) if pin_path.exists() else {}
    completed = {}
    for key, model in models.items():
        try:
            if key not in pinned:
                pinned[key] = _hub_metadata(model)
                write_atomic_json(pin_path, pinned)
            metadata = pinned[key]
            if metadata["model_id"] != model["model_id"]:
                raise ValueError("Cached Hub metadata belongs to another model")
            snapshot = CACHE / key / metadata["revision"]
            print(f"{key} {metadata['model_id']} @ {metadata['revision']} downloading", flush=True)
            if any(not (snapshot / name).is_file() for name in metadata["files"]):
                snapshot_download(
                    metadata["model_id"], revision=metadata["revision"], local_dir=snapshot,
                    allow_patterns=list(metadata["files"]), token=False, max_workers=2,
                )
            files = _verify_download(snapshot, metadata)
            tokenizer = load_tokenizer(snapshot)
            network, loading = load_encoder(snapshot, model, labels[model["dataset"]])
            texts = ["a short review", "an offline classification loading test"]
            if model["dataset"] == "movie_reviews_zh":
                texts = ["这 是 一部 电影", "这个 故事 很 有趣"]
            with torch.inference_mode():
                logits = network(**tokenizer(texts, return_tensors="pt", padding=True)).logits
            shape = [2, len(labels[model["dataset"]])]
            if list(logits.shape) != shape or not torch.isfinite(logits).all():
                raise ValueError(f"Offline classification head failed for {key}")
            completed[key] = {**metadata, "files": files, "cache": snapshot.as_posix(),
                              "parameters": sum(p.numel() for p in network.parameters()),
                              "offline_load": {"status": "PASS", "shape": shape,
                                               "model_class": type(network).__name__,
                                               "tokenizer_class": type(tokenizer).__name__,
                                               "loading": loading}}
            print(f"{key} PASS {completed[key]['parameters']:,} parameters, shape {shape}", flush=True)
            del network, tokenizer
            gc.collect()
        except Exception as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            write_atomic_json(OUTPUT / f"{key}-resource-failure-{stamp}.json", {
                "model": model, "upstream": pinned.get(key), "exception": repr(exc),
            })
            raise
    record = {"environment": environment(), "models": completed}
    if RESOURCE_LOCK.exists() and read_json(RESOURCE_LOCK) != record:
        raise ValueError("Model lock already exists with different metadata")
    write_atomic_json(RESOURCE_LOCK, record)
    return record


if __name__ == "__main__":
    prepare_resources()

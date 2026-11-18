"""Hash-audited train/validation inputs for pretrained tokenizers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class EncodedPartition:
    features: dict[str, np.ndarray]
    lengths: np.ndarray
    targets: np.ndarray


def model_text(text, chinese):
    return " ".join(text.split()) if chinese else text


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def tokenizer_fingerprint(resource):
    return digest_json({name: item["sha256"] for name, item in resource["files"].items()
                        if name not in {"README.md", "LICENSE", resource["weight_file"]}})


def _lengths_and_reconstruction(tokenizer, rows, chinese):
    lengths, differences, mixed, mixed_differences = [], 0, 0, 0
    for start in range(0, len(rows), 512):
        texts = [model_text(row.text, chinese) for row in rows[start:start + 512]]
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)
        lengths.extend(map(len, encoded["input_ids"]))
        if chinese:
            joined = ["".join(text.split()) for text in texts]
            alternate = tokenizer(joined, add_special_tokens=True, truncation=False, verbose=False)
            for text, left, right in zip(texts, encoded["input_ids"], alternate["input_ids"],
                                         strict=True):
                code_switch = any(char.isascii() and char.isalnum() for char in text)
                differences += left != right
                mixed += code_switch
                mixed_differences += code_switch and left != right
    return np.asarray(lengths), {"different_ids": differences, "mixed_ascii_rows": mixed,
                                 "different_ids_on_mixed_ascii": mixed_differences}


def choose_length(lengths, candidates, quantile):
    return next((cap for cap in candidates if (lengths <= cap).mean() >= quantile), candidates[-1])


def _length_stats(lengths, cap):
    return {"rows": len(lengths), "max": int(lengths.max()),
            "percentiles": {str(q): float(np.percentile(lengths, q)) for q in (50, 90, 95, 99)},
            "truncated_rows": int(np.sum(lengths > cap)),
            "truncated_fraction": float(np.mean(lengths > cap))}


def encode_partition(tokenizer, data, rows, cap, multiple=8):
    if cap % multiple:
        raise ValueError("Token limit must align with the padding multiple")
    features = {}
    lengths = np.empty(len(rows), dtype=np.int32)
    for start in range(0, len(rows), 512):
        texts = [model_text(row.text, data.binary) for row in rows[start:start + 512]]
        result = tokenizer(
            texts, add_special_tokens=True, max_length=cap, padding="max_length",
            truncation=True, return_attention_mask=True, return_special_tokens_mask=True,
            return_tensors="np",
        )
        slots = slice(start, start + len(texts))
        mask = result["attention_mask"]
        count = mask.sum(axis=1)
        if np.any(count < tokenizer.num_special_tokens_to_add(pair=False)):
            raise ValueError("Tokenizer discarded required special tokens")
        expected = np.arange(cap)[None, :] < count[:, None]
        if not np.array_equal(mask, expected):
            raise ValueError("Attention mask does not match right padding")
        if not np.all(result["input_ids"][~expected] == tokenizer.pad_token_id):
            raise ValueError("Masked positions do not contain the tokenizer PAD token")
        specials = result["special_tokens_mask"][expected]
        if specials.sum() != len(texts) * tokenizer.num_special_tokens_to_add(pair=False):
            raise ValueError("Special token count does not match the tokenizer template")
        row_indices = np.arange(len(texts))
        if not np.all(result["input_ids"][row_indices, 0] == tokenizer.cls_token_id) or not (
            np.all(result["input_ids"][row_indices, count - 1] == tokenizer.sep_token_id)
        ):
            raise ValueError("Packed rows do not match the encoder CLS/SEP rule")
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            if name in result:
                if name not in features:
                    features[name] = np.empty((len(rows), cap), dtype=np.int64)
                features[name][slots] = result[name]
        lengths[slots] = count
    return EncodedPartition(features, lengths, data.encode(rows))


def input_fingerprint(encoded):
    digest = hashlib.sha256()
    for name, values in sorted(encoded.features.items()):
        digest.update(name.encode())
        digest.update(values.tobytes())
    digest.update(encoded.targets.tobytes())
    return digest.hexdigest()


def audit_tokenization(tokenizer, data, policy, resource):
    dataset = data.manifest["dataset"]
    train, train_reconstruction = _lengths_and_reconstruction(tokenizer, data.train, data.binary)
    validation, val_reconstruction = _lengths_and_reconstruction(
        tokenizer, data.validation, data.binary,
    )
    cap = choose_length(train, policy["candidates"][dataset], policy["coverage_quantile"])
    multiple = policy["padding_multiple"]
    raw = {"train": train, "validation": validation}
    partitions = {
        key: encode_partition(tokenizer, data, rows, cap, multiple)
        for key, rows in (("train", data.train), ("validation", data.validation))
    }
    for key, part in partitions.items():
        if not np.array_equal(part.lengths, np.minimum(raw[key], cap)):
            raise ValueError(f"Audited lengths disagree with encoded {key} lengths")
    result = {
        "dataset": dataset, "dataset_manifest_sha256": data.settings["dataset_manifest_sha256"],
        "training_manifest_sha256": data.settings["training_manifest_sha256"],
        "tokenizer_sha256": tokenizer_fingerprint(resource), "model_revision": resource["revision"],
        "text_rule": policy["chinese_text"] if data.binary else "frozen_text",
        "selected_length": cap, "candidate_lengths": policy["candidates"][dataset],
        "train_coverage_target": policy["coverage_quantile"],
        "train_coverage": float((train <= cap).mean()),
        "padding_multiple": multiple, "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side, "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id, "sep_token_id": tokenizer.sep_token_id,
        "special_token_count": tokenizer.num_special_tokens_to_add(pair=False),
        "attention_mask_verified": True, "special_tokens_verified": True,
        "partitions": {key: {**_length_stats(raw[key], cap),
                              "encoded_sha256": input_fingerprint(part)}
                       for key, part in partitions.items()},
    }
    if data.binary:
        result["join_without_spaces_comparison"] = {
            "train": train_reconstruction, "validation": val_reconstruction,
        }
    return result

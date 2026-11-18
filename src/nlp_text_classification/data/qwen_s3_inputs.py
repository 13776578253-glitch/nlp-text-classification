"""GoEmotions tokenization for the Qwen text classifier."""

from __future__ import annotations

import numpy as np

from nlp_text_classification.data.transformer_inputs import (
    EncodedPartition,
    choose_length,
    input_fingerprint,
)


def encode(tokenizer, data, rows, cap, policy):
    pad = policy["pad_token_id"]
    ids = np.full((len(rows), cap), pad, dtype=np.int64)
    lengths = np.empty(len(rows), dtype=np.int32)
    for start in range(0, len(rows), 512):
        batch = rows[start:start + 512]
        tokens = tokenizer(
            [row.text for row in batch], add_special_tokens=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False, verbose=False,
        )["input_ids"]
        for i, (row, item) in enumerate(zip(batch, tokens, strict=True), start):
            if not item or pad in item:
                raise ValueError(f"Blank text or embedded PAD/EOS at {row.sample_id}")
            lengths[i] = len(item)
            ids[i, :min(cap, len(item))] = item[:cap]
    mask = ids != pad
    valid = np.minimum(lengths, cap)
    expected = np.arange(cap)[None, :] < valid[:, None]
    if not np.array_equal(mask, expected):
        raise ValueError("Right padding and non-PAD pooling disagree")
    return EncodedPartition({"input_ids": ids, "attention_mask": mask.astype(np.int64)},
                            valid, data.encode(rows)), lengths


def audit(tokenizer, data, policy):
    if (tokenizer.pad_token_id, tokenizer.eos_token_id) != (
        policy["pad_token_id"], policy["eos_token_id"]
    ) or not tokenizer.is_fast:
        raise ValueError("Unexpected Qwen tokenizer implementation or PAD/EOS")
    candidates = policy["length_candidates"]
    split_rows = {"train": data.train, "validation": data.validation}
    bundles = {split: encode(tokenizer, data, rows, max(candidates), policy)
               for split, rows in split_rows.items()}
    cap = choose_length(bundles["train"][1], candidates, policy["train_coverage"])
    evidence, partitions = {}, {}
    for name, (part, lengths) in bundles.items():
        partition = EncodedPartition({k: np.ascontiguousarray(v[:, :cap])
                                      for k, v in part.features.items()},
                                     np.minimum(part.lengths, cap), part.targets)
        partitions[name] = partition
        evidence[name] = {
            "rows": len(lengths), "min": int(lengths.min()), "max": int(lengths.max()),
            "percentiles": dict(zip(("p50", "p90", "p95", "p99"), np.percentile(
                lengths, (50, 90, 95, 99)
            ).tolist(), strict=True)),
            "coverage": {str(x): float((lengths <= x).mean()) for x in candidates},
            "truncated_rows": int((lengths > cap).sum()),
            "encoded_sha256": input_fingerprint(partition),
        }
    return {"selected_length": cap, "policy": policy, "partitions": evidence,
            "label_order": data.manifest["label_names"],
            "training_manifest_sha256": data.settings["training_manifest_sha256"]}, partitions

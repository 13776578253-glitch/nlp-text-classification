"""Train-only vocabulary and comparable Word2Vec initialization."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from nlp_text_classification.data.audit import sha256_file

_WORDS = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def tokens(text: str, tokenizer: str) -> list[str]:
    if tokenizer == "whitespace":
        return text.split()
    if tokenizer == "english_word":
        return _WORDS.findall(text.lower())
    raise ValueError(f"Unknown tokenizer: {tokenizer}")


def vocabulary(rows, tokenizer: str) -> tuple[dict[str, int], Counter]:
    counts = Counter(word for row in rows for word in tokens(row.text, tokenizer))
    words = sorted(counts, key=lambda word: (-counts[word], word))
    return {word: index + 2 for index, word in enumerate(words)}, counts


def encode(rows, tokenizer: str, vocab: dict, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.zeros((len(rows), max_length), dtype=np.int64)
    lengths = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        seq = [vocab.get(word, 1) for word in tokens(row.text, tokenizer)[:max_length]] or [1]
        lengths[index] = len(seq)
        inputs[index, :len(seq)] = seq
    return inputs, lengths


def prepare_word2vec(data, output: Path, seed: int, dimension: int):
    from gensim.models import KeyedVectors, Word2Vec

    if data.binary and "word2vec" in data.manifest:
        spec = data.manifest["word2vec"]
        path = Path(data.settings["raw_root"]) / Path(spec["file"]).name
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ValueError(f"Frozen Word2Vec file changed: {path}")
        vectors = KeyedVectors.load_word2vec_format(str(path), binary=True, unicode_errors="strict")
        source = "teacher_wiki_word2vec_50"
    else:
        output.mkdir(parents=True, exist_ok=True)
        identity = data.settings["training_manifest_sha256"]
        path = output / f"{data.manifest['dataset']}-{identity[:16]}-{dimension}d-{seed}.kv"
        if not path.exists():
            corpus = [tokens(row.text, data.settings["tokenizer"]) for row in data.train]
            model = Word2Vec(corpus, vector_size=dimension, min_count=2, workers=1,
                             seed=seed, epochs=5)
            temp = path.with_suffix(".partial")
            model.wv.save(str(temp), separately=[])
            temp.replace(path)
        vectors = KeyedVectors.load(str(path))
        digest = sha256_file(path)
        source = "train_only_unsupervised"
    if vectors.vector_size != dimension:
        raise ValueError(f"Expected {dimension} dimensions in {path}")
    if source == "teacher_wiki_word2vec_50" and len(vectors) != 426677:
        raise ValueError(f"Unexpected number of vectors in {path}: {len(vectors)}")
    return vectors, {"source": source, "sha256": digest,
                     "dimension": dimension, "vectors": len(vectors)}

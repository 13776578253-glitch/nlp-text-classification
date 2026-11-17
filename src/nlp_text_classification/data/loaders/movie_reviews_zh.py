"""Read the course-provided, pre-tokenized Chinese movie reviews."""

from __future__ import annotations

from pathlib import Path

from nlp_text_classification.data.audit import sha256_file, summarize_splits
from nlp_text_classification.data.contracts import Example

DEFAULT_ROOT = Path("data/raw/movie_reviews_zh")
SPLIT_FILES = {"train": "train.txt", "validation": "validation.txt", "test": "test.txt"}
VECTOR_FILE = "wiki_word2vec_50.bin"


def load_split(data_dir: str | Path, split: str) -> tuple[list[Example], int]:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split: {split}")
    path = Path(data_dir) / SPLIT_FILES[split]
    rows = []
    blank_rows = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                blank_rows += 1
                continue
            raw_label, separator, text = line.rstrip("\r\n").partition("\t")
            if raw_label not in ("0", "1") or not separator or not text.strip() or "\t" in text:
                raise ValueError(f"{path}:{line_no}: expected 0/1 and pre-tokenized text")
            rows.append(Example(f"{split}:{line_no}", text, (int(raw_label),), split))
    return rows, blank_rows


def audit_dataset(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    splits = {}
    blank_rows = {}
    for split in SPLIT_FILES:
        splits[split], blank_rows[split] = load_split(root, split)

    vector_path = root / VECTOR_FILE
    with vector_path.open("rb") as stream:
        header = stream.readline(128)
    try:
        if not header.endswith(b"\n"):
            raise ValueError("missing header terminator")
        vocab_size, dimension = map(int, header.split())
    except ValueError as exc:
        raise ValueError(f"{vector_path}: invalid Word2Vec header") from exc
    if vocab_size <= 0 or dimension != 50:
        raise ValueError(f"{vector_path}: expected a nonempty vocabulary and 50 dimensions")

    files = [*SPLIT_FILES.values(), VECTOR_FILE]
    return {
        "dataset": "movie_reviews_zh",
        "label_ids": [0, 1],
        "blank_rows": blank_rows,
        "sha256": {filename: sha256_file(root / filename) for filename in files},
        "word2vec": {"file": VECTOR_FILE, "header_vocab_size": vocab_size, "dimension": dimension},
        **summarize_splits(splits, (0, 1)),
    }

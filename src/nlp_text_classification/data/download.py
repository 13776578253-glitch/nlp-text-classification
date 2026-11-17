"""Fetch official Google Research GoEmotions files without adding dataset dependencies."""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

from nlp_text_classification.data.loaders.goemotions import validate_dataset

BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
FILES = ("emotions.txt", "train.tsv", "dev.tsv", "test.tsv")


def download_official(dest: str | Path, *, force: bool = False) -> dict:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = dest / filename
        if force or not target.is_file():
            temp = target.with_name(target.name + ".partial")
            try:
                with (
                    urllib.request.urlopen(f"{BASE_URL}/{filename}", timeout=120) as response,
                    temp.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out)
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
    summary = validate_dataset(dest, strict=True)
    manifest = {**summary, "source_url": BASE_URL}
    (dest / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest

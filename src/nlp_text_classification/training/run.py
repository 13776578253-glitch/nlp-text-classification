"""Reproducible run metadata. Artifacts are never committed to the repository."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def new_run_dir(root: str | Path, experiment: str, seed: int) -> Path:
    run_id = f"{experiment}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-s{seed}"
    directory = Path(root) / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory

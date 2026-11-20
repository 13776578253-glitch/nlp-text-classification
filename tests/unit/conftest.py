"""Fixtures for tests that read local frozen experiment evidence."""

from pathlib import Path

import pytest


@pytest.fixture
def frozen_experiment_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "data/processed").is_dir() and not (root / "artifacts/runs").is_dir():
        pytest.skip("Requires local frozen datasets and experiment artifacts")
    return root

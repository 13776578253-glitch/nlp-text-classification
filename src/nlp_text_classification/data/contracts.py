"""Shared row representation for fixed-split classification datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    sample_id: str
    text: str
    labels: tuple[int, ...]
    split: str

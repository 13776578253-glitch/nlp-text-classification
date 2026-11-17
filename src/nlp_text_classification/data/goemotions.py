"""Compatibility imports for existing GoEmotions callers."""

from nlp_text_classification.data.loaders.goemotions import (
    EXPECTED_LABELS,
    EXPECTED_ROWS,
    SPLIT_FILES,
    Example,
    load_labels,
    load_split,
    validate_dataset,
)

__all__ = [
    "EXPECTED_LABELS",
    "EXPECTED_ROWS",
    "SPLIT_FILES",
    "Example",
    "load_labels",
    "load_split",
    "validate_dataset",
]

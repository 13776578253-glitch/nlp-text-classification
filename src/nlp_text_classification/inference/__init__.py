"""Unified inference adapters and registry for EmotionLab."""

from nlp_text_classification.inference.adapters import PredictionResult, WorkerUnavailableError
from nlp_text_classification.inference.engine import InferenceEngine
from nlp_text_classification.inference.registry import ModelRegistry, ModelSpec

__all__ = [
    "InferenceEngine",
    "ModelRegistry",
    "ModelSpec",
    "PredictionResult",
    "WorkerUnavailableError",
]

"""Thread-safe unified inference engine with LRU caching and WSL2 Qwen proxy."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from nlp_text_classification.inference.adapters import (
    BatchANeuralAdapter,
    BatchASvmAdapter,
    BatchBEncoderAdapter,
    PredictionResult,
    QwenWorkerProxyAdapter,
    WorkerUnavailableError,
)
from nlp_text_classification.inference.registry import ModelRegistry, ModelSpec


class InferenceEngine:
    """Unified inference coordinator across Batch A, Batch B Encoders, and Qwen S4."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        light_cache_size: int = 16,
        encoder_cache_size: int = 4,
        qwen_worker_url: str | None = None,
    ) -> None:
        self.registry = ModelRegistry(root=root)
        self.light_cache_size = light_cache_size
        self.encoder_cache_size = encoder_cache_size
        self._light_cache: OrderedDict[tuple[str, int], BatchASvmAdapter | BatchANeuralAdapter] = (
            OrderedDict()
        )
        self._encoder_cache: OrderedDict[tuple[str, int], BatchBEncoderAdapter] = OrderedDict()
        self._qwen_proxy = QwenWorkerProxyAdapter(worker_url=qwen_worker_url)
        self._lock = threading.RLock()

    def _get_light_adapter(self, spec: ModelSpec) -> BatchASvmAdapter | BatchANeuralAdapter:
        key = (spec.model_id, spec.seed)
        with self._lock:
            if key in self._light_cache:
                self._light_cache.move_to_end(key)
                return self._light_cache[key]
            if spec.family == "batch_a_svm":
                adapter: BatchASvmAdapter | BatchANeuralAdapter = BatchASvmAdapter(spec)
            else:
                adapter = BatchANeuralAdapter(spec)
            self._light_cache[key] = adapter
            while len(self._light_cache) > self.light_cache_size:
                self._light_cache.popitem(last=False)
            return adapter

    def _get_encoder_adapter(self, spec: ModelSpec) -> BatchBEncoderAdapter:
        key = (spec.model_id, spec.seed)
        with self._lock:
            if key in self._encoder_cache:
                self._encoder_cache.move_to_end(key)
                return self._encoder_cache[key]
            while len(self._encoder_cache) >= self.encoder_cache_size:
                _, evicted = self._encoder_cache.popitem(last=False)
                evicted.unload()
            adapter = BatchBEncoderAdapter(spec)
            self._encoder_cache[key] = adapter
            return adapter

    def is_loaded(self, model_id: str, seed: int = 42) -> bool:
        key = (model_id, seed)
        return key in self._light_cache or key in self._encoder_cache

    def list_models(self, dataset: str | None = None) -> list[dict[str, Any]]:
        qwen_status = self._qwen_proxy.probe_health(timeout=0.8)
        items: list[dict[str, Any]] = []
        for spec in self.registry.list_default_specs(dataset=dataset):
            if spec.runtime == "wsl2_worker":
                available = spec.checkpoint_exists() and bool(qwen_status.get("online"))
                loaded = bool(qwen_status.get("online")) and qwen_status.get("loaded_seed") is not None
            else:
                available = spec.checkpoint_exists()
                loaded = self.is_loaded(spec.model_id, spec.seed)
            items.append(spec.to_public_dict(loaded=loaded, available=available))
        return items

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "run_id": spec.run_id,
                "model_id": spec.model_id,
                "model_code": spec.model_code,
                "dataset": spec.dataset,
                "seed": spec.seed,
                "batch": spec.batch,
                "score_kind": spec.score_kind,
                "threshold": spec.threshold,
                "validation_macro_f1": spec.validation_macro_f1,
                "test_macro_f1": spec.test_macro_f1,
                "checkpoint_exists": spec.checkpoint_exists(),
            }
            for spec in self.registry.list_all_runs()
        ]

    def predict_single(
        self,
        text: str,
        model_id: str,
        *,
        seed: int | None = None,
        top_k: int = 8,
    ) -> PredictionResult:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text must not be empty")
        spec = self.registry.resolve(model_id, seed=seed)
        if spec.runtime == "wsl2_worker":
            return self._qwen_proxy.predict(spec, cleaned, top_k=top_k)
        if not spec.checkpoint_exists():
            raise FileNotFoundError(f"Checkpoint not found for {spec.model_id}@{spec.seed}")
        if spec.family in {"batch_a_svm", "batch_a_neural"}:
            adapter = self._get_light_adapter(spec)
            return adapter.predict(cleaned, top_k=top_k)
        if spec.family == "batch_b_encoder":
            with self._lock:
                encoder = self._get_encoder_adapter(spec)
                return encoder.predict(cleaned, top_k=top_k)
        raise ValueError(f"Unsupported model family: {spec.family}")

    def predict(
        self,
        text: str,
        model_ids: list[str],
        *,
        seed: int | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        for mid in model_ids:
            try:
                res = self.predict_single(text, mid, seed=seed, top_k=top_k)
                results.append(res.to_dict())
            except (KeyError, ValueError, FileNotFoundError, WorkerUnavailableError, OSError) as exc:
                errors[mid] = str(exc)
        return {"results": results, "errors": errors}

    def health(self) -> dict[str, Any]:
        import torch

        gpu_info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_info.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "vram_free_mib": round(free_b / (1024**2), 1),
                    "vram_total_mib": round(total_b / (1024**2), 1),
                }
            )
        return {
            "status": "ok",
            "mode": "hybrid",
            "gpu": gpu_info,
            "loaded_models": [
                f"{mid}@{seed}"
                for mid, seed in [*self._light_cache.keys(), *self._encoder_cache.keys()]
            ],
            "qwen_worker": self._qwen_proxy.probe_health(timeout=0.8),
        }

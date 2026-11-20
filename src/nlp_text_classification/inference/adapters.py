"""Model-family inference adapters producing a unified prediction response schema."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import yaml

from nlp_text_classification.data.features import tokens as batch_a_tokens
from nlp_text_classification.data.transformer_inputs import model_text
from nlp_text_classification.inference.chinese_seg import VocabularySegmenter
from nlp_text_classification.inference.registry import ModelSpec
from nlp_text_classification.models.neural import make_model


class WorkerUnavailableError(RuntimeError):
    """Raised when the isolated WSL2 Qwen inference worker is offline or unreachable."""


@dataclass
class PredictionResult:
    model_id: str
    model_code: str
    display_name: str
    family: str
    dataset: str
    seed: int
    run_id: str
    score_kind: str
    threshold: float
    labels: list[str]
    scores: list[float]
    positive_score: float | None
    predicted_labels: list[str]
    top_k: list[dict[str, Any]]
    latency_ms: float
    device: str
    preprocessing: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_top_k(
    labels: list[str],
    scores: list[float],
    threshold: float,
    score_kind: str,
    dataset: str,
    k: int = 8,
) -> list[dict[str, Any]]:
    if dataset == "movie_reviews_zh" and score_kind == "decision":
        margin = scores[1]
        rows = [
            {
                "label": "positive",
                "score": float(margin),
                "threshold": float(threshold),
                "delta": float(margin - threshold),
                "active": bool(margin >= threshold),
            },
            {
                "label": "negative",
                "score": float(-margin),
                "threshold": float(-threshold),
                "delta": float(threshold - margin),
                "active": bool(margin < threshold),
            },
        ]
        rows.sort(key=lambda r: r["delta"], reverse=True)
        return rows

    if dataset == "movie_reviews_zh":
        p_pos = scores[1]
        rows = [
            {
                "label": "positive",
                "score": float(p_pos),
                "threshold": float(threshold),
                "delta": float(p_pos - threshold),
                "active": bool(p_pos >= threshold),
            },
            {
                "label": "negative",
                "score": float(1.0 - p_pos),
                "threshold": float(1.0 - threshold),
                "delta": float(threshold - p_pos),
                "active": bool(p_pos < threshold),
            },
        ]
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows

    ranked = sorted(
        (
            {
                "label": label,
                "score": float(score),
                "threshold": float(threshold),
                "delta": float(score - threshold),
                "active": bool(score >= threshold),
            }
            for label, score in zip(labels, scores, strict=True)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranked[:k]


class BatchASvmAdapter:
    """Adapter for Batch A TF-IDF + LinearSVC (decision_function margin)."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.bundle = joblib.load(spec.checkpoint_path)
        self.segmenter: VocabularySegmenter | None = None
        if spec.dataset == "movie_reviews_zh":
            self.segmenter = VocabularySegmenter(self.bundle["vectorizer"].vocabulary_.keys())

    def predict(self, text: str, top_k: int = 8) -> PredictionResult:
        started = time.perf_counter()
        normalized = text.strip()
        seg_applied = False
        preview_tokens: list[str] = []
        if self.segmenter is not None:
            normalized, preview_tokens, seg_applied = self.segmenter.segment(normalized)
        else:
            preview_tokens = normalized.split()

        features = self.bundle["vectorizer"].transform([normalized])
        raw = np.asarray(self.bundle["classifier"].decision_function(features), dtype=np.float64)
        flat = raw.reshape(-1).tolist()

        if self.spec.dataset == "movie_reviews_zh":
            margin = float(flat[0])
            scores = [-margin, margin]
            positive_score: float | None = margin
            predicted = ["positive"] if margin >= self.spec.threshold else ["negative"]
        else:
            scores = [float(x) for x in flat]
            positive_score = None
            predicted = [
                label
                for label, score in zip(self.spec.labels, scores, strict=True)
                if score >= self.spec.threshold
            ]

        latency_ms = (time.perf_counter() - started) * 1000.0
        return PredictionResult(
            model_id=self.spec.model_id,
            model_code=self.spec.model_code,
            display_name=self.spec.display_name,
            family=self.spec.family,
            dataset=self.spec.dataset,
            seed=self.spec.seed,
            run_id=self.spec.run_id,
            score_kind=self.spec.score_kind,
            threshold=self.spec.threshold,
            labels=list(self.spec.labels),
            scores=scores,
            positive_score=positive_score,
            predicted_labels=predicted,
            top_k=_build_top_k(
                list(self.spec.labels),
                scores,
                self.spec.threshold,
                self.spec.score_kind,
                self.spec.dataset,
                k=top_k,
            ),
            latency_ms=round(latency_ms, 2),
            device="cpu",
            preprocessing={
                "segmentation_applied": seg_applied,
                "token_count": len(preview_tokens),
                "truncated": False,
                "preview_tokens": preview_tokens[:24],
            },
        )


class BatchANeuralAdapter:
    """Adapter for Batch A TextCNN / Vanilla RNN / LSTM / BiLSTM models."""

    def __init__(self, spec: ModelSpec, device: str | None = None) -> None:
        import torch

        self.spec = spec
        self.device_name = device or os.environ.get("EMOTIONLAB_NEURAL_DEVICE", "cpu")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)

        if spec.config_path is None:
            raise ValueError(f"Missing resolved_config.yaml for {spec.model_id}")
        resolved = yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))
        bundle = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
        self.vocab: dict[str, int] = bundle["vocabulary"]
        self.tokenizer_mode = "whitespace" if spec.dataset == "movie_reviews_zh" else "english_word"
        exp = resolved["experiment"]
        settings = resolved["settings"]
        outputs = 1 if spec.dataset == "movie_reviews_zh" else len(spec.labels)
        self.min_len = max(settings["textcnn"]["kernel_sizes"]) if exp["model"] == "textcnn" else 1

        self.model = make_model(exp["model"], settings, len(self.vocab) + 2, outputs)
        self.model.load_state_dict(bundle["model"])
        self.model.to(self.device)
        self.model.eval()

        self.segmenter = (
            VocabularySegmenter(self.vocab.keys())
            if spec.dataset == "movie_reviews_zh"
            else None
        )

    def predict(self, text: str, top_k: int = 8) -> PredictionResult:
        import torch

        started = time.perf_counter()
        normalized = text.strip()
        seg_applied = False
        if self.segmenter is not None:
            normalized, seg_tokens, seg_applied = self.segmenter.segment(normalized)
            toks = seg_tokens
        else:
            toks = batch_a_tokens(normalized, self.tokenizer_mode)

        truncated = len(toks) > self.spec.max_length
        kept = toks[: self.spec.max_length]
        seq = [self.vocab.get(word, 1) for word in kept] or [1]
        seq_len = len(seq)
        width = max(self.min_len, seq_len)

        x = torch.zeros((1, width), dtype=torch.long, device=self.device)
        x[0, :seq_len] = torch.as_tensor(seq, dtype=torch.long, device=self.device)
        lengths = torch.tensor([seq_len], dtype=torch.long)

        with torch.inference_mode():
            probs = self.model(x, lengths).sigmoid().detach().cpu().numpy()[0].tolist()

        if self.spec.dataset == "movie_reviews_zh":
            p_pos = float(probs[0])
            scores = [1.0 - p_pos, p_pos]
            positive_score: float | None = p_pos
            predicted = ["positive"] if p_pos >= self.spec.threshold else ["negative"]
        else:
            scores = [float(p) for p in probs]
            positive_score = None
            predicted = [
                label
                for label, score in zip(self.spec.labels, scores, strict=True)
                if score >= self.spec.threshold
            ]

        latency_ms = (time.perf_counter() - started) * 1000.0
        return PredictionResult(
            model_id=self.spec.model_id,
            model_code=self.spec.model_code,
            display_name=self.spec.display_name,
            family=self.spec.family,
            dataset=self.spec.dataset,
            seed=self.spec.seed,
            run_id=self.spec.run_id,
            score_kind=self.spec.score_kind,
            threshold=self.spec.threshold,
            labels=list(self.spec.labels),
            scores=scores,
            positive_score=positive_score,
            predicted_labels=predicted,
            top_k=_build_top_k(
                list(self.spec.labels),
                scores,
                self.spec.threshold,
                self.spec.score_kind,
                self.spec.dataset,
                k=top_k,
            ),
            latency_ms=round(latency_ms, 2),
            device=self.device_name,
            preprocessing={
                "segmentation_applied": seg_applied,
                "token_count": len(toks),
                "truncated": truncated,
                "preview_tokens": kept[:24],
            },
        )


class BatchBEncoderAdapter:
    """Adapter for Batch B Transformer Encoders (DistilBERT, ModernBERT-base/large, MacBERT)."""

    def __init__(self, spec: ModelSpec, device: str | None = None) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification

        from nlp_text_classification.experiments.batch_b_resources import load_tokenizer
        from nlp_text_classification.training.transformer_runner import load_checkpoint

        self.spec = spec
        requested = device or os.environ.get("EMOTIONLAB_ENCODER_DEVICE")
        if requested:
            self.device_name = requested
        else:
            self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_name == "cuda" and not torch.cuda.is_available():
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)

        if spec.tokenizer_cache_dir is None:
            raise ValueError(f"Missing tokenizer_cache_dir for {spec.model_id}")
        self.tokenizer = load_tokenizer(spec.tokenizer_cache_dir)

        config = AutoConfig.from_pretrained(
            spec.run_dir / "checkpoint",
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForSequenceClassification.from_config(
            config,
            trust_remote_code=False,
            attn_implementation="sdpa",
        ).to(self.device)
        weights = load_checkpoint(spec.checkpoint_path, self.device)
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.eval()

    def unload(self) -> None:
        import gc

        import torch

        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if self.device_name == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, text: str, top_k: int = 8) -> PredictionResult:
        import torch

        started = time.perf_counter()
        chinese = self.spec.dataset == "movie_reviews_zh"
        prepared = model_text(text.strip(), chinese=chinese)

        full_ids = self.tokenizer(
            prepared,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )["input_ids"]
        raw_count = len(full_ids)
        truncated = raw_count > self.spec.max_length

        encoded = self.tokenizer(
            prepared,
            add_special_tokens=True,
            truncation=True,
            max_length=self.spec.max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        valid_len = len(encoded["input_ids"])
        multiple = self.spec.padding_multiple
        padded_len = min(
            self.spec.max_length,
            max(multiple, int(math.ceil(valid_len / multiple) * multiple)),
        )
        pad_len = max(0, padded_len - valid_len)
        pad_id = int(self.tokenizer.pad_token_id)

        features: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(
                [encoded["input_ids"] + [pad_id] * pad_len],
                dtype=torch.long,
                device=self.device,
            ),
            "attention_mask": torch.tensor(
                [encoded["attention_mask"] + [0] * pad_len],
                dtype=torch.long,
                device=self.device,
            ),
        }
        if "token_type_ids" in encoded and getattr(self.model.config, "type_vocab_size", 0):
            features["token_type_ids"] = torch.tensor(
                [encoded["token_type_ids"] + [0] * pad_len],
                dtype=torch.long,
                device=self.device,
            )

        use_bf16 = self.device_name == "cuda" and self.spec.precision == "bf16"
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = self.model(**features).logits
            logits_f = logits.float()
            if chinese:
                p_pos = float(logits_f.softmax(dim=-1)[0, 1].item())
                scores = [1.0 - p_pos, p_pos]
                positive_score: float | None = p_pos
                predicted = ["positive"] if p_pos >= self.spec.threshold else ["negative"]
            else:
                probs = logits_f.sigmoid()[0].detach().cpu().numpy().tolist()
                scores = [float(p) for p in probs]
                positive_score = None
                predicted = [
                    label
                    for label, score in zip(self.spec.labels, scores, strict=True)
                    if score >= self.spec.threshold
                ]

        latency_ms = (time.perf_counter() - started) * 1000.0
        return PredictionResult(
            model_id=self.spec.model_id,
            model_code=self.spec.model_code,
            display_name=self.spec.display_name,
            family=self.spec.family,
            dataset=self.spec.dataset,
            seed=self.spec.seed,
            run_id=self.spec.run_id,
            score_kind=self.spec.score_kind,
            threshold=self.spec.threshold,
            labels=list(self.spec.labels),
            scores=scores,
            positive_score=positive_score,
            predicted_labels=predicted,
            top_k=_build_top_k(
                list(self.spec.labels),
                scores,
                self.spec.threshold,
                self.spec.score_kind,
                self.spec.dataset,
                k=top_k,
            ),
            latency_ms=round(latency_ms, 2),
            device=self.device_name,
            preprocessing={
                "segmentation_applied": False,
                "token_count": raw_count,
                "truncated": truncated,
                "preview_tokens": self.tokenizer.convert_ids_to_tokens(
                    encoded["input_ids"][:24]
                ),
            },
        )


class QwenWorkerProxyAdapter:
    """HTTP proxy forwarding Qwen3.5-4B S4 requests to the isolated WSL2 worker."""

    def __init__(self, worker_url: str | None = None) -> None:
        self.worker_url = (
            worker_url or os.environ.get("EMOTIONLAB_QWEN_WORKER_URL") or "http://127.0.0.1:8001"
        ).rstrip("/")

    def probe_health(self, timeout: float = 1.5) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.worker_url}/health", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "configured_url": self.worker_url,
                    "online": True,
                    **data,
                }
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return {
                "configured_url": self.worker_url,
                "online": False,
                "status": "offline",
                "detail": str(exc),
                "start_command": "wsl bash scripts/start_qwen_worker_wsl.sh",
            }

    def predict(
        self,
        spec: ModelSpec,
        text: str,
        top_k: int = 8,
        timeout: float = 45.0,
    ) -> PredictionResult:
        payload = json.dumps({"text": text, "seed": spec.seed}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.worker_url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code >= 500:
                raise WorkerUnavailableError(f"Qwen worker returned HTTP {exc.code}: {detail}") from exc
            raise ValueError(detail) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise WorkerUnavailableError(
                f"WSL2 Qwen worker is offline at {self.worker_url} ({exc}). "
                "Start it with: wsl bash scripts/start_qwen_worker_wsl.sh"
            ) from exc

        scores = [float(x) for x in body["scores"]]
        predicted = [
            label
            for label, score in zip(spec.labels, scores, strict=True)
            if score >= spec.threshold
        ]
        return PredictionResult(
            model_id=spec.model_id,
            model_code=spec.model_code,
            display_name=spec.display_name,
            family=spec.family,
            dataset=spec.dataset,
            seed=spec.seed,
            run_id=spec.run_id,
            score_kind=spec.score_kind,
            threshold=spec.threshold,
            labels=list(spec.labels),
            scores=scores,
            positive_score=None,
            predicted_labels=predicted,
            top_k=_build_top_k(
                list(spec.labels),
                scores,
                spec.threshold,
                spec.score_kind,
                spec.dataset,
                k=top_k,
            ),
            latency_ms=round(float(body["latency_ms"]), 2),
            device=str(body.get("device", "wsl2-cuda")),
            preprocessing=dict(body.get("preprocessing", {})),
        )

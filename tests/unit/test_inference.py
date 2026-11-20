"""Unit and integration tests for the unified inference layer and FastAPI endpoints."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nlp_text_classification.inference import (
    InferenceEngine,
    ModelRegistry,
    WorkerUnavailableError,
)
from nlp_text_classification.inference.chinese_seg import VocabularySegmenter, is_pretokenized_zh
from nlp_text_classification.inference.qwen_worker import (
    QwenService,
    make_handler,
    tokenize_qwen_single,
)


def _asgi_request(
    app,
    method: str,
    path: str,
    *,
    query: str = "",
    payload: dict | None = None,
) -> tuple[int, dict]:
    body_bytes = json.dumps(payload).encode("utf-8") if payload is not None else b""
    headers = [(b"host", b"testserver")]
    if payload is not None:
        headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode("ascii")),
            ]
        )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query.encode("ascii"),
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    sent_body = False
    status_code = 500
    chunks: list[bytes] = []

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    raw = b"".join(chunks).decode("utf-8")
    return status_code, json.loads(raw) if raw else {}


def test_chinese_segmenter_raw_vs_pretokenized():
    seg = VocabularySegmenter(["这部", "电影", "真的", "很", "好看", "非常", "精彩"])
    assert is_pretokenized_zh("这 部 电影 真 好看") is True
    assert is_pretokenized_zh("这部电影真的非常精彩") is False

    spaced, tokens, applied = seg.segment("这部电影真的非常精彩！")
    assert applied is True
    assert tokens == ["这部", "电影", "真的", "非常", "精彩", "！"]
    assert spaced == "这部 电影 真的 非常 精彩 ！"

    spaced_mixed, tokens_mixed, applied_mixed = seg.segment("我看过 love actually 这部电影")
    assert applied_mixed is True
    assert "love" in tokens_mixed and "actually" in tokens_mixed
    assert "电影" in tokens_mixed
    assert " ".join(tokens_mixed) == spaced_mixed

    spaced_pre, tokens_pre, applied_pre = seg.segment("这 部 电影 很 好看")
    assert applied_pre is False
    assert tokens_pre == ["这", "部", "电影", "很", "好看"]
    assert spaced_pre == "这 部 电影 很 好看"


def test_registry_discovers_all_models_and_runs(frozen_experiment_root: Path):
    registry = ModelRegistry(frozen_experiment_root)
    en_models = registry.list_default_specs("goemotions")
    zh_models = registry.list_default_specs("movie_reviews_zh")

    assert len(en_models) == 11
    assert len(zh_models) == 8
    assert len(registry.list_all_runs()) == 62

    e1_en = registry.resolve("goemotions/E1")
    assert e1_en.score_kind == "decision"
    assert e1_en.threshold == pytest.approx(-0.6)

    b4_2026 = registry.resolve("movie_reviews_zh/B4", seed=2026)
    assert b4_2026.score_kind == "softmax_positive"
    assert b4_2026.threshold == pytest.approx(0.7)

    qwen_123 = registry.resolve("goemotions/Qwen3.5-4B@123")
    assert qwen_123.seed == 123
    assert qwen_123.runtime == "wsl2_worker"
    assert qwen_123.threshold == pytest.approx(0.3)

    with pytest.raises(KeyError):
        registry.resolve("goemotions/UNKNOWN")


def test_batch_a_live_inference_en_and_zh_short_text(frozen_experiment_root: Path):
    engine = InferenceEngine(frozen_experiment_root)

    en_svm = engine.predict_single(
        "Thank you so much for your help today, I really appreciate it!",
        "goemotions/E1",
    )
    assert en_svm.score_kind == "decision"
    assert len(en_svm.scores) == 28
    assert "gratitude" in en_svm.predicted_labels

    # Short 1-word input tests TextCNN min_len padding guard (kernel size max = 5)
    en_cnn_short = engine.predict_single("thanks", "goemotions/E2")
    assert en_cnn_short.score_kind == "sigmoid"
    assert len(en_cnn_short.scores) == 28
    assert all(0.0 <= s <= 1.0 for s in en_cnn_short.scores)

    zh_svm = engine.predict_single("这部电影非常精彩感人，演员演技很棒", "movie_reviews_zh/E1")
    assert zh_svm.score_kind == "decision"
    assert zh_svm.preprocessing["segmentation_applied"] is True
    assert zh_svm.predicted_labels == ["positive"]

    zh_lstm = engine.predict_single("太烂了，无聊透顶，浪费票钱", "movie_reviews_zh/E3-LSTM")
    assert zh_lstm.score_kind == "sigmoid"
    assert zh_lstm.preprocessing["segmentation_applied"] is True
    assert zh_lstm.predicted_labels == ["negative"]


def test_batch_b_encoder_live_inference_en_and_zh(frozen_experiment_root: Path):
    engine = InferenceEngine(frozen_experiment_root)

    b1 = engine.predict_single(
        "Thank you so much for teaching me something new today!",
        "goemotions/B1",
    )
    assert b1.score_kind == "sigmoid"
    assert len(b1.scores) == 28
    assert "gratitude" in b1.predicted_labels

    b4 = engine.predict_single(
        "这部电影节奏紧凑，结尾非常温暖动人，强烈推荐观看。",
        "movie_reviews_zh/B4",
        seed=123,
    )
    assert b4.score_kind == "softmax_positive"
    assert b4.seed == 123
    assert len(b4.scores) == 2
    assert b4.positive_score is not None
    assert 0.0 <= b4.positive_score <= 1.0
    assert b4.predicted_labels == ["positive"]


def test_qwen_proxy_unavailable_and_http_roundtrip(frozen_experiment_root: Path):
    closed_engine = InferenceEngine(frozen_experiment_root, qwen_worker_url="http://127.0.0.1:59999")
    with pytest.raises(WorkerUnavailableError):
        closed_engine.predict_single("I am so grateful!", "goemotions/Qwen3.5-4B")

    mock_service = MagicMock(spec=QwenService)
    mock_service.health.return_value = {
        "status": "ok",
        "model_id": "goemotions/Qwen3.5-4B",
        "loaded_seed": 42,
        "loaded_seeds": [42],
        "available_seeds": [42, 123, 2026],
        "device": "mock-cuda",
        "vram_free_mib": 7100.0,
    }
    mock_scores = [0.02] * 28
    mock_scores[15] = 0.88  # gratitude
    mock_service.predict.return_value = {
        "seed": 42,
        "scores": mock_scores,
        "threshold": 0.2,
        "latency_ms": 123.4,
        "device": "mock-cuda",
        "preprocessing": {
            "segmentation_applied": False,
            "token_count": 5,
            "truncated": False,
            "preview_tokens": ["Thank", " you"],
        },
    }

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(mock_service))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        live_engine = InferenceEngine(
            frozen_experiment_root, qwen_worker_url=f"http://127.0.0.1:{port}"
        )
        health = live_engine.health()
        assert health["qwen_worker"]["online"] is True
        res = live_engine.predict_single("Thank you!", "goemotions/Qwen3.5-4B", seed=42)
        assert res.predicted_labels == ["gratitude"]
        assert res.scores[15] == pytest.approx(0.88)
    finally:
        server.shutdown()
        server.server_close()


def test_qwen_tokenization_helper_padding_multiple_8():
    fake_tok = MagicMock()
    fake_tok.return_value = {"input_ids": [10, 20, 30, 40, 50]}
    fake_tok.convert_ids_to_tokens.return_value = ["a", "b", "c", "d", "e"]

    out = tokenize_qwen_single(fake_tok, "hello world", max_length=64, pad_token_id=248044)
    assert len(out["input_ids"]) == 8
    assert out["input_ids"][5:] == [248044, 248044, 248044]
    assert out["attention_mask"] == [1, 1, 1, 1, 1, 0, 0, 0]
    assert out["truncated"] is False


def test_fastapi_endpoints_with_offline_qwen_graceful_503(monkeypatch, frozen_experiment_root: Path):
    pytest.importorskip("fastapi")
    from apps.api import main as api_main

    engine = InferenceEngine(frozen_experiment_root, qwen_worker_url="http://127.0.0.1:59998")
    monkeypatch.setattr(api_main, "_ENGINE", engine)

    code, health = _asgi_request(api_main.app, "GET", "/api/health")
    assert code == 200
    assert health["qwen_worker"]["online"] is False

    code, models = _asgi_request(api_main.app, "GET", "/api/models", query="dataset=goemotions")
    assert code == 200
    assert models["count"] == 11

    code, labels = _asgi_request(api_main.app, "GET", "/api/labels", query="dataset=movie_reviews_zh")
    assert code == 200
    assert labels["labels"] == ["negative", "positive"]

    code, runs = _asgi_request(api_main.app, "GET", "/api/runs")
    assert code == 200
    assert runs["count"] == 62

    code, single = _asgi_request(
        api_main.app,
        "POST",
        "/api/predict/single",
        payload={"text": "I love this so much!", "model_id": "goemotions/E1"},
    )
    assert code == 200
    assert "love" in single["result"]["predicted_labels"]

    code, qwen_err = _asgi_request(
        api_main.app,
        "POST",
        "/api/predict/single",
        payload={"text": "I love this!", "model_id": "goemotions/Qwen3.5-4B"},
    )
    assert code == 503
    assert qwen_err["code"] == "QWEN_WORKER_OFFLINE"

    code, batch = _asgi_request(
        api_main.app,
        "POST",
        "/api/predict/batch",
        payload={
            "text": "这部电影真不错！",
            "model_ids": ["movie_reviews_zh/E1", "movie_reviews_zh/E2"],
        },
    )
    assert code == 200
    assert len(batch["results"]) == 2
    assert batch["errors"] == {}

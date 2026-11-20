"""FastAPI unified inference service wrapping Batch A, Batch B Encoders, and WSL2 Qwen S4."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nlp_text_classification.data.loaders.goemotions import load_labels
from nlp_text_classification.experiments.e1_svm import predict as legacy_e1_predict
from nlp_text_classification.inference import InferenceEngine, WorkerUnavailableError

app = FastAPI(title="EmotionLab Inference API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_ENGINE: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine()
    return _ENGINE


class SinglePredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    model_id: str = Field(min_length=1, max_length=120)
    seed: int | None = None
    top_k: int = Field(default=8, ge=1, le=28)


class BatchPredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    model_ids: list[str] = Field(min_length=1, max_length=12)
    seed: int | None = None
    top_k: int = Field(default=8, ge=1, le=28)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    run_id: str | None = None
    model_id: str | None = None
    seed: int | None = None
    top_k: int = Field(default=8, ge=1, le=28)


def _legacy_run_dir(run_id: str | None) -> Path:
    root = Path(os.environ.get("NLP_TEXT_CLASSIFICATION_ARTIFACTS", "artifacts")).resolve() / "runs"
    target = run_id or os.environ.get("NLP_TEXT_CLASSIFICATION_RUN_ID")
    if not target:
        raise HTTPException(status_code=503, detail="No trained model configured")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", target):
        raise HTTPException(status_code=400, detail="Invalid run_id")
    path = root / target
    if not (path / "model.joblib").is_file():
        raise HTTPException(status_code=404, detail="Requested model artifact not found")
    return path


@app.get("/api/health")
def health() -> dict[str, Any]:
    return get_engine().health()


@app.get("/api/models")
def list_models(dataset: str | None = Query(default=None)) -> dict[str, Any]:
    if dataset is not None and dataset not in {"goemotions", "movie_reviews_zh"}:
        raise HTTPException(status_code=400, detail=f"Unsupported dataset: {dataset}")
    models = get_engine().list_models(dataset=dataset)
    return {"models": models, "count": len(models)}


@app.get("/api/labels")
def labels(dataset: str = Query(default="goemotions")) -> dict[str, Any]:
    engine = get_engine()
    if dataset in engine.registry.datasets_labels:
        return {"dataset": dataset, "labels": list(engine.registry.datasets_labels[dataset])}
    data_env = os.environ.get("NLP_TEXT_CLASSIFICATION_DATA")
    if dataset == "goemotions" and data_env and (Path(data_env) / "emotions.txt").exists():
        return {"dataset": "goemotions", "labels": load_labels(Path(data_env))}
    raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")


@app.get("/api/runs")
def runs() -> dict[str, Any]:
    formal = get_engine().list_runs()
    root = Path(os.environ.get("NLP_TEXT_CLASSIFICATION_ARTIFACTS", "artifacts")) / "runs"
    legacy = []
    for path in sorted(root.glob("*/run_manifest.json"), reverse=True):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        legacy.append(
            {
                "run_id": path.parent.name,
                "experiment": metadata["experiment"],
                "dataset": metadata["dataset"],
                "has_model": (path.parent / "model.joblib").is_file(),
            }
        )
    return {"runs": formal, "legacy_runs": legacy, "count": len(formal)}


@app.post("/api/predict/single")
def predict_single(payload: SinglePredictRequest):
    engine = get_engine()
    try:
        result = engine.predict_single(
            payload.text,
            payload.model_id,
            seed=payload.seed,
            top_k=payload.top_k,
        )
        return {"result": result.to_dict()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except WorkerUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "detail": str(exc),
                "code": "QWEN_WORKER_OFFLINE",
                "start_command": "wsl bash scripts/start_qwen_worker_wsl.sh",
            },
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> dict[str, Any]:
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Input text must not be empty")
    return get_engine().predict(
        payload.text,
        payload.model_ids,
        seed=payload.seed,
        top_k=payload.top_k,
    )


@app.post("/api/predict")
def classify(payload: PredictRequest):
    if payload.model_id:
        return predict_single(
            SinglePredictRequest(
                text=payload.text,
                model_id=payload.model_id,
                seed=payload.seed,
                top_k=payload.top_k,
            )
        )
    try:
        return {"result": legacy_e1_predict(_legacy_run_dir(payload.run_id), [payload.text])[0]}
    except HTTPException:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

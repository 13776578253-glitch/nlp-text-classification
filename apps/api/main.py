"""FastAPI demo service; inference uses pre-trained local artifacts only."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nlp_text_classification.data.goemotions import load_labels
from nlp_text_classification.experiments.e1_svm import predict

app = FastAPI(title="nlp-text-classification API", version="0.1.0")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    run_id: str | None = None


def _run_dir(run_id: str | None) -> Path:
    root = Path(os.environ.get("NLP_TEXT_CLASSIFICATION_ARTIFACTS", "artifacts")).resolve() / "runs"
    if run_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise HTTPException(status_code=400, detail="Invalid run_id")
        path = root / run_id
    else:
        configured = os.environ.get("NLP_TEXT_CLASSIFICATION_RUN_ID")
        if not configured:
            raise HTTPException(status_code=503, detail="No trained model configured")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", configured):
            raise HTTPException(status_code=503, detail="Invalid server run configuration")
        path = root / configured
    if not (path / "model.joblib").is_file():
        raise HTTPException(status_code=404, detail="Requested model artifact not found")
    return path


@app.get("/api/health")
def health():
    return {"status": "ok", "task": "goemotions-multilabel"}


@app.get("/api/labels")
def labels():
    path = Path(os.environ.get("NLP_TEXT_CLASSIFICATION_DATA", "data/goemotions"))
    if not (path / "emotions.txt").exists():
        raise HTTPException(status_code=503, detail="Download GoEmotions dataset first")
    return {"labels": load_labels(path)}


@app.get("/api/runs")
def runs():
    root = Path(os.environ.get("NLP_TEXT_CLASSIFICATION_ARTIFACTS", "artifacts")) / "runs"
    result = []
    for path in sorted(root.glob("*/run_manifest.json"), reverse=True):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        result.append({"run_id": path.parent.name, "experiment": metadata["experiment"],
                       "dataset": metadata["dataset"], "has_model": (path.parent / "model.joblib").is_file()})
    return {"runs": result}


@app.post("/api/predict")
def classify(payload: PredictRequest):
    try:
        return {"result": predict(_run_dir(payload.run_id), [payload.text])[0]}
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

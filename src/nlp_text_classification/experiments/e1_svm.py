"""E1: fully working TF-IDF + One-vs-Rest Linear SVM baseline."""
from __future__ import annotations

import json
import platform
import hashlib
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from nlp_text_classification.data.goemotions import SPLIT_FILES, load_labels, load_split, validate_dataset
from nlp_text_classification.evaluation.multilabel import evaluate, indicator_matrix, select_threshold
from nlp_text_classification.training.run import git_commit, new_run_dir, write_json


def train(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    max_features: int = 100_000,
    min_df: int = 2,
    strict_data: bool = True,
) -> Path:
    started_at = time.perf_counter()
    dataset_meta = validate_dataset(data_dir, strict=strict_data)
    data_hashes = {name: hashlib.sha256((Path(data_dir) / filename).read_bytes()).hexdigest()
                   for name, filename in SPLIT_FILES.items()}
    data_hashes["labels"] = hashlib.sha256((Path(data_dir) / "emotions.txt").read_bytes()).hexdigest()
    labels = load_labels(data_dir)
    tr = load_split(data_dir, "train")
    va = load_split(data_dir, "validation")
    x_train, x_val = [r.text for r in tr], [r.text for r in va]
    y_train = indicator_matrix(tr, len(labels))
    y_val = indicator_matrix(va, len(labels))
    if not np.all(np.any(y_train, axis=0)) or not np.all(np.any(1-y_train, axis=0)):
        raise ValueError("Training split must contain positive and negative examples for each label")
    vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=min_df,
        max_features=max_features, sublinear_tf=True
    )
    xtr = vec.fit_transform(x_train)
    xva = vec.transform(x_val)
    model = OneVsRestClassifier(LinearSVC(class_weight="balanced", random_state=seed))
    model.fit(xtr, y_train)
    scores_val = np.asarray(model.decision_function(xva))
    tuning = select_threshold(y_val, scores_val, score_kind="decision")
    val_metrics = evaluate(y_val, scores_val, tuning["threshold"])

    run_dir = new_run_dir(output_root, "e1-svm", seed)
    joblib.dump({"vectorizer": vec, "classifier": model, "labels": labels}, run_dir / "model.joblib")
    write_json(run_dir / "thresholds.json", tuning)
    write_json(run_dir / "metrics_validation.json", val_metrics)
    write_json(run_dir / "config.json", {
        "experiment": "E1", "seed": seed, "max_features": max_features,
        "min_df": min_df, "ngram_range": [1, 2], "sublinear_tf": True,
        "class_weight": "balanced", "selection_metric": "validation_micro_f1",
    })
    _write_predictions(run_dir / "predictions_validation.jsonl", va, scores_val, labels,
                       tuning["threshold"])
    write_json(run_dir / "run_manifest.json", {
        "experiment": "E1", "dataset": dataset_meta["dataset"],
        "dataset_counts": dataset_meta["counts"], "num_labels": len(labels),
        "seed": seed, "git_commit": git_commit(), "data_sha256": data_hashes,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "model": "TF-IDF (train-only fit) + OneVsRest(LinearSVC)",
        "tfidf_max_features": max_features, "tfidf_min_df": min_df,
        "threshold_source": "validation only", "final_test_evaluated": False,
        "score_kind": "decision", "test_split_used_for_selection": False,
        "training_elapsed_seconds": round(time.perf_counter() - started_at, 3),
    })
    return run_dir


def evaluate_final(run_dir: str | Path, data_dir: str | Path, *, strict_data: bool = True) -> dict:
    """Blind test on a frozen E1 run. Only allowed once for this run."""
    run_dir = Path(run_dir)
    output = run_dir / "metrics_test.json"
    if output.exists():
        raise FileExistsError(f"This run already has final test metrics: {output}")
    meta_path = run_dir / "run_manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("experiment") != "E1":
        raise ValueError("Only E1 runs can be evaluated with this command")
    validate_dataset(data_dir, strict=strict_data)
    for name, filename in SPLIT_FILES.items():
        actual_sha = hashlib.sha256((Path(data_dir) / filename).read_bytes()).hexdigest()
        if meta["data_sha256"][name] != actual_sha:
            raise ValueError(f"Dataset changed since training: {filename}")
    labels_sha = hashlib.sha256((Path(data_dir) / "emotions.txt").read_bytes()).hexdigest()
    if meta["data_sha256"]["labels"] != labels_sha:
        raise ValueError("Label mapping changed since training")
    bundle, threshold = _load_trusted_run(str(run_dir.resolve()))
    test = load_split(data_dir, "test")
    scores = np.asarray(bundle["classifier"].decision_function(
        bundle["vectorizer"].transform([row.text for row in test])
    ))
    y_true = indicator_matrix(test, len(bundle["labels"]))
    metrics = evaluate(y_true, scores, threshold)
    _write_predictions(run_dir / "predictions_test.jsonl", test, scores, bundle["labels"], threshold)
    write_json(output, metrics)
    meta["final_test_evaluated"] = True
    meta["final_test_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(meta_path, meta)
    return metrics


def _write_predictions(path: Path, rows: list, scores: np.ndarray,
                       label_names: list[str], threshold: float) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example, row_scores in zip(rows, scores):
            selected = [i for i, score in enumerate(row_scores) if score >= threshold]
            item = {"sample_id": example.sample_id, "gold_labels": list(example.labels),
                    "predicted_labels": selected,
                    "scores": [round(float(value), 6) for value in row_scores]}
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


@lru_cache(maxsize=4)
def _load_trusted_run(path: str):
    run_dir = Path(path)
    # Never deserialize third-party or user-supplied pickle/joblib models.
    bundle = joblib.load(run_dir / "model.joblib")
    threshold = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))["threshold"]
    return bundle, threshold


def predict(run_dir: str | Path, texts: list[str]) -> list[dict]:
    run_dir = Path(run_dir)
    if not texts or any(not s.strip() for s in texts):
        raise ValueError("Prediction requires nonempty texts")
    # Only load artifacts from trusted local training runs; joblib uses pickle internally.
    bundle, threshold = _load_trusted_run(str(run_dir.resolve()))
    score_matrix = np.asarray(bundle["classifier"].decision_function(bundle["vectorizer"].transform(texts)))
    result = []
    for scores in score_matrix:
        ranking = sorted(zip(bundle["labels"], scores), key=lambda item: item[1], reverse=True)
        result.append({"predicted": [label for label, s in ranking if s >= threshold],
                       "scores": [{"label": label, "score": float(score)} for label, score in ranking],
                       "score_kind": "decision", "threshold": float(threshold)})
    return result

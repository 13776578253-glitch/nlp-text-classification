"""Batch A TF-IDF + LinearSVC for both dataset tasks."""

from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from nlp_text_classification.evaluation.batch_metrics import select_threshold


def predict_model(bundle, rows):
    features = bundle["vectorizer"].transform([row.text for row in rows])
    scores = np.asarray(bundle["classifier"].decision_function(features))
    return scores.reshape(len(rows), -1)


def fit(data, resolved, attempt, seed):
    params = resolved["settings"]
    y_train = data.encode(data.train)
    if not np.all(y_train.any(0) & ~y_train.all(0)):
        raise ValueError("Each training target requires positive and negative examples")
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(params["ngram_range"]), min_df=params["min_df"],
        max_features=params["max_features"], sublinear_tf=params["sublinear_tf"],
        tokenizer=str.split if data.binary else None, token_pattern=None if data.binary else
        r"(?u)\b\w\w+\b", lowercase=not data.binary,
    )
    started = time.perf_counter()
    design = vectorizer.fit_transform([row.text for row in data.train])
    svc = LinearSVC(C=params["c"], class_weight=params["class_weight"], random_state=seed)
    classifier = svc if data.binary else OneVsRestClassifier(svc)
    classifier.fit(design, y_train[:, 0] if data.binary else y_train)
    bundle = {"vectorizer": vectorizer, "classifier": classifier}
    scores = predict_model(bundle, data.validation)
    tuning = select_threshold(data.encode(data.validation), scores, data.binary,
                              resolved["selection"], "decision")
    attempt.epoch(epoch=0, val_macro_f1=tuning["macro_f1"],
                  threshold=tuning["threshold"], elapsed_seconds=time.perf_counter() - started)
    checkpoint = attempt.path / "checkpoint" / "model.joblib"
    joblib.dump(bundle, checkpoint)
    restored = joblib.load(checkpoint)
    train_scores = np.asarray(classifier.decision_function(design)).reshape(len(y_train), -1)
    return (scores, predict_model(restored, data.validation), train_scores,
            {"stop_reason": "single_fit", "features": len(vectorizer.vocabulary_)})

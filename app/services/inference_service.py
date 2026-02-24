from __future__ import annotations
import joblib
import pandas as pd
from typing import Any

MODEL_PATH = "artifacts/model.joblib"

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pipeline = get_pipeline()
    df = pd.DataFrame(rows)

    preds = pipeline.predict(df)
    proba = pipeline.predict_proba(df)
    probs = proba[:, 1]

    results: list[dict[str, Any]] = []
    for i, pred in enumerate(preds):
        item = {"row_index": i, "prediction": int(pred), "probability": float(probs[i])}
        results.append(item)

    return results

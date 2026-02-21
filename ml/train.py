from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import make_xy, build_preprocessor


# =========================
# Config
# =========================
DATA_PATH = Path("data/data.csv")
ARTIFACT_DIR = Path("artifacts")

TARGET_COL = "Label"
SEED = 17


# =========================
# Data
# =========================
def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    df = df.drop_duplicates(subset=feature_cols + [TARGET_COL]).copy()

    X, y = make_xy(df)
    return X, y


# =========================
# Model / Tuning
# =========================
def make_xgboost_pipeline(X_full: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X_full)

    return Pipeline([
        ("preprocess", pre),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=SEED,
            tree_method="hist",
            n_jobs=-1,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
        )),
    ])


def get_xgb_param_candidates() -> dict:
    return {
        "clf__n_estimators": [200, 300, 500, 700, 1000],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__max_depth": [3, 4, 5, 6, 8, 10],
        "clf__min_child_weight": [1, 2, 3, 5, 7],
        "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__gamma": [0, 0.1, 0.3, 0.5, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
    }


def tune_xgboost_on_full_data(
    base_pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = SEED,
    n_iter: int = 25,
) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=get_xgb_param_candidates(),
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        verbose=2,
        random_state=seed,
        refit=True,
    )

    print("\nStarting XGBoost hyperparameter tuning on full dataset (CV)...")
    search.fit(X, y)

    print(f"\nBest CV AUC: {search.best_score_:.6f}")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    return search


# =========================
# Artifact I/O
# =========================
def save_artifacts(model: Pipeline, metadata: dict, artifact_dir: Path = ARTIFACT_DIR) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "model_metadata.json"

    joblib.dump(model, model_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model artifact: {model_path}")
    print(f"Saved metadata: {metadata_path}")


# =========================
# Main
# =========================
def main() -> None:
    print("Loading and preparing data...")
    X, y = load_and_prepare_data(DATA_PATH)

    print("Building XGBoost pipeline...")
    base_pipe = make_xgboost_pipeline(X)

    search = tune_xgboost_on_full_data(
        base_pipe=base_pipe,
        X=X,
        y=y,
        n_iter=100,
    )

    final_model: Pipeline = search.best_estimator_

    metadata = {
        "model_name": "xgboost",
        "target_col": TARGET_COL,
        "seed": SEED,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows_total": int(len(X)),
        "expected_input_columns": list(X.columns),
        "training_mode": "full_dataset_refit_after_cv_tuning",
        "tuning": {
            "search_type": "RandomizedSearchCV",
            "scoring": "roc_auc",
            "cv_folds": 5,
            "n_iter": 100,
            "best_cv_auc": float(search.best_score_),
            "best_params": search.best_params_,
        },
        "artifacts": {
            "model": "artifacts/model.joblib",
            "metadata": "artifacts/model_metadata.json",
        },
    }

    save_artifacts(final_model, metadata)


if __name__ == "__main__":
    main()
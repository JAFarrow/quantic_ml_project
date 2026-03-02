from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from features import make_xy, build_preprocessor
from metrics import compute_metrics, Metrics
from torch_model import TorchMLPConfig, fit_predict_proba

# =========================
# Config
# =========================
DATA_PATH = Path("data/data.csv")
TARGET_COL = "Label"
SEED = 17
N_SPLITS = 10
TEST_SIZE = 0.2


# =========================
# Data
# =========================
def load_and_prepare_data(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    df = df.drop_duplicates(subset=feature_cols + [TARGET_COL]).copy()

    X, y = make_xy(df)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# =========================
# Models
# =========================
def make_models(preprocessor: Pipeline) -> Dict[str, Optional[Pipeline]]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=10000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    DecisionTreeClassifier(
                        random_state=SEED,
                        class_weight="balanced",
                        max_depth=None,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=SEED,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=SEED,
                        tree_method="hist",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "adaboost": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    AdaBoostClassifier(
                        estimator=DecisionTreeClassifier(
                            max_depth=2, random_state=SEED
                        ),
                        n_estimators=500,
                        learning_rate=0.05,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=31,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "torch_mlp": None,
    }


# =========================
# Training / Evaluation
# =========================
def run_torch_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    fold_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pre_fold = build_preprocessor(X_tr)
    Xt_tr = pre_fold.fit_transform(X_tr)
    Xt_val = pre_fold.transform(X_val)

    cfg = TorchMLPConfig(
        epochs=10,
        batch_size=512,
        seed=fold_seed,
        verbose=True,
    )
    y_pred, y_proba = fit_predict_proba(Xt_tr, y_tr.to_numpy(), Xt_val, cfg=cfg)
    return y_pred, y_proba


def run_sklearn_fold(
    pipe: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    fold_pipe = clone(pipe)
    fold_pipe.fit(X_tr, y_tr)
    y_pred = fold_pipe.predict(X_val)
    y_proba = fold_pipe.predict_proba(X_val)[:, 1]
    return y_pred, y_proba


def cross_validate_models(
    models: Dict[str, Optional[Pipeline]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    rows = []

    for name, pipe in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training model: {name}")
        print(f"{'=' * 60}")

        fold_aucs, fold_accs = [], []
        model_start = time.time()

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train), start=1
        ):
            fold_start = time.time()
            print(f"\n[{name}] Fold {fold}/{cv.n_splits}")

            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            if name == "torch_mlp":
                y_pred, y_proba = run_torch_fold(
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_val=X_val,
                    fold_seed=SEED + fold,
                )
            else:
                y_pred, y_proba = run_sklearn_fold(
                    pipe=pipe,
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_val=X_val,
                )

            m: Metrics = compute_metrics(y_val, y_pred, y_proba)
            fold_time = time.time() - fold_start
            print(f"AUC: {m.auc:.5f} | ACC: {m.accuracy:.5f} | time: {fold_time:.2f}s")

            fold_aucs.append(m.auc)
            fold_accs.append(m.accuracy)

        total_time = time.time() - model_start
        rows.append(
            {
                "model": name,
                "cv_auc_mean": float(np.nanmean(fold_aucs)),
                "cv_auc_std": float(np.nanstd(fold_aucs, ddof=1)),
                "cv_acc_mean": float(np.nanmean(fold_accs)),
                "cv_acc_std": float(np.nanstd(fold_accs, ddof=1)),
                "total_time_sec": round(total_time, 2),
            }
        )

        print(f"\nFinished {name} in {total_time:.2f} sec")

    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)


def evaluate_best_on_test(
    best_model_name: str,
    models: Dict[str, Optional[Pipeline]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Metrics:
    if best_model_name == "torch_mlp":
        pre_final = build_preprocessor(X_train)
        Xt_tr = pre_final.fit_transform(X_train)
        Xt_te = pre_final.transform(X_test)

        cfg = TorchMLPConfig(
            epochs=10,
            batch_size=512,
            seed=SEED,
            verbose=True,
        )
        y_test_pred, y_test_proba = fit_predict_proba(
            Xt_tr, y_train.to_numpy(), Xt_te, cfg=cfg
        )
    else:
        best_pipe = clone(models[best_model_name])
        best_pipe.fit(X_train, y_train)
        y_test_pred = best_pipe.predict(X_test)
        y_test_proba = best_pipe.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_test_pred, y_test_proba)


def main() -> None:
    X, y = load_and_prepare_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pre = build_preprocessor(X_train)
    models = make_models(pre)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    out = cross_validate_models(models, X_train, y_train, cv)

    print("\n\nFINAL RESULTS")
    print(out.to_string(index=False))

    best_model_name = out.iloc[0]["model"]
    print(f"\nSelected production model from CV: {best_model_name}")

    test_metrics = evaluate_best_on_test(
        best_model_name=best_model_name,
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nFINAL HOLD-OUT TEST RESULTS")
    print(f"model: {best_model_name}")
    print(f"test_auc: {test_metrics.auc:.6f}")
    print(f"test_acc: {test_metrics.accuracy:.6f}")
    print("confusion_matrix:")
    print(test_metrics.confusion)


if __name__ == "__main__":
    main()

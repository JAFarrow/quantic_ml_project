from __future__ import annotations

from pathlib import Path
import pandas as pd
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from features import make_xy, build_preprocessor


DATA_PATH = Path("data/data.csv")
SEED = 42


def make_models(pre):
    return {
        "logistic_regression": Pipeline([
            ("preprocess", pre),
            ("clf", LogisticRegression(
                max_iter=10000,
                random_state=SEED,
            )),
        ]),
        "decision_tree": Pipeline([
            ("preprocess", pre),
            ("clf", DecisionTreeClassifier(
                random_state=SEED,
                class_weight="balanced",
                max_depth=None,
                min_samples_leaf=5,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocess", pre),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                random_state=SEED,
                class_weight="balanced_subsample",
            )),
        ]),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pre = build_preprocessor(df)
    models = make_models(pre)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    rows = []

    for name, pipe in models.items():
        print(f"\n{'='*60}")
        print(f"Training model: {name}")
        print(f"{'='*60}")

        fold_aucs = []
        fold_accs = []

        model_start = time.time()

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            fold_start = time.time()

            print(f"\n[{name}] Fold {fold}/10")

            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            pipe.fit(X_tr, y_tr)

            y_pred = pipe.predict(X_val)
            y_proba = pipe.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, y_proba)
            acc = accuracy_score(y_val, y_pred)

            fold_time = time.time() - fold_start

            print(f"AUC: {auc:.5f} | ACC: {acc:.5f} | time: {fold_time:.2f}s")

            fold_aucs.append(auc)
            fold_accs.append(acc)

        total_time = time.time() - model_start

        rows.append({
            "model": name,
            "cv_auc_mean": sum(fold_aucs)/len(fold_aucs),
            "cv_auc_std": pd.Series(fold_aucs).std(),
            "cv_acc_mean": sum(fold_accs)/len(fold_accs),
            "cv_acc_std": pd.Series(fold_accs).std(),
            "total_time_sec": round(total_time, 2),
        })

        print(f"\nFinished {name} in {total_time:.2f} sec")

    out = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)

    print("\n\nFINAL RESULTS")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

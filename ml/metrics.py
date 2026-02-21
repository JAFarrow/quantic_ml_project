from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    auc: float
    confusion: np.ndarray


def compute_metrics(y_true, y_pred, y_proba: Optional[np.ndarray]) -> Metrics:
    """
    Computes ACC, AUC, and confusion matrix.
    If AUC cannot be computed (e.g., only one class in y_true), returns np.nan for auc.
    """
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    auc = np.nan
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            auc = np.nan

    return Metrics(
        accuracy=acc,
        auc=auc,
        confusion=cm,
    )

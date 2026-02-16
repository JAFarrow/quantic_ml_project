from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    auc: float
    confusion: np.ndarray


def compute_metrics(y_true, y_pred, y_proba) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc=float(roc_auc_score(y_true, y_proba)),
        confusion=confusion_matrix(y_true, y_pred),
    )

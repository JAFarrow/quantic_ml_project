import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.torch_model import TorchMLPConfig, fit_predict_proba, to_dense_float32


@pytest.fixture
def toy_data():
    X_train = np.array(
        [
            [0.0, 0.1],
            [1.0, 0.9],
            [0.3, 0.4],
            [0.2, 0.2],
            [0.8, 0.3],
            [0.7, 0.6],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 1, 0, 0, 1, 1], dtype=np.int32)
    X_val = np.array(
        [
            [0.05, 0.2],
            [0.9, 0.95],
        ],
        dtype=np.float32,
    )
    return X_train, y_train, X_val


def test_to_dense_float32_from_numpy():
    arr = np.array([[1, 2]], dtype=np.float64)
    out = to_dense_float32(arr)
    assert out.dtype == np.float32
    assert out.shape == arr.shape
    assert out[0, 1] == 2.0


def test_to_dense_float32_from_sparse():
    matrix = sparse.csr_matrix([[0, 1], [2, 0]])
    out = to_dense_float32(matrix)
    assert out.dtype == np.float32
    assert out.tolist() == [[0.0, 1.0], [2.0, 0.0]]


def test_fit_predict_proba_outputs_valid_probs(toy_data):
    X_train, y_train, X_val = toy_data
    cfg = TorchMLPConfig(
        epochs=2,
        batch_size=2,
        hidden1=16,
        hidden2=8,
        dropout=0.0,
        seed=42,
        verbose=False,
    )
    y_pred, y_proba = fit_predict_proba(X_train, y_train, X_val, cfg=cfg)
    assert y_pred.shape == (len(X_val),)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert np.all(0.0 <= y_proba) and np.all(y_proba <= 1.0)

    _, y_proba_repeat = fit_predict_proba(X_train, y_train, X_val, cfg=cfg)
    assert np.allclose(y_proba, y_proba_repeat, rtol=1e-6, atol=1e-6)

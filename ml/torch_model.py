# src/torch_model.py
from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TorchMLPConfig:
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden1: int = 512
    hidden2: int = 128
    dropout: float = 0.2
    threshold: float = 0.5
    seed: int = 42
    verbose: bool = True


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_dense_float32(X) -> np.ndarray:
    """Convert sklearn output (often sparse) -> dense float32 numpy array."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def fit_predict_proba(
    Xt_train,
    y_train,
    Xt_val,
    *,
    cfg: TorchMLPConfig = TorchMLPConfig(),
):
    """
    Train a small MLP on (dense) Xt_train and return (val_pred, val_proba).
    Xt_* can be sparse or dense; will be densified to float32.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _set_seeds(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr_np = to_dense_float32(Xt_train)
    Xva_np = to_dense_float32(Xt_val)

    Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
    ytr = torch.tensor(np.asarray(y_train, dtype=np.float32)).view(-1, 1)

    Xva = torch.tensor(Xva_np, dtype=torch.float32)

    in_dim = Xtr.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden1),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden1, cfg.hidden2),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden2, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        if cfg.verbose:
            print(f"    epoch {ep:02d}/{cfg.epochs} | loss {total_loss/len(Xtr):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xva.to(device)).cpu().numpy().ravel()
        proba = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
        pred = (proba >= cfg.threshold).astype(int)

    return pred, proba

from __future__ import annotations

import numpy as np


def _as_1d(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same number of values.")
    return y_true, y_pred


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error in percent.

    Values where ``y_true == 0`` are ignored to avoid division-by-zero blowups.
    Returns 0.0 when all targets are zero.
    """
    y_true, y_pred = _as_1d(y_true, y_pred)
    denom = np.abs(y_true)
    mask = denom > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom[mask])) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Returns 1.0 for a perfect fit of constant data."""
    y_true, y_pred = _as_1d(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot

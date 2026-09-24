from __future__ import annotations

from typing import Any

import numpy as np


def validate_X(X: Any) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array-like object.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    return X


def validate_xy(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    X = validate_X(X)
    y = np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError("y must be a 1D array-like object or a column vector.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or infinite values.")
    return X, y

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseSurrogateModel(ABC):
    """Abstract base template for all surrogate models in ddmo.

    Subclasses are expected to implement the training and prediction logic
    while keeping the public interface consistent across models.
    """

    def __init__(self, *, name: str = "surrogate", normalize: bool = True):
        self.name = name
        self.normalize = normalize
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.x_train_ = None
        self.y_train_ = None
        self._x_train_norm_ = None
        self._x_mean = None
        self._x_scale = None

    def _validate_inputs(self, X: Any, y: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array-like object.")

        if y is not None:
            y = np.asarray(y, dtype=float)
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            if y.ndim != 1:
                raise ValueError("y must be a 1D array-like object or a column vector.")
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must contain the same number of samples.")

        return X, y

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return X

        if self._x_mean is None or self._x_scale is None:
            self._x_mean = X.mean(axis=0)
            self._x_scale = X.std(axis=0)
            self._x_scale[self._x_scale == 0.0] = 1.0

        return (X - self._x_mean) / self._x_scale

    def _prepare_X_for_prediction(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array-like object.")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was trained on {self.n_features_in_}."
            )
        return self._normalize_features(X) if self.normalize else X

    def fit(self, X: Any, y: Any):
        X, y = self._validate_inputs(X, y)
        self.n_features_in_ = X.shape[1]
        self.x_train_ = X.copy()
        self.y_train_ = y.copy()

        self._x_mean = X.mean(axis=0)
        self._x_scale = X.std(axis=0)
        self._x_scale[self._x_scale == 0.0] = 1.0

        Xn = self._normalize_features(X)
        self._x_train_norm_ = Xn.copy()
        self._fit_impl(Xn, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError(f"The model {self.name} has not been fitted yet.")
        X = self._prepare_X_for_prediction(np.asarray(X, dtype=float))
        y_pred = self._predict_impl(X)
        return np.asarray(y_pred, dtype=float)

    def score(self, X: Any, y: Any) -> float:
        X, y = self._validate_inputs(X, y)
        y_pred = self.predict(X)
        residual = y - y_pred
        return float(np.mean(residual ** 2))

    @abstractmethod
    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        """Model-specific training logic."""

    @abstractmethod
    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Model-specific prediction logic."""

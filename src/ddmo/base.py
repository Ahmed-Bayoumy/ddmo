from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .datasets import validate_X, validate_xy
from .metrics import r2


class BaseSurrogateModel(ABC):
    """Abstract base template for all surrogate models in ddmo.

    Subclasses implement ``_fit_impl`` and ``_predict_impl``, which receive
    inputs that are already validated and (if ``normalize``) standardized with
    statistics computed from the training data in ``fit``.
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

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return X
        if self._x_mean is None or self._x_scale is None:
            raise RuntimeError("Normalization statistics are only available after fit.")
        return (X - self._x_mean) / self._x_scale

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"The model {self.name} has not been fitted yet.")

    def _prepare_X_for_prediction(self, X: Any) -> np.ndarray:
        X = validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was trained on {self.n_features_in_}."
            )
        return self._normalize_features(X)

    def fit(self, X: Any, y: Any):
        X, y = validate_xy(X, y)
        self.is_fitted_ = False
        self.n_features_in_ = X.shape[1]
        self.x_train_ = X.copy()
        self.y_train_ = y.copy()

        if self.normalize:
            self._x_mean = X.mean(axis=0)
            self._x_scale = X.std(axis=0)
            self._x_scale[self._x_scale == 0.0] = 1.0

        Xn = self._normalize_features(X)
        self._x_train_norm_ = Xn.copy()
        self._fit_impl(Xn, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X = self._prepare_X_for_prediction(X)
        return np.asarray(self._predict_impl(X), dtype=float)

    def predict_gradient(self, X: Any) -> np.ndarray:
        """Gradient of the prediction with respect to the inputs, shape ``(n_samples, n_features)``.

        Returned in the original (unnormalized) input units.
        """
        self._check_fitted()
        Xn = self._prepare_X_for_prediction(X)
        grad = np.asarray(self._gradient_impl(Xn), dtype=float)
        # chain rule for the standardization z = (x - mean) / scale
        return grad / self._x_scale if self.normalize else grad

    def score(self, X: Any, y: Any) -> float:
        """Coefficient of determination R^2 (higher is better)."""
        X, y = validate_xy(X, y)
        return r2(y, self.predict(X))

    @abstractmethod
    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        """Model-specific training logic."""

    @abstractmethod
    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Model-specific prediction logic."""

    def _gradient_impl(self, X: np.ndarray) -> np.ndarray:
        """Model-specific gradient with respect to the (normalized) inputs."""
        raise NotImplementedError(f"{type(self).__name__} does not provide gradients.")

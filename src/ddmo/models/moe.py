from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..base import BaseSurrogateModel


class MixtureOfExperts(BaseSurrogateModel):
    """Simple mixture-of-experts wrapper around multiple surrogate models."""

    def __init__(
        self,
        *,
        experts: Sequence[BaseSurrogateModel] | None = None,
        weights: Sequence[float] | None = None,
        name: str = "moe",
        normalize: bool = True,
    ):
        super().__init__(name=name, normalize=normalize)
        self.experts = list(experts or [])
        self.weights = list(weights or [1.0 for _ in self.experts])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if not self.experts:
            raise ValueError("At least one expert model must be provided.")
        if len(self.experts) != len(self.weights):
            self.weights = [1.0 for _ in self.experts]

        for expert in self.experts:
            expert.fit(X, y)

        self.is_fitted_ = True
        self.x_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        predictions = [expert.predict(X) for expert in self.experts]
        stacked = np.column_stack(predictions)
        if stacked.size == 0:
            return np.zeros(len(X))
        return stacked @ np.asarray(self.weights, dtype=float) / sum(self.weights)


class MOE(MixtureOfExperts):
    """Backward-compatible alias for the original model name."""

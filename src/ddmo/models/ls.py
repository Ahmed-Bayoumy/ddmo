from __future__ import annotations

import numpy as np

from ..base import BaseSurrogateModel


class LS(BaseSurrogateModel):
    """Least-squares surrogate using the original matrix-solve formulation."""

    def __init__(
        self,
        *,
        name: str = "ls_surrogate",
        normalize: bool = True,
        degree: int = 1,
        ridge: float = 0.0,
    ):
        super().__init__(name=name, normalize=normalize)
        self.degree = degree
        self.ridge = ridge
        self.coefficients_ = None

    def _expand_features(self, X: np.ndarray) -> np.ndarray:
        if self.degree < 0:
            raise ValueError("degree must be non-negative")

        if self.degree <= 1:
            return np.column_stack([np.ones(X.shape[0]), X])

        X_poly = np.ones((X.shape[0], 1), dtype=float)
        for power in range(1, self.degree + 1):
            X_poly = np.hstack([X_poly, X ** power])
        return X_poly

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        X_design = self._expand_features(X)
        reg = self.ridge * np.eye(X_design.shape[1])
        reg[0, 0] = 0.0
        A = X_design.T @ X_design + reg
        b = X_design.T @ y
        self.coefficients_ = np.linalg.solve(A, b)
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        X_design = self._expand_features(X)
        return X_design @ self.coefficients_


class LinearSurrogate(LS):
    """Backward-compatible alias matching the public naming used in the new package."""

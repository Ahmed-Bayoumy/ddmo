from __future__ import annotations

import numpy as np

from ..base import BaseSurrogateModel


class Kriging(BaseSurrogateModel):
    """Custom kriging-style surrogate using a Gaussian correlation kernel."""

    def __init__(
        self,
        *,
        name: str = "kriging_surrogate",
        normalize: bool = True,
        theta: float = 1.0,
        p: float = 2.0,
        nugget: float = 1e-8,
    ):
        super().__init__(name=name, normalize=normalize)
        self.theta = theta
        self.p = p
        self.nugget = nugget
        self.inv_r_ = None
        self.beta_ = None

    def _distance_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, None, :] - X2[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _correlation_matrix(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-((self.theta * distances) ** self.p))

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        D = self._distance_matrix(X, X)
        R = self._correlation_matrix(D)
        R += self.nugget * np.eye(X.shape[0])

        self.inv_r_ = np.linalg.inv(R)
        ones = np.ones((X.shape[0], 1))
        denominator = float((ones.T @ self.inv_r_ @ ones)[0, 0])
        beta_num = float((ones.T @ self.inv_r_ @ y.reshape(-1, 1))[0, 0])
        self.beta_ = beta_num / denominator
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        if self.inv_r_ is None or self.beta_ is None:
            raise RuntimeError("The kriging surrogate has not been fitted correctly.")

        D = self._distance_matrix(X, self._x_train_norm_)
        r = self._correlation_matrix(D)
        y_centered = self.y_train_ - self.beta_
        pred = self.beta_ + (r @ (self.inv_r_ @ y_centered))
        return np.asarray(pred, dtype=float)


class KrigingSurrogate(Kriging):
    """Backward-compatible alias matching the public naming used in the new package."""

from __future__ import annotations

import numpy as np

from ..base import BaseSurrogateModel


class RBF(BaseSurrogateModel):
    """Radial basis function surrogate with the legacy kernel family preserved."""

    def __init__(
        self,
        *,
        name: str = "rbf_surrogate",
        normalize: bool = True,
        gamma: float = 1.0,
        regularization: float = 1e-8,
        kernel: str = "gaussian",
    ):
        super().__init__(name=name, normalize=normalize)
        self.gamma = gamma
        self.regularization = regularization
        self.kernel = kernel
        self.weights_ = None

    def _distance_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, None, :] - X2[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _kernel_matrix(self, distances: np.ndarray) -> np.ndarray:
        key = str(self.kernel).lower()
        if key == "gaussian":
            return np.exp(-(self.gamma * distances) ** 2)
        if key == "multiquadratic":
            return np.sqrt(distances**2 + self.gamma**2)
        if key == "inverse_multiquadratic":
            return 1.0 / np.sqrt(distances**2 + self.gamma**2)
        if key == "absolute":
            return np.abs(distances)
        if key == "linear":
            return distances
        if key == "cubic":
            return distances**3
        if key == "thin_plate":
            safe = distances.copy()
            zero_mask = safe == 0.0
            safe[zero_mask] = 1.0
            out = safe**2 * np.log(np.abs(safe))
            out[zero_mask] = 0.0
            return out
        raise ValueError(
            "Unsupported kernel. Use one of: "
            "multiquadratic, inverse_multiquadratic, absolute, gaussian, linear, cubic, thin_plate."
        )

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        D = self._distance_matrix(X, X)
        K = self._kernel_matrix(D)
        K += self.regularization * np.eye(X.shape[0])
        self.weights_ = np.linalg.solve(K, y)
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        D = self._distance_matrix(X, self._x_train_norm_)
        K = self._kernel_matrix(D)
        return K @ self.weights_


class RBFSurrogate(RBF):
    """Backward-compatible alias matching the public naming used in the new package."""

    pass

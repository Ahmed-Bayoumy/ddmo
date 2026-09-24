from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import LinAlgError, inv, solve

from .._utils import (
    pairwise_distances,
    polynomial_features,
    polynomial_gradients,
    polynomial_terms,
)
from ..base import BaseSurrogateModel

# Minimum polynomial-tail degree for which the interpolation system is
# guaranteed to be uniquely solvable (-1 means positive definite, no tail needed).
_MIN_DEGREE = {
    "gaussian": -1,
    "inverse_multiquadric": -1,
    "multiquadric": 0,
    "linear": 0,
    "cubic": 1,
    "thin_plate": 1,
}
_SHAPE_KERNELS = {"gaussian", "inverse_multiquadric", "multiquadric"}
_DEPRECATED_NAMES = {
    "multiquadratic": "multiquadric",
    "inverse_multiquadratic": "inverse_multiquadric",
    "absolute": "linear",
}
_GAMMA_GRID = np.logspace(-2, 2, 41)
_MAX_CONDITION = 1e10


class RBF(BaseSurrogateModel):
    """Radial basis function interpolant with a polynomial tail.

    The model is ``s(x) = sum_i w_i * phi(gamma * ||x - x_i||) + q(x)`` where
    ``q`` is a polynomial of total degree ``degree`` and the weights satisfy
    the usual moment conditions ``P^T w = 0``. ``gamma`` is a shape parameter
    with the same meaning for every shape kernel (larger = more localized);
    it has no effect on ``linear``, ``cubic`` and ``thin_plate``. Pass
    ``gamma="auto"`` to pick it by Rippa's leave-one-out cross-validation.

    ``degree=None`` uses 1 for ``cubic``/``thin_plate`` and 0 otherwise;
    ``degree=-1`` (no tail) is only allowed for positive definite kernels.
    ``regularization`` is added to the kernel diagonal (a smoothing term).
    """

    def __init__(
        self,
        *,
        name: str = "rbf_surrogate",
        normalize: bool = True,
        gamma: float | str = 1.0,
        regularization: float = 1e-10,
        kernel: str = "gaussian",
        degree: int | None = None,
    ):
        super().__init__(name=name, normalize=normalize)
        self.gamma = gamma
        self.regularization = regularization
        self.kernel = kernel
        self.degree = degree
        self.kernel_ = None
        self.gamma_ = None
        self.degree_ = None
        self.terms_ = None
        self.weights_ = None
        self.poly_coefficients_ = None

    def _resolve_kernel(self) -> str:
        key = str(self.kernel).lower()
        if key in _DEPRECATED_NAMES:
            new = _DEPRECATED_NAMES[key]
            warnings.warn(
                f"RBF kernel '{key}' is deprecated; use '{new}'.", DeprecationWarning, stacklevel=4
            )
            key = new
        if key not in _MIN_DEGREE:
            raise ValueError(f"Unsupported kernel '{self.kernel}'. Use one of: {', '.join(_MIN_DEGREE)}.")
        return key

    def _phi(self, r: np.ndarray, gamma: float) -> np.ndarray:
        k = self.kernel_
        if k == "gaussian":
            return np.exp(-((gamma * r) ** 2))
        if k == "multiquadric":
            return np.sqrt(1.0 + (gamma * r) ** 2)
        if k == "inverse_multiquadric":
            return 1.0 / np.sqrt(1.0 + (gamma * r) ** 2)
        if k == "linear":
            return r
        if k == "cubic":
            return r**3
        # thin_plate: r^2 log r, with the removable singularity at r = 0
        out = np.zeros_like(r)
        pos = r > 0
        out[pos] = r[pos] ** 2 * np.log(r[pos])
        return out

    def _dphi_over_r(self, r: np.ndarray, gamma: float) -> np.ndarray:
        """``phi'(r) / r``, so that grad phi(||x - c||) = (phi'(r) / r) * (x - c).

        ``linear`` and ``thin_plate`` are not differentiable at their centres; the
        (symmetric) subgradient 0 is used there.
        """
        k = self.kernel_
        g2 = gamma**2
        if k == "gaussian":
            return -2.0 * g2 * np.exp(-g2 * r**2)
        if k == "multiquadric":
            return g2 / np.sqrt(1.0 + g2 * r**2)
        if k == "inverse_multiquadric":
            return -g2 * (1.0 + g2 * r**2) ** -1.5
        if k == "cubic":
            return 3.0 * r
        out = np.zeros_like(r)
        pos = r > 0
        out[pos] = 1.0 / r[pos] if k == "linear" else 2.0 * np.log(r[pos]) + 1.0
        return out

    def _system(self, D: np.ndarray, P: np.ndarray, gamma: float) -> np.ndarray:
        n, m = P.shape
        A = np.zeros((n + m, n + m))
        A[:n, :n] = self._phi(D, gamma) + self.regularization * np.eye(n)
        A[:n, n:] = P
        A[n:, :n] = P.T
        return A

    def _loo_error(self, D: np.ndarray, P: np.ndarray, y: np.ndarray, gamma: float) -> float:
        """Rippa's closed-form leave-one-out error for the augmented system."""
        n = D.shape[0]
        A = self._system(D, P, gamma)
        try:
            A_inv = inv(A)
        except LinAlgError:
            return np.inf
        # Flat kernels often minimize the LOO error on paper but are numerically unusable.
        if np.linalg.norm(A, 1) * np.linalg.norm(A_inv, 1) > _MAX_CONDITION:
            return np.inf
        coef = A_inv[:, :n] @ y
        errors = coef[:n] / np.diag(A_inv)[:n]
        return float(np.mean(errors**2)) if np.all(np.isfinite(errors)) else np.inf

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        self.kernel_ = self._resolve_kernel()
        min_degree = _MIN_DEGREE[self.kernel_]
        degree = max(min_degree, 0) if self.degree is None else int(self.degree)
        if degree < min_degree:
            raise ValueError(
                f"Kernel '{self.kernel_}' requires a polynomial tail of degree >= {min_degree}."
            )
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative.")

        self.degree_ = degree
        self.terms_ = polynomial_terms(X.shape[1], degree) if degree >= 0 else []
        P = polynomial_features(X, self.terms_)
        if X.shape[0] < P.shape[1]:
            raise ValueError(
                f"Need at least {P.shape[1]} samples for a degree-{degree} tail in {X.shape[1]} dimensions."
            )

        D = pairwise_distances(X, X)
        if isinstance(self.gamma, str):
            if self.gamma != "auto":
                raise ValueError("gamma must be a positive number or 'auto'.")
            if self.kernel_ in _SHAPE_KERNELS:
                errors = np.array([self._loo_error(D, P, y, g) for g in _GAMMA_GRID])
                best = int(np.argmin(errors))
                self.gamma_ = float(_GAMMA_GRID[best]) if np.isfinite(errors[best]) else 1.0
            else:
                self.gamma_ = 1.0
        else:
            if self.gamma <= 0:
                raise ValueError("gamma must be positive.")
            self.gamma_ = float(self.gamma)

        rhs = np.concatenate([y, np.zeros(P.shape[1])])
        try:
            coef = solve(self._system(D, P, self.gamma_), rhs)
        except LinAlgError as exc:
            raise LinAlgError(
                "RBF system is singular; check for duplicate points or points that are not "
                "unisolvent for the polynomial tail (e.g. collinear points with degree 1)."
            ) from exc
        n = X.shape[0]
        self.weights_ = coef[:n]
        self.poly_coefficients_ = coef[n:]
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        D = pairwise_distances(X, self._x_train_norm_)
        tail = polynomial_features(X, self.terms_) @ self.poly_coefficients_
        return self._phi(D, self.gamma_) @ self.weights_ + tail

    def _gradient_impl(self, X: np.ndarray) -> np.ndarray:
        D = pairwise_distances(X, self._x_train_norm_)
        coef = self._dphi_over_r(D, self.gamma_) * self.weights_  # (n, m)
        radial = coef.sum(axis=1)[:, None] * X - coef @ self._x_train_norm_
        tail = np.einsum(
            "nmd,m->nd", polynomial_gradients(X, self.terms_), self.poly_coefficients_
        )
        return radial + tail


RBFSurrogate = RBF

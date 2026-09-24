from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize

from ..base import BaseSurrogateModel


class Kriging(BaseSurrogateModel):
    """Ordinary kriging with an anisotropic powered-exponential correlation.

    The correlation between two points is::

        R(x, x') = exp(-sum_k theta_k * |x_k - x'_k| ** p)

    with ``0 < p <= 2`` (``p = 2`` is the Gaussian kernel). When ``theta`` is
    ``None`` the per-dimension ``theta_k`` are estimated by maximizing the
    concentrated log-likelihood (multi-start L-BFGS-B in log10 space within
    ``theta_bounds``). A scalar or per-dimension array fixes ``theta``.

    ``predict(X, return_std=True)`` also returns the kriging standard error
    (square root of the ordinary-kriging mean squared error), which is zero at
    the training points and grows away from them.
    """

    def __init__(
        self,
        *,
        name: str = "kriging_surrogate",
        normalize: bool = True,
        theta: float | np.ndarray | None = None,
        p: float = 2.0,
        nugget: float = 1e-10,
        theta_bounds: tuple[float, float] = (1e-3, 1e2),
        n_restarts: int = 5,
        random_state: int | None = 0,
    ):
        super().__init__(name=name, normalize=normalize)
        self.theta = theta
        self.p = p
        self.nugget = nugget
        self.theta_bounds = theta_bounds
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.theta_ = None
        self.nugget_ = None
        self.beta_ = None
        self.sigma2_ = None
        self._chol = None
        self._alpha = None
        self._rinv_ones = None
        self._ones_rinv_ones = None

    def _correlation(self, X1: np.ndarray, X2: np.ndarray, theta: np.ndarray) -> np.ndarray:
        weighted = np.zeros((X1.shape[0], X2.shape[0]))
        for k in range(X1.shape[1]):
            weighted += theta[k] * np.abs(X1[:, k, None] - X2[None, :, k]) ** self.p
        return np.exp(-weighted)

    def _factorize(self, R: np.ndarray) -> tuple[tuple[np.ndarray, bool], float]:
        """Cholesky of ``R + nugget*I``, growing the nugget until R is numerically SPD."""
        n = R.shape[0]
        nugget = self.nugget
        for _ in range(10):
            try:
                return cho_factor(R + nugget * np.eye(n), lower=True), nugget
            except LinAlgError:
                nugget = max(nugget * 10.0, 1e-12)
        raise LinAlgError("Correlation matrix is not positive definite even with a large nugget.")

    def _concentrated_fit(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> dict[str, Any]:
        n = X.shape[0]
        chol, nugget = self._factorize(self._correlation(X, X, theta))
        ones = np.ones(n)
        rinv_ones = cho_solve(chol, ones)
        ones_rinv_ones = float(ones @ rinv_ones)
        beta = float(rinv_ones @ y) / ones_rinv_ones
        resid = y - beta
        alpha = cho_solve(chol, resid)
        sigma2 = max(float(resid @ alpha) / n, np.finfo(float).tiny)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
        return {
            "chol": chol,
            "nugget": nugget,
            "beta": beta,
            "alpha": alpha,
            "sigma2": sigma2,
            "rinv_ones": rinv_ones,
            "ones_rinv_ones": ones_rinv_ones,
            "neg_log_likelihood": n * np.log(sigma2) + log_det,
        }

    def _estimate_theta(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        lo, hi = np.log10(self.theta_bounds[0]), np.log10(self.theta_bounds[1])

        def objective(log_theta: np.ndarray) -> float:
            try:
                return self._concentrated_fit(X, y, 10.0**log_theta)["neg_log_likelihood"]
            except LinAlgError:
                return np.inf

        rng = np.random.default_rng(self.random_state)
        starts = [np.full(d, 0.5 * (lo + hi))]
        starts += [rng.uniform(lo, hi, size=d) for _ in range(max(self.n_restarts, 0))]

        best_x, best_f = starts[0], np.inf
        for x0 in starts:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=[(lo, hi)] * d)
            if np.isfinite(res.fun) and res.fun < best_f:
                best_x, best_f = res.x, res.fun
        return 10.0**best_x

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        if not 0.0 < self.p <= 2.0:
            raise ValueError("p must satisfy 0 < p <= 2 for a valid correlation function.")
        if self.nugget < 0:
            raise ValueError("nugget must be non-negative.")

        d = X.shape[1]
        if self.theta is None:
            theta = self._estimate_theta(X, y)
        else:
            theta = np.broadcast_to(np.asarray(self.theta, dtype=float), (d,)).copy()
            if np.any(theta <= 0):
                raise ValueError("theta must be strictly positive.")

        fit = self._concentrated_fit(X, y, theta)
        self.theta_ = theta
        self.nugget_ = fit["nugget"]
        self.beta_ = fit["beta"]
        self.sigma2_ = fit["sigma2"]
        self._chol = fit["chol"]
        self._alpha = fit["alpha"]
        self._rinv_ones = fit["rinv_ones"]
        self._ones_rinv_ones = fit["ones_rinv_ones"]
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        r = self._correlation(X, self._x_train_norm_, self.theta_)
        return self.beta_ + r @ self._alpha

    def _gradient_impl(self, X: np.ndarray) -> np.ndarray:
        # For p <= 1 the correlation has a kink where x_k equals a training x_k; 0 is used there.
        Xt = self._x_train_norm_
        weighted_r = self._correlation(X, Xt, self.theta_) * self._alpha  # (n, m)
        grad = np.zeros_like(X)
        for k in range(X.shape[1]):
            diff = X[:, k, None] - Xt[None, :, k]
            dcorr = np.zeros_like(diff)
            nz = diff != 0
            dcorr[nz] = self.p * np.abs(diff[nz]) ** (self.p - 1) * np.sign(diff[nz])
            grad[:, k] = -self.theta_[k] * np.sum(weighted_r * dcorr, axis=1)
        return grad

    def _std_impl(self, X: np.ndarray) -> np.ndarray:
        r = self._correlation(X, self._x_train_norm_, self.theta_)
        rinv_r = cho_solve(self._chol, r.T)
        u = 1.0 - r @ self._rinv_ones
        mse = self.sigma2_ * (
            1.0 - np.sum(r.T * rinv_r, axis=0) + u**2 / self._ones_rinv_ones
        )
        return np.sqrt(np.clip(mse, 0.0, None))

    def predict(self, X: Any, return_std: bool = False):
        self._check_fitted()
        Xn = self._prepare_X_for_prediction(X)
        mean = np.asarray(self._predict_impl(Xn), dtype=float)
        if not return_std:
            return mean
        return mean, self._std_impl(Xn)


KrigingSurrogate = Kriging

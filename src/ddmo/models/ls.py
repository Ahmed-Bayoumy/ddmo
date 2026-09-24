from __future__ import annotations

import numpy as np

from .._utils import polynomial_features, polynomial_gradients, polynomial_terms
from ..base import BaseSurrogateModel


def _coordinate_descent(
    Z: np.ndarray,
    y: np.ndarray,
    lasso: float,
    ridge: float,
    w0: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """Minimize ``1/2 ||y - Z w||^2 + ridge/2 ||w||^2 + lasso ||w||_1`` (centered Z and y)."""
    w = w0.copy()
    resid = y - Z @ w
    col_sq = np.einsum("ij,ij->j", Z, Z)
    for it in range(1, max_iter + 1):
        max_change, max_w = 0.0, 0.0
        for j in range(Z.shape[1]):
            if col_sq[j] == 0.0:
                continue
            old = w[j]
            rho = Z[:, j] @ resid + col_sq[j] * old
            w[j] = np.sign(rho) * max(abs(rho) - lasso, 0.0) / (col_sq[j] + ridge)
            if w[j] != old:
                resid -= Z[:, j] * (w[j] - old)
                max_change = max(max_change, abs(w[j] - old))
            max_w = max(max_w, abs(w[j]))
        if max_change <= tol * max(max_w, 1.0):
            return w, it
    return w, max_iter


class LS(BaseSurrogateModel):
    """Polynomial least-squares (response surface) surrogate with optional ridge/lasso.

    The basis is the full polynomial of total degree ``degree``, including
    interaction terms (e.g. degree 2 in two variables gives
    ``1, x0, x1, x0^2, x0*x1, x1^2``). Non-constant basis columns are scaled
    to unit standard deviation so the penalties treat all terms equally, and the
    intercept is never penalized. The objective is::

        1/2 ||y - A w||^2 + ridge/2 ||w||^2 + lasso ||w||_1

    * ``lasso == 0``: solved with an orthogonal-factorization least-squares solver
      (minimum-norm solution when underdetermined and ``ridge == 0``).
    * ``lasso > 0``: solved by coordinate descent. The L1 penalty sets the
      coefficients of uninformative terms exactly to zero and, among strongly
      correlated terms, tends to keep one (for exact duplicates the split is
      arbitrary; remove them first with ``ddmo.CollinearityFilter``). Combined
      with ``ridge`` this is the elastic net.
    * ``lasso="auto"``: chooses the lasso strength from a log-spaced path of
      ``n_lassos`` values (from the smallest value that zeroes every coefficient
      down to 1e-3 times that) by ``n_folds``-fold cross-validation.

    Penalties are summed over samples, so the same value is relatively weaker
    with more data. ``support_`` marks the input features used by at least one
    nonzero term; ``selected_features_`` lists their indices.
    """

    def __init__(
        self,
        *,
        name: str = "ls_surrogate",
        normalize: bool = True,
        degree: int = 1,
        ridge: float = 0.0,
        lasso: float | str = 0.0,
        n_lassos: int = 30,
        n_folds: int = 5,
        max_iter: int = 10_000,
        tol: float = 1e-8,
        random_state: int | None = 0,
    ):
        super().__init__(name=name, normalize=normalize)
        self.degree = degree
        self.ridge = ridge
        self.lasso = lasso
        self.n_lassos = n_lassos
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coefficients_ = None
        self.terms_ = None
        self.lasso_ = None
        self.lasso_path_ = None
        self.cv_mse_ = None
        self.n_iter_ = None
        self.support_ = None
        self.selected_features_ = None
        self._col_scale = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        return polynomial_features(X, self.terms_) / self._col_scale

    @staticmethod
    def _scaled_basis(X: np.ndarray, terms: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
        raw = polynomial_features(X, terms)
        scale = raw.std(axis=0)
        scale[scale == 0.0] = 1.0
        scale[0] = 1.0
        return raw / scale, scale

    def _solve_lasso(self, A: np.ndarray, y: np.ndarray, lasso: float, ridge: float, w0=None):
        """Elastic-net fit with an unpenalized intercept; returns coefficients for ``A``."""
        col_mean = A[:, 1:].mean(axis=0)
        Z = A[:, 1:] - col_mean
        y_mean = y.mean()
        w0 = np.zeros(Z.shape[1]) if w0 is None else w0[1:]
        w, n_iter = _coordinate_descent(Z, y - y_mean, lasso, ridge, w0, self.max_iter, self.tol)
        return np.concatenate([[y_mean - col_mean @ w], w]), n_iter

    def _select_lasso(self, X: np.ndarray, y: np.ndarray) -> float:
        n = X.shape[0]
        n_folds = min(self.n_folds, n)
        if n_folds < 2:
            raise ValueError("lasso='auto' needs at least 2 samples and n_folds >= 2.")

        A, _ = self._scaled_basis(X, self.terms_)
        Z = A[:, 1:] - A[:, 1:].mean(axis=0)
        lasso_max = float(np.abs(Z.T @ (y - y.mean())).max()) if Z.size else 0.0
        if lasso_max == 0.0:
            self.lasso_path_, self.cv_mse_ = np.array([0.0]), None
            return 0.0
        path = np.logspace(np.log10(lasso_max), np.log10(lasso_max * 1e-3), self.n_lassos)

        folds = np.array_split(np.random.default_rng(self.random_state).permutation(n), n_folds)
        mse = np.zeros(path.size)
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            A_train, scale = self._scaled_basis(X[train], self.terms_)
            A_test = polynomial_features(X[test], self.terms_) / scale
            # Penalties are sums over samples; rescale so a fold matches the full data.
            ratio = train.size / n
            coef = None
            for i, lam in enumerate(path):  # warm start along the decreasing path
                coef, _ = self._solve_lasso(
                    A_train, y[train], lam * ratio, self.ridge * ratio, coef
                )
                mse[i] += np.sum((A_test @ coef - y[test]) ** 2)
        self.lasso_path_ = path
        self.cv_mse_ = mse / n
        return float(path[int(np.argmin(self.cv_mse_))])

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        if int(self.degree) != self.degree or self.degree < 0:
            raise ValueError("degree must be a non-negative integer.")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative.")

        self.terms_ = polynomial_terms(X.shape[1], int(self.degree))
        if isinstance(self.lasso, str):
            if self.lasso != "auto":
                raise ValueError("lasso must be a non-negative number or 'auto'.")
            lasso = self._select_lasso(X, y)
        else:
            lasso = float(self.lasso)
            if lasso < 0:
                raise ValueError("lasso must be non-negative.")

        A, self._col_scale = self._scaled_basis(X, self.terms_)
        if lasso > 0:
            self.coefficients_, self.n_iter_ = self._solve_lasso(A, y, lasso, self.ridge)
        else:
            b = y
            if self.ridge > 0:
                penalty = np.sqrt(self.ridge) * np.eye(A.shape[1])[1:]  # intercept unpenalized
                A = np.vstack([A, penalty])
                b = np.concatenate([y, np.zeros(penalty.shape[0])])
            self.coefficients_, *_ = np.linalg.lstsq(A, b, rcond=None)
            self.n_iter_ = None
        self.lasso_ = lasso

        support = np.zeros(X.shape[1], dtype=bool)
        for term, c in zip(self.terms_[1:], self.coefficients_[1:]):
            if c != 0.0:
                support[list(term)] = True
        self.support_ = support
        self.selected_features_ = np.flatnonzero(support)
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.coefficients_

    def _gradient_impl(self, X: np.ndarray) -> np.ndarray:
        dP = polynomial_gradients(X, self.terms_) / self._col_scale[None, :, None]
        return np.einsum("nmd,m->nd", dP, self.coefficients_)


LinearSurrogate = LS

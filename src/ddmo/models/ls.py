from __future__ import annotations

import numpy as np

from .._utils import polynomial_features, polynomial_terms
from ..base import BaseSurrogateModel


class LS(BaseSurrogateModel):
    """Polynomial least-squares (response surface) surrogate.

    The basis is the full polynomial of total degree ``degree``, including
    interaction terms (e.g. degree 2 in two variables gives
    ``1, x0, x1, x0^2, x0*x1, x1^2``). Non-constant basis columns are scaled
    to unit standard deviation so that ``ridge`` penalizes all terms equally;
    the intercept is never penalized. The problem is solved with an
    orthogonal-factorization least-squares solver rather than the normal
    equations, and the minimum-norm solution is returned when the system is
    underdetermined and ``ridge == 0``.
    """

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
        self.terms_ = None
        self._col_scale = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        return polynomial_features(X, self.terms_) / self._col_scale

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        if int(self.degree) != self.degree or self.degree < 0:
            raise ValueError("degree must be a non-negative integer.")
        if self.ridge < 0:
            raise ValueError("ridge must be non-negative.")

        self.terms_ = polynomial_terms(X.shape[1], int(self.degree))
        raw = polynomial_features(X, self.terms_)
        scale = raw.std(axis=0)
        scale[scale == 0.0] = 1.0
        scale[0] = 1.0
        self._col_scale = scale

        A = raw / scale
        b = y
        if self.ridge > 0:
            penalty = np.sqrt(self.ridge) * np.eye(A.shape[1])
            penalty = penalty[1:]  # leave the intercept unpenalized
            A = np.vstack([A, penalty])
            b = np.concatenate([y, np.zeros(penalty.shape[0])])

        self.coefficients_, *_ = np.linalg.lstsq(A, b, rcond=None)
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.coefficients_


LinearSurrogate = LS

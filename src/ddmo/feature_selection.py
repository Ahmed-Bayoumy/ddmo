"""Utilities for detecting and dropping collinear or uninformative input features.

All functions take the raw input matrix ``X`` of shape ``(n_samples, n_features)``
and return indices of the features to keep, in their original order. Constant
columns are always dropped.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .datasets import validate_X, validate_xy


def _non_constant(X: np.ndarray) -> np.ndarray:
    return np.flatnonzero(X.std(axis=0) > 0.0)


def variance_inflation_factors(X: Any) -> np.ndarray:
    """Variance inflation factor of each column: ``1 / (1 - R^2_j)``.

    ``R^2_j`` is from regressing column ``j`` (with intercept) on all other columns.
    A value of 1 means no collinearity; values above 5-10 are usually considered
    problematic; exact linear dependence gives ``inf``. Constant columns get ``nan``.
    """
    X = validate_X(X)
    n, d = X.shape
    vif = np.full(d, np.nan)
    for j in _non_constant(X):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        ss_res = float(np.sum((X[:, j] - A @ coef) ** 2))
        ss_tot = float(np.sum((X[:, j] - X[:, j].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        vif[j] = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
    return vif


def vif_filter(X: Any, threshold: float = 10.0) -> np.ndarray:
    """Repeatedly drop the feature with the largest VIF until all VIFs are ``<= threshold``.

    Ties are broken by dropping the later column, so earlier columns are preferred.
    """
    X = validate_X(X)
    if threshold < 1.0:
        raise ValueError("threshold must be >= 1 (the VIF of an uncorrelated feature).")
    keep = list(_non_constant(X))
    while len(keep) > 1:
        vif = variance_inflation_factors(X[:, keep])
        worst = len(vif) - 1 - int(np.argmax(vif[::-1]))
        if vif[worst] <= threshold:
            break
        keep.pop(worst)
    return np.asarray(keep, dtype=int)


def correlation_filter(X: Any, threshold: float = 0.95) -> np.ndarray:
    """Keep a feature only if its absolute Pearson correlation with every feature
    already kept is ``<= threshold``. Features are considered in column order.

    This catches pairwise collinearity only; use :func:`vif_filter` to also catch a
    feature that is a combination of several others.
    """
    X = validate_X(X)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")
    candidates = _non_constant(X)
    if candidates.size == 0:
        return candidates
    corr = np.abs(np.atleast_2d(np.corrcoef(X[:, candidates], rowvar=False)))
    kept: list[int] = []
    for i in range(candidates.size):
        if all(corr[i, k] <= threshold for k in kept):
            kept.append(i)
    return candidates[kept]


def lasso_select(X: Any, y: Any, lasso: float | str = "auto", **ls_params: Any) -> np.ndarray:
    """Indices of the features kept by a lasso-penalized :class:`~ddmo.models.LS` fit.

    Other keyword arguments (``degree``, ``ridge``, ``n_folds``, ...) are passed to ``LS``.
    With ``degree > 1`` a feature is kept if it appears in any nonzero term.
    """
    from .models.ls import LS

    if lasso == 0:
        raise ValueError("lasso must be positive (or 'auto') for feature selection.")
    X, y = validate_xy(X, y)
    return LS(lasso=lasso, **ls_params).fit(X, y).selected_features_


class CollinearityFilter:
    """Reusable filter that learns which input columns to keep and applies it to new data.

    ``method`` is ``"vif"`` (default threshold 10) or ``"correlation"`` (default
    threshold 0.95). After ``fit``: ``support_`` is a boolean mask over the input
    columns, ``selected_features_`` the kept indices and ``dropped_features_`` the rest.

    Example::

        filt = CollinearityFilter(method="vif").fit(X)
        model = Kriging().fit(filt.transform(X), y)
        model.predict(filt.transform(X_new))
    """

    def __init__(self, method: str = "vif", threshold: float | None = None):
        self.method = method
        self.threshold = threshold
        self.n_features_in_ = None
        self.support_ = None
        self.selected_features_ = None
        self.dropped_features_ = None

    def fit(self, X: Any, y: Any = None):
        X = validate_X(X)
        if self.method == "vif":
            keep = vif_filter(X, 10.0 if self.threshold is None else self.threshold)
        elif self.method == "correlation":
            keep = correlation_filter(X, 0.95 if self.threshold is None else self.threshold)
        else:
            raise ValueError("method must be 'vif' or 'correlation'.")
        self.n_features_in_ = X.shape[1]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[keep] = True
        self.selected_features_ = np.flatnonzero(self.support_)
        self.dropped_features_ = np.flatnonzero(~self.support_)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.support_ is None:
            raise RuntimeError("CollinearityFilter has not been fitted yet.")
        X = validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but the filter was fitted on {self.n_features_in_}."
            )
        return X[:, self.support_]

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

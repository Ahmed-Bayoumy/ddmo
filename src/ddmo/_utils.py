from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from scipy.spatial.distance import cdist


def pairwise_distances(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of ``X1`` and ``X2``."""
    return cdist(X1, X2, metric="euclidean")


def polynomial_terms(n_features: int, degree: int) -> list[tuple[int, ...]]:
    """Exponent-free description of a full polynomial basis.

    Each term is a tuple of feature indices whose product forms the monomial,
    e.g. ``()`` is the constant, ``(0,)`` is ``x0`` and ``(0, 1)`` is ``x0*x1``.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    terms: list[tuple[int, ...]] = []
    for d in range(degree + 1):
        terms.extend(combinations_with_replacement(range(n_features), d))
    return terms


def polynomial_features(X: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    """Evaluate the monomials described by ``terms`` at the rows of ``X``."""
    out = np.ones((X.shape[0], len(terms)), dtype=float)
    for j, term in enumerate(terms):
        for i in term:
            out[:, j] *= X[:, i]
    return out


def polynomial_gradients(X: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    """Derivatives of the monomials in ``terms``, shape ``(n_samples, n_terms, n_features)``."""
    out = np.zeros((X.shape[0], len(terms), X.shape[1]), dtype=float)
    for j, term in enumerate(terms):
        for k in set(term):
            rest = list(term)
            rest.remove(k)
            deriv = np.full(X.shape[0], float(term.count(k)))
            for i in rest:
                deriv *= X[:, i]
            out[:, j, k] = deriv
    return out

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from ..base import BaseSurrogateModel


class WeightedEnsemble(BaseSurrogateModel):
    """Weighted average of several surrogate models.

    The weights are global (not input-dependent), so this is an ensemble
    rather than a gated mixture of experts. ``weights`` may be:

    * ``None`` or ``"uniform"``: equal weights;
    * ``"cv"``: weights proportional to ``1 / MSE`` of each expert under
      ``n_folds``-fold cross-validation (computed on copies of the experts);
    * a sequence of non-negative numbers, one per expert.

    Each expert handles its own input normalization, so the ensemble passes
    raw inputs through unchanged. The fitted, normalized weights are stored
    in ``weights_``.
    """

    def __init__(
        self,
        *,
        experts: Sequence[BaseSurrogateModel] | None = None,
        weights: Sequence[float] | str | None = None,
        n_folds: int = 5,
        random_state: int | None = 0,
        name: str = "ensemble",
    ):
        super().__init__(name=name, normalize=False)
        self.experts = list(experts or [])
        self.weights = weights
        self.n_folds = n_folds
        self.random_state = random_state
        self.weights_ = None
        self.cv_mse_ = None

    def _cv_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        n_folds = min(self.n_folds, n)
        if n_folds < 2:
            raise ValueError("Cross-validated weights need at least 2 samples and n_folds >= 2.")
        folds = np.array_split(np.random.default_rng(self.random_state).permutation(n), n_folds)

        mse = np.zeros(len(self.experts))
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            for j, expert in enumerate(self.experts):
                model = copy.deepcopy(expert).fit(X[train], y[train])
                mse[j] += np.sum((model.predict(X[test]) - y[test]) ** 2)
        mse /= n
        self.cv_mse_ = mse
        return 1.0 / np.maximum(mse, np.finfo(float).tiny)

    def _fit_impl(self, X: np.ndarray, y: np.ndarray):
        if not self.experts:
            raise ValueError("At least one expert model must be provided.")

        if self.weights is None or (isinstance(self.weights, str) and self.weights == "uniform"):
            weights = np.ones(len(self.experts))
        elif isinstance(self.weights, str):
            if self.weights != "cv":
                raise ValueError("weights must be a sequence, None, 'uniform' or 'cv'.")
            weights = self._cv_weights(X, y)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != (len(self.experts),):
                raise ValueError(
                    f"Got {weights.size} weights for {len(self.experts)} experts."
                )
            if np.any(weights < 0) or not np.all(np.isfinite(weights)):
                raise ValueError("weights must be finite and non-negative.")
        if weights.sum() <= 0:
            raise ValueError("weights must not all be zero.")
        self.weights_ = weights / weights.sum()

        for expert in self.experts:
            expert.fit(X, y)
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        predictions = np.column_stack([expert.predict(X) for expert in self.experts])
        return predictions @ self.weights_


MixtureOfExperts = WeightedEnsemble
MOE = WeightedEnsemble

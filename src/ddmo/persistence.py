"""Save fitted surrogate models to disk and load them back without retraining.

Models are stored with :mod:`pickle` (the approach scikit-learn recommends for
persisting estimators), wrapped in a small envelope that records the ddmo
version, the input/output column names and any user metadata.

.. warning::
    Loading a pickle file can execute arbitrary code. Only load model files
    that you created yourself or that come from a source you trust.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import IO, Any, Union

import numpy as np

from .base import BaseSurrogateModel

FORMAT_NAME = "ddmo-model"
FORMAT_VERSION = 1

PathOrFile = Union[str, "os.PathLike[str]", IO[bytes]]


@dataclass
class ModelBundle:
    """A fitted model together with the information needed to reuse it."""

    model: BaseSurrogateModel
    feature_names: list[str] | None = None
    target_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ddmo_version: str | None = None
    created_at: str | None = None

    def _select_features(self, X: Any) -> Any:
        # Accept a DataFrame (or anything with named columns) and reorder it
        # to match the training columns; plain arrays are passed through.
        if self.feature_names and hasattr(X, "columns"):
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Input is missing feature columns: {missing}")
            return X[self.feature_names].to_numpy(dtype=float)
        return X

    def predict(self, X: Any) -> np.ndarray:
        return self.model.predict(self._select_features(X))

    def predict_gradient(self, X: Any) -> np.ndarray:
        return self.model.predict_gradient(self._select_features(X))


def save_model(
    model: BaseSurrogateModel,
    file: PathOrFile,
    *,
    feature_names: list[str] | None = None,
    target_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a fitted ``model`` to ``file`` (a path or a binary file object)."""
    from . import __version__

    if not getattr(model, "is_fitted_", False):
        raise ValueError("Only fitted models can be saved; call fit() first.")
    if feature_names is not None and len(feature_names) != model.n_features_in_:
        raise ValueError(
            f"Got {len(feature_names)} feature names but the model was trained on {model.n_features_in_} features."
        )

    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "ddmo_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_class": type(model).__name__,
        "model": model,
        "feature_names": list(feature_names) if feature_names is not None else None,
        "target_name": target_name,
        "metadata": dict(metadata or {}),
    }
    if hasattr(file, "write"):
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file: PathOrFile) -> ModelBundle:
    """Read a model written by :func:`save_model`. Only load files you trust."""
    from . import __version__

    if hasattr(file, "read"):
        payload = pickle.load(file)
    else:
        with open(file, "rb") as fh:
            payload = pickle.load(fh)

    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError("File is not a ddmo model file.")
    if payload.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError(
            f"Model file format v{payload['format_version']} is newer than this ddmo supports (v{FORMAT_VERSION})."
        )
    if payload.get("ddmo_version") != __version__:
        warnings.warn(
            f"Model was saved with ddmo {payload.get('ddmo_version')} and is being loaded with ddmo {__version__}; "
            "predictions may differ if the model code changed.",
            UserWarning,
            stacklevel=2,
        )

    return ModelBundle(
        model=payload["model"],
        feature_names=payload.get("feature_names"),
        target_name=payload.get("target_name"),
        metadata=payload.get("metadata", {}),
        ddmo_version=payload.get("ddmo_version"),
        created_at=payload.get("created_at"),
    )

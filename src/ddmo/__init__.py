"""DDMO: Data-Driven Models for Optimization."""

from . import feature_selection, metrics, persistence
from .base import BaseSurrogateModel
from .feature_selection import CollinearityFilter
from .models import (
    LS,
    MOE,
    RBF,
    Kriging,
    KrigingSurrogate,
    LinearSurrogate,
    MixtureOfExperts,
    RBFSurrogate,
    WeightedEnsemble,
)
from .persistence import ModelBundle, load_model, save_model

# Keep in sync with [project].version in pyproject.toml (checked by tests/test_version.py).
__version__ = "2609.0"
__all__ = [
    "LS",
    "MOE",
    "RBF",
    "BaseSurrogateModel",
    "CollinearityFilter",
    "Kriging",
    "KrigingSurrogate",
    "LinearSurrogate",
    "MixtureOfExperts",
    "ModelBundle",
    "RBFSurrogate",
    "WeightedEnsemble",
    "__version__",
    "feature_selection",
    "load_model",
    "metrics",
    "persistence",
    "save_model",
]

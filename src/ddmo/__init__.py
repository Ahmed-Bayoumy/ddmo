"""DDMO: Data-Driven Models for Optimization."""

from . import feature_selection, metrics
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

__version__ = "0.0.1"
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
    "RBFSurrogate",
    "WeightedEnsemble",
    "__version__",
    "feature_selection",
    "metrics",
]

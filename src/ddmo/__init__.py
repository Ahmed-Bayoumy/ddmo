"""DDMO: Data-Driven Models for Optimization."""

from .base import BaseSurrogateModel
from .models import (
    LS,
    MOE,
    RBF,
    Kriging,
    KrigingSurrogate,
    LinearSurrogate,
    MixtureOfExperts,
    RBFSurrogate,
)

__version__ = "0.0.1"
__all__ = [
    "LS",
    "MOE",
    "RBF",
    "BaseSurrogateModel",
    "Kriging",
    "KrigingSurrogate",
    "LinearSurrogate",
    "MixtureOfExperts",
    "RBFSurrogate",
    "__version__",
]
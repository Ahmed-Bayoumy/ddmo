"""DDMO: Data-Driven Models for Optimization."""

from .base import BaseSurrogateModel
from .models import Kriging, KrigingSurrogate, LS, LinearSurrogate, MOE, MixtureOfExperts, RBF, RBFSurrogate

__version__ = "0.0.1"
__all__ = [
    "BaseSurrogateModel",
    "LinearSurrogate",
    "LS",
    "RBFSurrogate",
    "RBF",
    "KrigingSurrogate",
    "Kriging",
    "MOE",
    "MixtureOfExperts",
    "__version__",
]
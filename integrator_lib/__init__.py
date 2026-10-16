"""Coordinate-independent time integrators."""

from .constant_mean_field import IMCMF1, IMCMF2
from .crank_nicolson import CrankNicolson, LinearSolveError
from .runge_kutta import RK4
from .splitting import Strang2

__all__ = [
    "IMCMF1",
    "IMCMF2",
    "CrankNicolson",
    "LinearSolveError",
    "RK4",
    "Strang2",
]
__version__ = "0.2.0"

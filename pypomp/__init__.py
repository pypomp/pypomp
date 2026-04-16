"""
pypomp: Modeling and inference using partially observed Markov process (POMP) models.
"""

from .core.parameters import PanelParameters, PompParameters
from .core.par_trans import ParTrans
from .core.pomp import Pomp
from .core.rw_sigma import RWSigma
from .panel.panel import PanelPomp

from .mcap import mcap

from . import random, models, benchmarks, types, maths


def _get_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pypomp")
    except PackageNotFoundError:
        return "unknown"


__version__ = _get_version()
del _get_version

__all__ = [
    "__version__",
    # Core
    "PanelParameters",
    "PanelPomp",
    "ParTrans",
    "Pomp",
    "PompParameters",
    "RWSigma",
    # Inference / Algorithms
    "mcap",
    # Submodules
    "benchmarks",
    "models",
    "random",
    "types",
    "maths",
]

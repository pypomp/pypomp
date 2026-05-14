"""
pypomp: Modeling and inference using partially observed Markov process (POMP) models.
"""

from .core.parameters import PanelParameters, PompParameters
from .core.par_trans import ParTrans
from .core.pomp import Pomp
from .core.rw_sigma import RWSigma
from .panel.panel import PanelPomp

from .mcap import mcap
from .proposals import mvn_diag_rw, mvn_rw, mvn_rw_adaptive, MVNRWAdaptive

from . import random, models, benchmarks, types, maths, functional


def _get_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pypomp")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"  # pragma: no cover


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
    "mvn_diag_rw",
    "mvn_rw",
    "mvn_rw_adaptive",
    "MVNRWAdaptive",
    # Submodules
    "benchmarks",
    "models",
    "random",
    "types",
    "maths",
    "functional",
]

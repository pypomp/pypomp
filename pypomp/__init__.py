"""
pypomp: Modeling and inference using partially observed Markov process (POMP) models.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pypomp")
except PackageNotFoundError:
    __version__ = "unknown"

from .core.parameters import PanelParameters, PompParameters
from .core.par_trans import ParTrans
from .core.pomp import Pomp
from .core.rw_sigma import RWSigma
from .panel.panel import PanelPomp

from .mcap import mcap
from .util import expit, logit, logmeanexp, logmeanexp_se

from . import random, models, benchmarks

from .types import (
    CovarDict,
    InitialTimeFloat,
    ObservationDict,
    ParamDict,
    RNGKey,
    StateDict,
    StepSizeFloat,
    TimeFloat,
)

__all__ = [
    "__version__",
    # Core
    "PanelParameters",
    "PanelPomp",
    "ParTrans",
    "Pomp",
    "PompParameters",
    "RWSigma",
    # Inference / Utils
    "expit",
    "logit",
    "logmeanexp",
    "logmeanexp_se",
    "mcap",
    # Types
    "CovarDict",
    "InitialTimeFloat",
    "ObservationDict",
    "ParamDict",
    "RNGKey",
    "StateDict",
    "StepSizeFloat",
    "TimeFloat",
    # Submodules
    "random",
    "models",
    "benchmarks",
]

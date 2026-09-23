"""
pypomp: JAX-accelerated modeling and inference for partially observed Markov process (POMP) models.

The top-level namespace exports the core modelling classes (:class:`Pomp`,
:class:`PanelPomp`), parameter containers (:class:`PompParameters`,
:class:`PanelParameters`), configuration helpers (:class:`ParTrans`,
:class:`RWSigma`, :class:`LearningRate`), optimizers (:class:`Adam`,
:class:`SGD`, etc.), and the :func:`mcap` inference utility along with
its result container (:class:`MCAPResult`).

Submodules
----------
random
    JAX-compatible GPU-optimized random variable samplers.
functional
    Pure-functional JAX implementations of pfilter, mif, train, simulate.
maths
    Numerical utilities (logmeanexp, logit, expit).
models
    Built-in example POMP models (SIR, Dacca, measles, etc.).
benchmarks
    Baseline statistical benchmarks (ARMA, negative binomial).
types
    Annotated type aliases used in user-defined model component functions.
"""

import sys as _sys

from . import benchmarks, functional, maths, models, random, types
from .core.learning_rate import LearningRate
from .core.model_mechanics import vectorized
from .core.optimizer import (
    BFGS,
    SGD,
    Adam,
    FullMatrixAdam,
    Newton,
    WeightedNewton,
)
from .core.par_trans import ParTrans
from .core.parameters import PanelParameters, PompParameters
from .core.pomp import Pomp
from .core.rw_sigma import RWSigma
from .mcap import MCAPResult, mcap
from .bake import ArchiveValue, archive_directory, bake, freeze, r_uniform, stew
from .panel.panel import PanelPomp
from .proposals import (
    MVNDiagRW,
    MVNRWAdaptive,
    MVNRWFull,
    Proposal,
)

_sys.modules[__name__ + ".recipes"] = _sys.modules[__name__ + ".bake"]
_sys.modules[__name__ + "._recipe_compat"] = _sys.modules[__name__ + "._bake_compat"]
del _sys


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
    "LearningRate",
    "vectorized",
    "SGD",
    "Adam",
    "FullMatrixAdam",
    "BFGS",
    "Newton",
    "WeightedNewton",
    # Inference / Algorithms
    "MCAPResult",
    "ArchiveValue",
    "archive_directory",
    "freeze",
    "r_uniform",
    "bake",
    "stew",
    "mcap",
    "Proposal",
    "MVNDiagRW",
    "MVNRWFull",
    "MVNRWAdaptive",
    # Submodules
    "benchmarks",
    "models",
    "random",
    "types",
    "maths",
    "functional",
]

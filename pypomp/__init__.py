"""
pypomp: JAX-accelerated modeling and inference for partially observed Markov process (POMP) models.

The top-level namespace exports the core modelling classes (:class:`Pomp`,
:class:`PanelPomp`), parameter containers (:class:`PompParameters`,
:class:`PanelParameters`), configuration helpers (:class:`ParTrans`,
:class:`RWSigma`, :class:`LearningRate`), optimizers (:class:`Adam`,
:class:`SGD`, etc.), and the :func:`mcap` inference utility.

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

from .core.parameters import PanelParameters, PompParameters
from .core.par_trans import ParTrans
from .core.pomp import Pomp
from .core.rw_sigma import RWSigma
from .core.learning_rate import LearningRate
from .core.optimizer import (
    SGD,
    Adam,
    FullMatrixAdam,
    BFGS,
    Newton,
    WeightedNewton,
)
from .panel.panel import PanelPomp

from .mcap import mcap
from .proposals import (
    MVNDiagRW,
    MVNRWFull,
    MVNRWAdaptive,
    mvn_diag_rw,
    mvn_rw,
    mvn_rw_adaptive,
)

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
    "LearningRate",
    "SGD",
    "Adam",
    "FullMatrixAdam",
    "BFGS",
    "Newton",
    "WeightedNewton",
    # Inference / Algorithms
    "mcap",
    "MVNDiagRW",
    "MVNRWFull",
    "MVNRWAdaptive",
    "mvn_diag_rw",
    "mvn_rw",
    "mvn_rw_adaptive",
    # Submodules
    "benchmarks",
    "models",
    "random",
    "types",
    "maths",
    "functional",
]

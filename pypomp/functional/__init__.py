"""
Pure-functional JAX implementations of the core POMP algorithms.

This submodule exposes the low-level, stateless versions of the particle
filter, iterated filter, MOP training, and simulation algorithms for users
who need to compose them within custom JAX loops or higher-order functions.

For the standard object-oriented interface, use the :class:`~pypomp.Pomp`
and :class:`~pypomp.PanelPomp` classes instead.
"""

from .abc import abc
from .dpop import dpop
from .mif import mif, panel_mif
from .mop import mop
from .pfilter import panel_pfilter, pfilter
from .pmcmc import pmcmc
from .simulate import simulate
from .structs import PanelPompStruct, PompStruct
from .train import panel_train, train
from .utils import align_params

__all__ = [
    "PanelPompStruct",
    "PompStruct",
    "abc",
    "align_params",
    "dpop",
    "mif",
    "mop",
    "panel_mif",
    "panel_pfilter",
    "panel_train",
    "pfilter",
    "pmcmc",
    "simulate",
    "train",
]

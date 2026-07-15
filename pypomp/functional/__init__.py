"""
Pure-functional JAX implementations of the core POMP algorithms.

This submodule exposes the low-level, stateless versions of the particle
filter, iterated filter, MOP training, and simulation algorithms for users
who need to compose them within custom JAX loops or higher-order functions.

For the standard object-oriented interface, use the :class:`~pypomp.Pomp`
and :class:`~pypomp.PanelPomp` classes instead.
"""

from .structs import PompStruct, PanelPompStruct
from .utils import align_params
from .mif import mif, panel_mif
from .pfilter import pfilter, panel_pfilter
from .mop import mop
from .dpop import dpop
from .simulate import simulate
from .train import train, panel_train
from .pmcmc import pmcmc
from .abc import abc

__all__ = [
    "PompStruct",
    "PanelPompStruct",
    "align_params",
    "mif",
    "panel_mif",
    "pfilter",
    "panel_pfilter",
    "mop",
    "dpop",
    "simulate",
    "train",
    "panel_train",
    "pmcmc",
    "abc",
]

"""
JAX-compatible random variable samplers optimized for GPUs.
"""

from .poissoninvf import fast_approx_rpoisson
from .binominvf import fast_approx_rbinom, fast_approx_rmultinom
from .gammainvf import fast_approx_rgamma
from .nbinom import fast_approx_rnbinom

__all__ = [
    "fast_approx_rpoisson",
    "fast_approx_rbinom",
    "fast_approx_rmultinom",
    "fast_approx_rgamma",
    "fast_approx_rnbinom",
]

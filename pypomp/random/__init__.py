"""
JAX-compatible random variable samplers optimized for GPUs.
"""

from . import poissoninvf, binominvf, gammainvf, nbinom, _dtype_helpers

fast_approx_rpoisson = poissoninvf.fast_approx_rpoisson
fast_approx_rbinom = binominvf.fast_approx_rbinom
fast_approx_rmultinom = binominvf.fast_approx_rmultinom
fast_approx_rgamma = gammainvf.fast_approx_rgamma
fast_approx_rnbinom = nbinom.fast_approx_rnbinom

__all__ = [
    "fast_approx_rpoisson",
    "fast_approx_rbinom",
    "fast_approx_rmultinom",
    "fast_approx_rgamma",
    "fast_approx_rnbinom",
]

del poissoninvf, binominvf, gammainvf, nbinom, _dtype_helpers

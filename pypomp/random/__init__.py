"""
JAX-compatible random variable samplers optimized for GPUs.
"""

from . import poisson, binom, gamma, nbinom, _dtype_helpers

fast_poisson = poisson.fast_poisson
fast_binomial = binom.fast_binomial
fast_multinomial = binom.fast_multinomial
fast_gamma = gamma.fast_gamma
fast_nbinomial = nbinom.fast_nbinomial

__all__ = [
    "fast_poisson",
    "fast_binomial",
    "fast_multinomial",
    "fast_gamma",
    "fast_nbinomial",
]

del poisson, binom, gamma, nbinom, _dtype_helpers

"""
JAX-compatible random variable samplers optimized for GPU execution.

All samplers are JIT-compiled and vectorized.  The implementations use approximate
inverse CDF methods so they can run on GPUs without incurring major warp divergence.

Exported Samplers
-----------------
fast_poisson
    Approximate Poisson sampler (Giles 2016).
fast_binomial
    Approximate Binomial sampler (Giles & Beentjes 2024).
fast_multinomial
    Approximate Multinomial sampler based on :func:`fast_binomial`.
fast_gamma
    Approximate Gamma sampler (Temme 1992).
fast_nbinomial
    Negative Binomial sampler via Gamma-Poisson mixture.

Exported Inverse CDFs
---------------------
poissoninv
    Vectorised inverse Poisson CDF.
binominv
    Vectorised inverse Binomial CDF.
gammainv
    Vectorised inverse Gamma CDF.
"""

from . import poisson, binom, gamma, nbinom, _dtype_helpers

fast_poisson = poisson.fast_poisson
fast_binomial = binom.fast_binomial
fast_multinomial = binom.fast_multinomial
fast_gamma = gamma.fast_gamma
fast_nbinomial = nbinom.fast_nbinomial

poissoninv = poisson.poissoninv
binominv = binom.binominv
gammainv = gamma.gammainv

__all__ = [
    "fast_poisson",
    "fast_binomial",
    "fast_multinomial",
    "fast_gamma",
    "fast_nbinomial",
    "poissoninv",
    "binominv",
    "gammainv",
]

del poisson, binom, gamma, nbinom, _dtype_helpers

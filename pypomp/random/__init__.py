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

Exported DPOP Log-Weights
-------------------------
poisson_logw, binomial_logw, multinomial_logw, euler_multinomial_logw
    Log-weight increments of draws, for a DPOP model's ``_logw`` state.
"""

from . import _dtype_helpers, binom, dpop, gamma, nbinom, poisson

fast_poisson = poisson.fast_poisson
fast_binomial = binom.fast_binomial
fast_multinomial = binom.fast_multinomial
fast_gamma = gamma.fast_gamma
fast_nbinomial = nbinom.fast_nbinomial

poissoninv = poisson.poissoninv
binominv = binom.binominv
gammainv = gamma.gammainv

poisson_logw = dpop.poisson_logw
binomial_logw = dpop.binomial_logw
multinomial_logw = dpop.multinomial_logw
euler_multinomial_logw = dpop.euler_multinomial_logw

__all__ = [
    "binomial_logw",
    "binominv",
    "euler_multinomial_logw",
    "fast_binomial",
    "fast_gamma",
    "fast_multinomial",
    "fast_nbinomial",
    "fast_poisson",
    "gammainv",
    "multinomial_logw",
    "poisson_logw",
    "poissoninv",
]

del poisson, binom, dpop, gamma, nbinom, _dtype_helpers

"""
JAX implementation of Negative Binomial sampling using Gamma-Poisson mixture.
"""

from __future__ import annotations

from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes

from ._dtype_helpers import _get_available_dtype, check_and_canonicalize_user_dtype
from .gammainvf import fast_approx_rgamma
from .poissoninvf import fast_approx_rpoisson


@partial(jax.jit, static_argnames=["dtype"])
def fast_approx_rnbinom(
    key: Array,
    n: Array,
    p: Array | None = None,
    mu: Array | None = None,
    dtype: np.dtype | None = None,
) -> Array:
    """
    Generate negative binomial random variables using a Gamma-Poisson mixture.

    The Negative Binomial distribution NB(n, p) represents the number of failures
    before n successes, where p is the probability of success.
    Alternatively, it can be parameterized by n (size) and mu (mean).

    NB(n, p) has mean mu = n * (1-p) / p.

    Args:
        key: PRNG key used as the random key.
        n: Number of successes (size parameter). Must be positive.
        p: Probability of success (0 < p <= 1). Exactly one of p or mu must be provided.
        mu: Mean of the distribution. Exactly one of p or mu must be provided.
        dtype: optional, a float or integer dtype for the returned values (default float64 if
            jax_enable_x64 is true, otherwise float32).

    Returns:
        Negative binomial random variables with the same broadcast shape as the inputs.
    """
    if (p is None and mu is None) or (p is not None and mu is not None):
        raise ValueError("Exactly one of p or mu must be provided.")

    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    assert dtype is not None

    if not (
        dtypes.issubdtype(dtype, np.floating) or dtypes.issubdtype(dtype, np.integer)
    ):
        raise ValueError(
            f"dtype argument to `fast_approx_rnbinom` must be a float or integer dtype, got {dtype}"
        )

    current_float_64 = dtypes.issubdtype(dtype, np.int64) or dtypes.issubdtype(
        dtype, np.float64
    )
    float_dtype = jnp.float64 if current_float_64 else jnp.float32
    float_dtype = _get_available_dtype(float_dtype)
    assert float_dtype is not None

    n = jnp.asarray(n, dtype=float_dtype)

    if p is not None:
        p = jnp.asarray(p, dtype=float_dtype)
        invalid = (n <= 0.0) | (p <= 0.0) | (p > 1.0)
        scale = (1.0 - p) / jnp.maximum(p, jnp.finfo(float_dtype).tiny)
    else:
        mu = jnp.asarray(mu, dtype=float_dtype)
        invalid = (n <= 0.0) | (mu < 0.0)
        scale = mu / jnp.maximum(n, jnp.finfo(float_dtype).tiny)

    safe_n = jnp.where(invalid, 1.0, n)
    safe_scale = jnp.where(invalid, 1.0, scale)

    key_gamma, key_poisson = jax.random.split(key)

    gamma_samples = fast_approx_rgamma(key_gamma, safe_n, dtype=float_dtype)
    lam = gamma_samples * safe_scale
    poisson_dtype = (
        jnp.int64
        if dtypes.issubdtype(dtype, np.int64) or dtypes.issubdtype(dtype, np.float64)
        else jnp.int32
    )
    x = fast_approx_rpoisson(key_poisson, lam, dtype=poisson_dtype)
    x = jnp.where(lam == 0.0, 0, x)

    if dtypes.issubdtype(dtype, np.floating):
        res = x.astype(dtype)
        res = jnp.where(invalid, jnp.nan, res)
    else:
        res = x.astype(dtype)
        # For integer dtype, follow the convention of returning -1 for invalid inputs
        res = jnp.where(invalid, -1, res)

    return res

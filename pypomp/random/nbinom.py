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
from .gamma import fast_gamma
from .poisson import fast_poisson


@partial(
    jax.jit,
    static_argnames=[
        "dtype",
        "gamma_newton_loops",
        "poisson_newton_loops",
        "poisson_inverse_cdf_loops",
        "gamma_adjustment_size",
    ],
)
def fast_nbinomial(
    key: Array,
    n: Array,
    p: Array | None = None,
    mu: Array | None = None,
    dtype: np.dtype | None = None,
    gamma_newton_loops: int = 3,
    poisson_newton_loops: int = 5,
    poisson_inverse_cdf_loops: int = 20,
    gamma_adjustment_size: int = 3,
) -> Array:
    """Sample Negative Binomial random variates using a GPU-optimized algorithm.

    Draws from NB(n, p) (number of failures before ``n`` successes) via a
    Gamma-Poisson mixture.  Both steps use the approximate GPU-optimized
    samplers :func:`fast_gamma` and :func:`fast_poisson`.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    n : jax.Array
        Size (number of successes) parameter.  Must be positive.
    p : jax.Array or None, optional
        Success probability in ``(0, 1]``.  Mutually exclusive with
        ``mu``.
    mu : jax.Array or None, optional
        Mean of the distribution: ``mu = n * (1 - p) / p``.  Mutually
        exclusive with ``p``.
    dtype : np.dtype or None, optional
        Output dtype (float or integer).  Defaults to ``float64`` if
        ``jax_enable_x64=True``, otherwise ``float32``.
    gamma_newton_loops : int, optional
        Newton-Raphson iterations for the Gamma sampler.  Defaults to
        ``3``.
    poisson_newton_loops : int, optional
        Newton-Raphson iterations for the Poisson sampler.  Defaults to
        ``5``.
    poisson_inverse_cdf_loops : int, optional
        Exact inverse CDF iterations for the Poisson sampler.  Defaults
        to ``20``.
    gamma_adjustment_size : int, optional
        Uniform adjustment steps for the Gamma sampler.  Defaults to
        ``3``.

    Returns
    -------
    jax.Array
        Negative Binomial samples with the broadcast shape of the
        inputs.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pypomp.random import fast_nbinomial
    >>> fast_nbinomial(jax.random.key(0), n=jnp.array(5.0), mu=jnp.array(3.0))
    Array(2, dtype=int32)

    See Also
    --------
    fast_gamma : Gamma sampler used internally.
    fast_poisson : Poisson sampler used internally.
    """
    if (p is None and mu is None) or (p is not None and mu is not None):
        raise ValueError("Exactly one of p or mu must be provided.")

    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    assert dtype is not None

    if not (
        dtypes.issubdtype(dtype, np.floating) or dtypes.issubdtype(dtype, np.integer)
    ):
        raise ValueError(
            f"dtype argument to `fast_nbinomial` must be a float or integer dtype, got {dtype}"
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
        invalid = (n <= 0.0) | (p <= 0.0) | (p > 1.0) | jnp.isnan(n) | jnp.isnan(p)
        scale = (1.0 - p) / jnp.maximum(p, jnp.finfo(float_dtype).tiny)
    else:
        mu = jnp.asarray(mu, dtype=float_dtype)
        invalid = (n <= 0.0) | (mu < 0.0) | jnp.isnan(n) | jnp.isnan(mu)
        scale = mu / jnp.maximum(n, jnp.finfo(float_dtype).tiny)

    safe_n: Array = jnp.where(invalid, 1.0, n)
    safe_scale: Array = jnp.where(invalid, 1.0, scale)

    key_gamma, key_poisson = jax.random.split(key)

    gamma_samples = fast_gamma(
        key_gamma,
        safe_n,
        dtype=float_dtype,
        adjustment_size=gamma_adjustment_size,
        newton_steps=gamma_newton_loops,
    )
    lam = gamma_samples * safe_scale
    # Clamp to prevent float overflow to inf in fast_poisson
    lam = jnp.minimum(lam, jnp.finfo(float_dtype).max / 10.0)
    poisson_dtype = (
        jnp.int64
        if dtypes.issubdtype(dtype, np.int64) or dtypes.issubdtype(dtype, np.float64)
        else jnp.int32
    )
    x: Array = fast_poisson(
        key_poisson,
        lam,
        dtype=poisson_dtype,
        max_newton_loops=poisson_newton_loops,
        max_inverse_cdf_loops=poisson_inverse_cdf_loops,
    )
    x = jnp.where(lam == 0.0, 0, x)

    if dtypes.issubdtype(dtype, np.floating):
        res: Array = x.astype(dtype)
        res = jnp.where(invalid, jnp.nan, res)
    else:
        res: Array = x.astype(dtype)
        # For integer dtype, follow the convention of returning -1 for invalid inputs
        res = jnp.where(invalid, -1, res)

    return res

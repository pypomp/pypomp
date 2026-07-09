"""
JAX implementation of the inverse incomplete beta function approximation.

The implementation follows the methodology from Giles and Beentjes (2024)
"Approximation of an Inverse of the Incomplete Beta Function".

This implements the normal asymptotic expansion formulas Q_N0, Q_N1, Q_N2
from Section 2 of the paper.
"""

from __future__ import annotations

from functools import partial

import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
from jax._src import dtypes

from ._dtype_helpers import (
    check_and_canonicalize_user_dtype,
    _get_available_dtype,
)


@partial(jax.jit, static_argnames=["order", "exact_max", "dtype"])
def fast_multinomial(
    key: Array,
    n: Array,
    p: Array,
    order: int = 2,
    exact_max: int = 5,
    dtype: np.dtype | None = None,
) -> Array:
    """Sample multinomial random variates using a GPU-optimized inverse CDF algorithm.

    Generates multinomial counts by sequentially sampling binomial
    components via :func:`fast_binomial`.  Follows the methodology from
    Giles and Beentjes (2024).  Results are very close to exact but not
    guaranteed to be identical to a reference sampler.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    n : jax.Array
        Number of trials.  Shape ``(...,)``.
    p : jax.Array
        Category probabilities.  Shape ``(..., k)`` where ``k`` is the
        number of categories.  Probabilities along the last axis are
        normalised automatically.
    order : int, optional
        Order of the beta-function approximation (0, 1, or 2).  Defaults
        to ``2`` (most accurate).
    exact_max : int, optional
        Maximum iterations for the bottom-up exact inverse CDF stage.
        Defaults to ``5``.
    dtype : np.dtype or None, optional
        Output dtype (float or integer).  Defaults to ``float64`` if
        ``jax_enable_x64=True``, otherwise ``float32``.  Integer dtypes
        return ``-1`` for invalid inputs.

    Returns
    -------
    jax.Array
        Multinomial count array with the same shape as ``p`` and the
        specified ``dtype``.

    References
    ----------
    Giles, Michael B., and Casper Beentjes. "Approximation of an
    Inverse of the Incomplete Beta Function." In *Mathematical Software
    – ICMS 2024*, vol. 14749. Springer, 2024.
    https://doi.org/10.1007/978-3-031-64529-7_22.
    """
    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    if not (
        dtypes.issubdtype(dtype, np.floating) or dtypes.issubdtype(dtype, np.integer)
    ):
        raise ValueError(
            f"dtype argument to `fast_multinomial` must be a float or integer dtype, got {dtype}"
        )
    n = jnp.asarray(n)
    p = jnp.asarray(p)
    if p.ndim < 1:
        raise ValueError("p must have at least 1 dimension (categories)")

    p_shape = p.shape
    shape_batch = p_shape[:-1]
    num_cat = int(p_shape[-1])

    n_broadcast = jnp.broadcast_to(n, shape_batch)

    p_sum = jnp.sum(p, axis=-1, keepdims=True)
    p_safe_sum = jnp.where(p_sum == 0, 1.0, p_sum)
    p = p / p_safe_sum

    keys = jax.random.split(key, num_cat - 1)
    n_remaining = n_broadcast
    p_remain = jnp.ones(shape_batch, dtype=p.dtype)
    out = []

    for j in range(num_cat - 1):
        p_remain_safe = jnp.where(p_remain > 0.0, p_remain, 1.0)
        p_cur = p[..., j] / p_remain_safe
        p_cur = jnp.clip(p_cur, 0.0, 1.0)
        x = fast_binomial(
            keys[j],
            n_remaining,
            p_cur,
            order=order,
            exact_max=exact_max,
            dtype=dtype,
        )
        out.append(x)
        n_remaining = n_remaining - x
        p_remain = p_remain - p[..., j]
    out.append(n_remaining)

    return jnp.stack(out, axis=-1).astype(dtype)


@partial(jax.jit, static_argnames=["order", "exact_max", "dtype"])
def fast_binomial(
    key: Array,
    n: Array,
    p: Array,
    order: int = 2,
    exact_max: int = 5,
    dtype: np.dtype | None = None,
) -> Array:
    """Sample binomial random variates using a GPU-optimized inverse CDF algorithm.

    Generates binomial counts with parameters ``(n, p)`` using an
    approximate inverse incomplete beta function method.  The
    implementation follows Giles and Beentjes (2024), with an optional
    exact inverse CDF correction for small or extreme quantiles.
    Results are very close to exact but not guaranteed to be identical
    to a reference sampler.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    n : jax.Array
        Number of Bernoulli trials.
    p : jax.Array
        Success probability in ``[0, 1]``.
    order : int, optional
        Order of the beta-function approximation (0, 1, or 2).  Defaults
        to ``2`` (most accurate).
    exact_max : int, optional
        Maximum iterations for the bottom-up exact inverse CDF stage.
        Defaults to ``5``.
    dtype : np.dtype or None, optional
        Output dtype (float or integer).  Defaults to ``float64`` if
        ``jax_enable_x64=True``, otherwise ``float32``.  Integer dtypes
        return ``-1`` for invalid inputs.

    Returns
    -------
    jax.Array
        Binomial samples with the broadcast shape of ``n`` and ``p`` and
        the specified ``dtype``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pypomp.random import fast_binomial
    >>> fast_binomial(jax.random.key(0), n=jnp.array(10), p=jnp.array(0.3))
    Array(3., dtype=float32)

    References
    ----------
    Giles, Michael B., and Casper Beentjes. "Approximation of an
    Inverse of the Incomplete Beta Function." In *Mathematical Software
    – ICMS 2024*, vol. 14749. Springer, 2024.
    https://doi.org/10.1007/978-3-031-64529-7_22.
    """
    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    assert dtype is not None
    if not (
        dtypes.issubdtype(dtype, np.floating) or dtypes.issubdtype(dtype, np.integer)
    ):
        raise ValueError(
            f"dtype argument to `fast_binomial` must be a float or integer dtype, got {dtype}"
        )

    dtype = _get_available_dtype(dtype)
    assert dtype is not None

    if dtypes.issubdtype(dtype, np.integer):
        if dtypes.issubdtype(dtype, np.int64):
            float_dtype = jnp.float64
        else:
            float_dtype = jnp.float32
    else:
        float_dtype = dtype

    float_dtype = _get_available_dtype(float_dtype)
    assert float_dtype is not None

    shape = jnp.broadcast_shapes(n.shape, p.shape)

    u = jax.random.uniform(key, shape, dtype=float_dtype)
    # Clamp u to avoid discretization artifacts (0.0) and extreme tail issues (1.0)
    u_min = jnp.finfo(float_dtype).tiny
    u_max = jnp.nextafter(
        jnp.array(1.0, dtype=float_dtype), jnp.array(0.0, dtype=float_dtype)
    )
    u = jnp.clip(u, u_min, u_max)

    n_float = jnp.asarray(n, dtype=float_dtype)
    p_float = jnp.asarray(p, dtype=float_dtype)
    x = binominv(u, n_float, p_float, exact_max, order=order, dtype=float_dtype)

    if jnp.issubdtype(dtype, jnp.integer):
        x = jnp.nan_to_num(x, nan=-1.0).astype(dtype)

    return x.astype(dtype)


@partial(jax.jit, static_argnames=["order", "exact_max", "dtype"])
def binominv(
    u: Array,
    n: Array,
    p: Array,
    exact_max: int = 5,
    order: int = 2,
    dtype: np.dtype | None = None,
) -> Array:
    """Compute the approximate inverse binomial CDF using JAX primitives.

    Vectorised implementation using the normal asymptotic expansion
    formulas from Giles and Beentjes (2024).  A bottom-up exact
    inverse CDF calculation is performed for small values.

    Parameters
    ----------
    u : jax.Array
        Uniform probabilities in ``[0, 1]``.  Scalar or array.
    n : jax.Array
        Number of trials.  Must be a positive integer or float.
    p : jax.Array
        Success probability in ``[0, 1]``.
    exact_max : int, optional
        Maximum iterations for the bottom-up exact inverse CDF stage.
        Defaults to ``5``.
    order : int, optional
        Order of approximation (0, 1, or 2).  Defaults to ``2``.
    dtype : np.dtype or None, optional
        Floating-point dtype for computation.  Inferred from inputs if
        ``None``.

    Returns
    -------
    jax.Array
        Array of binomial quantiles with the broadcast shape of the
        inputs.

    References
    ----------
    .. [1] Giles, Michael B., and Casper Beentjes. "Approximation of an
       Inverse of the Incomplete Beta Function." In *Mathematical Software
       – ICMS 2024*, vol. 14749. Springer, 2024.
       https://doi.org/10.1007/978-3-031-64529-7_22.

    See Also
    --------
    fast_binomial : High-level sampler that wraps this function.
    """
    u, n, p = jnp.broadcast_arrays(u, n, p)
    if dtype is None:
        dtype = jnp.result_type(u, n, p)
        if not dtypes.issubdtype(dtype, np.floating):
            dtype = jnp.float32

    dtype = check_and_canonicalize_user_dtype(dtype)
    assert dtype is not None
    if not (
        dtypes.issubdtype(dtype, np.floating) or dtypes.issubdtype(dtype, np.integer)
    ):
        raise ValueError(
            f"dtype argument to `binominv` must be a float or integer dtype, got {dtype}"
        )

    if dtypes.issubdtype(dtype, np.integer):
        if dtypes.issubdtype(dtype, np.int64):
            float_dtype = jnp.float64
        else:
            float_dtype = jnp.float32
    else:
        float_dtype = dtype

    float_dtype = _get_available_dtype(float_dtype)
    assert float_dtype is not None

    u_float = jnp.asarray(u, dtype=float_dtype)
    n_float = jnp.asarray(n, dtype=float_dtype)
    p_float = jnp.asarray(p, dtype=float_dtype)

    nan = jnp.array(jnp.nan, dtype=float_dtype)

    invalid_n = n_float < 0.0
    invalid_p = (p_float < 0.0) | (p_float > 1.0)
    invalid_u = (u_float < 0.0) | (u_float > 1.0)
    invalid = invalid_n | invalid_p | invalid_u

    n_is_zero = n_float == 0.0
    u_is_zero = u_float == 0.0
    u_is_one = u_float == 1.0
    p_is_zero = p_float == 0.0
    p_is_one = p_float == 1.0

    p_val = jnp.asarray(p_float, dtype=float_dtype)
    flip = p_val > 0.5
    p_safe = jnp.where(flip, 1.0 - p_val, p_val)
    u_flipped = jnp.where(flip, 1.0 - u_float, u_float)

    n_safe = jnp.where(invalid_n, 1.0, n_float)
    p_safe = jnp.clip(p_safe, 0.0, 1.0)

    # Clip u_safe to avoid norm.ppf underflow/overflow
    u_clip_min = jnp.finfo(float_dtype).eps
    u_safe = jnp.clip(u_flipped, u_clip_min, 1.0 - u_clip_min)

    q = 1.0 - p_safe
    w = norm.ppf(u_safe)
    w2 = w * w
    np_ = n_safe * p_safe
    npq_ = np_ * q
    sqrt_npq = jnp.sqrt(jnp.maximum(npq_, jnp.finfo(float_dtype).tiny))
    pq = p_safe * q

    args = (u_safe, n_safe, p_safe, q, w, w2, np_, sqrt_npq, pq)

    safe_order = max(0, min(2, int(order)))
    order_idx = safe_order
    branches = [
        lambda x: _q_n0(*x),
        lambda x: _q_n1(*x),
        lambda x: _q_n2(*x),
    ]
    q_u = lax.switch(order_idx, branches, args)
    k_approx = jnp.clip(jnp.floor(q_u), 0.0, n_safe)
    # Cap to prevent wild tail divergence of asymptotic expansions when np is small
    max_reasonable = np_ + 6.0 * sqrt_npq + 5.0
    k_approx = jnp.minimum(k_approx, max_reasonable)

    u_exact = jnp.clip(u_flipped, 0.0, 1.0)
    k_bottom_up = _binom_bottom_up(
        u_exact, n_safe, p_safe, k_approx, float_dtype, max_k=exact_max
    )

    x_cutoff = 10
    np_cutoff = 4.0
    k_small = k_approx < x_cutoff
    np_small = np_ < np_cutoff
    use_bottom_up = k_small | np_small
    k_approx = jnp.where(use_bottom_up, k_bottom_up, k_approx)

    k_flipped = jnp.where(flip, n_safe - k_approx, k_approx)

    k_result = k_flipped
    k_result = jnp.where(n_is_zero, 0.0, k_result)
    k_result = jnp.where(u_is_zero, 0.0, k_result)
    k_result = jnp.where(u_is_one, n_safe, k_result)
    k_result = jnp.where(p_is_zero, 0.0, k_result)
    k_result = jnp.where(p_is_one, n_safe, k_result)
    k_result = jnp.clip(k_result, 0.0, n_safe)
    k_result = jnp.where(invalid, nan, k_result)

    if dtypes.issubdtype(dtype, np.integer):
        return jnp.nan_to_num(k_result, nan=-1.0).astype(dtype)
    return k_result.astype(dtype)


def _binom_bottom_up(
    u: Array,
    n: Array,
    p: Array,
    approx: Array,
    dtype,
    max_k: int = 20,
) -> Array:
    """
    Compute the exact inverse CDF for small k by accumulating the binomial CDF.
    Includes protection against numerical stalling in the tail.
    """
    tiny = jnp.finfo(dtype).tiny
    epsilon = jnp.finfo(dtype).eps

    q = jnp.clip(1.0 - p, tiny, 1.0)
    p_safe = jnp.clip(p, 0.0, 1.0)
    q_safe = jnp.clip(q, tiny, 1.0)
    ratio_multiplier = p_safe / q_safe

    log_q = jnp.log1p(-p_safe)
    pmf = jnp.where(
        n == 0.0,
        1.0,
        jnp.exp(n * log_q),
    )
    cdf = pmf

    found = cdf >= u

    result = jnp.where(found, 0.0, approx)

    for i in range(1, max_k):
        k_curr_val = i
        k_curr = jnp.full_like(result, k_curr_val)
        k_prev = i - 1

        num = jnp.maximum(n - k_prev, 0.0)
        den = k_curr_val

        pmf = pmf * (num / den) * ratio_multiplier
        cdf = cdf + pmf

        # Check for numerical stall:
        # If CDF is nearly 1.0 and PMF is negligible, the CDF won't increase further.
        # We claim any remaining u values belong to this tail bucket.
        stall_threshold = 1.0 - (epsilon * 10.0)
        is_stalled = (cdf > stall_threshold) & (pmf < epsilon)
        found_now = (cdf >= u) | is_stalled

        is_new_discovery = jnp.logical_and(~found, found_now)
        result = jnp.where(is_new_discovery, k_curr, result)

        found = jnp.logical_or(found, found_now)

    return jnp.clip(result, 0.0, n)


def _q_n0(
    u: Array,
    n: Array,
    p: Array,
    q: Array,
    w: Array,
    w2: Array,
    np_: Array,
    sqrt_npq: Array,
    pq: Array,
) -> Array:
    """
    Computes the Q_N0 approximation (Order 0) from Giles and Beentjes (2024).

    Q_N0 = np + sqrt(npq)w + (2 + 2p + (q-p)w^2) / 6
    """
    return np_ + sqrt_npq * w + (2.0 + 2.0 * p + (q - p) * w2) / 6.0


def _q_n1(
    u: Array,
    n: Array,
    p: Array,
    q: Array,
    w: Array,
    w2: Array,
    np_: Array,
    sqrt_npq: Array,
    pq: Array,
) -> Array:
    """
    Computes the Q_N1 approximation (Order 1) from Giles and Beentjes (2024).

    Q_N1 = Q_N0 + ((-2+14pq)w + (-1-2pq)w^3) / (72 * sqrt(npq))
    """
    q_n0 = _q_n0(u, n, p, q, w, w2, np_, sqrt_npq, pq)
    w3 = w2 * w
    numerator_t2 = (-2.0 + 14.0 * pq) * w + (-1.0 - 2.0 * pq) * w3
    denominator_t2 = 72.0 * sqrt_npq
    tiny = jnp.finfo(w.dtype).tiny
    term2 = numerator_t2 / jnp.maximum(denominator_t2, tiny)
    return q_n0 + term2


def _q_n2(
    u: Array,
    n: Array,
    p: Array,
    q: Array,
    w: Array,
    w2: Array,
    np_: Array,
    sqrt_npq: Array,
    pq: Array,
) -> Array:
    """
    Computes the Q_N2 approximation (Order 2) from Giles and Beentjes (2024).

    Q_N2 = Q_N1 + ((p-q)(2+pq)(16-7w^2-3w^4)) / (1620 * npq)
    """
    q_n1 = _q_n1(u, n, p, q, w, w2, np_, sqrt_npq, pq)
    w3 = w2 * w
    w4 = w3 * w
    npq_ = sqrt_npq * sqrt_npq
    numerator_t3 = (p - q) * (2.0 + pq) * (16.0 - 7.0 * w2 - 3.0 * w4)
    denominator_t3 = 1620.0 * npq_
    tiny = jnp.finfo(w.dtype).tiny
    term3 = numerator_t3 / jnp.maximum(denominator_t3, tiny)
    return q_n1 + term3

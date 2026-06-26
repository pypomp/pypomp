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


@partial(jax.jit, static_argnames=["dtype"])
def fast_multinomial(
    key: Array, n: Array, p: Array, dtype: np.dtype | None = None
) -> Array:
    """
    Generate multinomial random variables using the inverse CDF method with fast_binomial in order to run fast on GPUs.

    The implementation follows the methodology from Giles and Beentjes (2024). To more accurately handle cases where np is very small or the random draw is expected to be close to 0 or n, we apply the exact inverse CDF method in a manner similar to Giles (2016). Our implementation of the method does not produce exact multinomial random variables, but it is very close to exact.

    Args:
        key: PRNG key used as the random key.
        n: Number of trials for the multinomial distribution. Shape: (...,)
        p: Probabilities for each category. Shape: (..., k), where k = num categories.
           Probabilities along the last axis must sum to 1.
        dtype: optional, a float dtype for the returned values (default float64 if
            jax_enable_x64 is true, otherwise float32). If integer, returns -1 for invalid inputs instead of nan.
    Returns:
        Multinomial counts. Same shape as p, but with specified dtype.
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
        x = fast_binomial(keys[j], n_remaining, p_cur, dtype=dtype)
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
    """
    Generate binomial random variables using a JAX implementation of the inverse incomplete beta function approximation in order to run fast on GPUs.

    The implementation follows the methodology from Giles and Beentjes (2024). To more accurately handle cases where np is very small or the random draw is expected to be close to 0 or n, we apply the exact inverse CDF method in a manner similar to Giles (2016). Our implementation of the method does not produce exact binomial random variables, but it is very close to exact.

    Args:
        key: PRNG key used as the random key.
        n: Number of trials for the binomial distribution.
        p: Success probability for the binomial distribution.
        order: Order of approximation (0, 1, or 2). Default is 2 for best accuracy.
        exact_max: Maximum number of loop iterations to perform for the bottom up exact inverse CDF method.
        dtype: optional, a float dtype for the returned values (default float64 if
            jax_enable_x64 is true, otherwise float32). If integer, returns -1 for invalid inputs instead of nan.

    Returns:
        Binomial random variables with the same shape as n and p.

    References:
        * Giles, Michael B., and Casper Beentjes. "Approximation of an Inverse of the Incomplete Beta Function." In Mathematical Software – ICMS 2024, vol. 14749. Lecture Notes in Computer Science. Springer Nature Switzerland, 2024. https://doi.org/10.1007/978-3-031-64529-7_22.
        * Giles, Michael B. "Algorithm 955: Approximation of the Inverse Poisson Cumulative Distribution Function." ACM Transactions on Mathematical Software 42, no. 1 (2016): 1–22. https://doi.org/10.1145/2699466.
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
    exact_max: int,
    order: int = 2,
    dtype: np.dtype | None = None,
) -> Array:
    """
    Inverse binomial CDF approximation using Giles and Beentjes (2024).

    Computes the smallest integer k such that P(X <= k) >= u, where
    X ~ Binomial(n, p).

    Uses the normal asymptotic expansion formulas from Section 2 of the paper:
    - Q_N0 (order=0): Basic approximation
    - Q_N1 (order=1): First-order correction
    - Q_N2 (order=2): Second-order correction (default, most accurate)

    The binomial CDF can be expressed as:
        F(k; n, p) = I_{1-p}(n-k, k+1)
    where I_x(a, b) is the regularized incomplete beta function.

    Args:
        u: Probabilities (scalar or array) in the interval [0, 1].
        n: Number of trials (must be positive integer or positive float).
        p: Success probability (must be in [0, 1]).
        exact_max: Maximum number of loop iterations to perform for the bottom up exact inverse CDF method.
        order: Order of approximation (0, 1, or 2).
        dtype: Data type for computation.

    Returns:
        Array with the same broadcast shape as inputs, containing integer
        values k such that P(X <= k) >= u.
    """
    u, n, p = jnp.broadcast_arrays(u, n, p)
    if dtype is None:
        dtype = jnp.result_type(u, n, p)
        if not dtypes.issubdtype(dtype, np.floating):
            dtype = jnp.float32

    dtype = _get_available_dtype(dtype)
    assert dtype is not None

    u = jnp.asarray(u, dtype=dtype)
    n = jnp.asarray(n, dtype=dtype)
    p = jnp.asarray(p, dtype=dtype)

    nan = jnp.array(jnp.nan, dtype=dtype)

    invalid_n = n < 0.0
    invalid_p = (p < 0.0) | (p > 1.0)
    invalid_u = (u < 0.0) | (u > 1.0)
    invalid = invalid_n | invalid_p | invalid_u

    n_is_zero = n == 0.0
    u_is_zero = u == 0.0
    u_is_one = u == 1.0
    p_is_zero = p == 0.0
    p_is_one = p == 1.0

    p_val = jnp.asarray(p, dtype=dtype)
    flip = p_val > 0.5
    p_safe = jnp.where(flip, 1.0 - p_val, p_val)
    u_flipped = jnp.where(flip, 1.0 - u, u)

    n_safe = jnp.where(invalid_n, 1.0, n)
    p_safe = jnp.clip(p_safe, 0.0, 1.0)

    # Clip u_safe to avoid norm.ppf underflow/overflow
    u_clip_min = (
        jnp.array(1e-16, dtype=dtype)
        if dtypes.issubdtype(dtype, np.float64)
        else jnp.array(1e-9, dtype=dtype)
    )
    u_safe = jnp.clip(u_flipped, u_clip_min, 1.0 - u_clip_min)

    q = 1.0 - p_safe
    w = norm.ppf(u_safe)
    w2 = w * w
    np_ = n_safe * p_safe
    npq_ = np_ * q
    sqrt_npq = jnp.sqrt(jnp.maximum(npq_, jnp.finfo(dtype).tiny))
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
        u_exact, n_safe, p_safe, k_approx, dtype, max_k=exact_max
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

    return k_result


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

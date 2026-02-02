"""
JAX implementation of the inverse incomplete beta function approximation.

The implementation follows the methodology from Giles and Beentjes (2024)
"Approximation of an Inverse of the Incomplete Beta Function".

This implements the normal asymptotic expansion formulas Q_N0, Q_N1, Q_N2
from Section 2 of the paper.
"""

from __future__ import annotations

import math
from typing import cast
from functools import partial

import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
from jax._src import dtypes

from ._dtype_helpers import check_and_canonicalize_user_dtype, _get_available_dtype


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
    term2 = numerator_t2 / jnp.maximum(denominator_t2, jnp.finfo(jnp.float32).tiny)
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
    npq_ = sqrt_npq * sqrt_npq  # npq
    numerator_t3 = (p - q) * (2.0 + pq) * (16.0 - 7.0 * w2 - 3.0 * w4)
    denominator_t3 = 1620.0 * npq_
    term3 = numerator_t3 / jnp.maximum(denominator_t3, jnp.finfo(jnp.float32).tiny)
    return q_n1 + term3


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
    Includes protection against float32 numerical stalling in the tail.
    """
    tiny = jnp.finfo(dtype).tiny
    epsilon = 1e-7  # Approx machine epsilon for float32

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

    # Initialize result with approx (fallback), but if found at k=0, use 0.
    result = cast(Array, jnp.where(found, 0.0, approx))

    for i in range(1, max_k):
        k_curr_val = i
        k_curr = jnp.full_like(result, k_curr_val)
        k_prev = i - 1

        # Recurrence: P(k) = P(k-1) * ((n - k + 1) / k) * (p / q)
        num = jnp.maximum(n - k_prev, 0.0)
        den = k_curr_val

        pmf = pmf * (num / den) * ratio_multiplier
        cdf = cdf + pmf

        # Check for numerical stall:
        # If CDF is nearly 1.0 and PMF is negligible, the CDF won't increase further
        # in float32. We claim any remaining u values belong to this tail bucket.
        if dtype == jnp.float32:
            is_stalled = (cdf > 0.95) & (pmf < epsilon)
        else:
            is_stalled = False

        found_now = (cdf >= u) | is_stalled

        is_new_discovery = jnp.logical_and(~found, found_now)
        result = cast(Array, jnp.where(is_new_discovery, k_curr, result))

        found = jnp.logical_or(found, found_now)

    return cast(Array, jnp.clip(result, 0.0, n))


def _binominvf_scalar(
    u: Array, n: Array, p: Array, exact_max: int, order: int = 2
) -> Array:
    """
    Scalar version of inverse binomial CDF approximation using Giles and Beentjes (2024).

    Computes the smallest integer k such that P(X <= k) >= u, where
    X ~ Binomial(n, p). Uses the normal asymptotic expansion formulas
    Q_N0, Q_N1, or Q_N2 from the paper.

    Args:
        u: Probability in [0, 1]
        n: Number of trials
        p: Success probability
        order: Order of approximation (0, 1, or 2). Default is 2 for best accuracy.
    """
    dtype = jnp.float32
    u = jnp.asarray(u, dtype=dtype)
    n = jnp.asarray(n, dtype=dtype)
    p = jnp.asarray(p, dtype=dtype)

    # Handle edge cases
    nan = jnp.array(jnp.nan, dtype=dtype)

    invalid_n = n < 0.0
    invalid_p = (p < 0.0) | (p > 1.0)
    invalid_u = (u < 0.0) | (u > 1.0)
    invalid = invalid_n | invalid_p | invalid_u

    # Special cases
    n_is_zero = n == 0.0
    u_is_zero = u == 0.0
    u_is_one = u == 1.0
    p_is_zero = p == 0.0
    p_is_one = p == 1.0

    # For p = 0, all outcomes are 0
    # For p = 1, all outcomes are n
    # For u = 0, return 0
    # For u = 1, return n

    # Use symmetry: if p > 0.5, work with 1-p and flip result
    p_val = jnp.asarray(p, dtype=dtype)
    flip = p_val > 0.5
    p_safe = cast(Array, jnp.where(flip, 1.0 - p_val, p_val))
    u_flipped = cast(Array, jnp.where(flip, 1.0 - u, u))

    n_safe = cast(Array, jnp.where(invalid_n, 1.0, n))
    p_safe = cast(Array, jnp.clip(p_safe, 0.0, 1.0))
    u_safe = cast(
        Array,
        jnp.clip(u_flipped, 1e-9, 1.0 - 1e-9),
    )  # Clip to avoid inf from norm.ppf

    # Pre-compute shared values for the approximation formulas
    q = 1.0 - p_safe
    w = norm.ppf(u_safe)  # w = Φ^{-1}(u)
    w2 = w * w
    np_ = n_safe * p_safe
    npq_ = np_ * q
    sqrt_npq = jnp.sqrt(jnp.maximum(npq_, jnp.finfo(dtype).tiny))
    pq = p_safe * q

    # Package arguments for helper functions
    args = (u_safe, n_safe, p_safe, q, w, w2, np_, sqrt_npq, pq)

    # Use lax.switch to select the correct function based on order
    # Order is a static Python int, so we clamp it in Python and convert to JAX int
    safe_order = max(0, min(2, int(order)))
    order_idx = safe_order
    branches = [
        lambda x: _q_n0(*x),
        lambda x: _q_n1(*x),
        lambda x: _q_n2(*x),
    ]
    q_u = lax.switch(order_idx, branches, args)

    # The paper states \overline{C}^{-1}(u) = floor(C^{-1}(u))
    # Clip to valid range [0, n] and take floor
    k_approx = jnp.clip(jnp.floor(q_u), 0.0, n_safe)

    # Compute x from the bottom up if it is less than the cutoff
    x_cutoff = 10
    np_cutoff = 4.0
    u_exact = jnp.clip(u_flipped, 0.0, 1.0)
    k_bottom_up = _binom_bottom_up(
        u_exact, n_safe, p_safe, k_approx, dtype, max_k=exact_max
    )
    k_small = k_approx < x_cutoff
    np_small = np_ < np_cutoff
    use_bottom_up = k_small | np_small
    k_approx = cast(Array, jnp.where(use_bottom_up, k_bottom_up, k_approx))

    # Apply symmetry flip if needed
    k_flipped = cast(Array, jnp.where(flip, n_safe - k_approx, k_approx))

    k_result = k_flipped
    k_result = cast(Array, jnp.where(n_is_zero, 0.0, k_result))
    k_result = cast(Array, jnp.where(u_is_zero, 0.0, k_result))
    k_result = cast(Array, jnp.where(u_is_one, n_safe, k_result))
    k_result = cast(Array, jnp.where(p_is_zero, 0.0, k_result))
    k_result = cast(Array, jnp.where(p_is_one, n_safe, k_result))
    k_result = cast(Array, jnp.clip(k_result, 0.0, n_safe))
    k_result = cast(Array, jnp.where(invalid, nan, k_result))

    return k_result


_binominvf_vmap = jax.vmap(_binominvf_scalar, in_axes=(0, 0, 0, None, None))


@partial(jax.jit, static_argnames=["order", "exact_max"])
def binominvf(
    u: Array,
    n: Array,
    p: Array,
    exact_max: int,
    order: int = 2,
) -> Array:
    """
    Vectorized inverse binomial CDF approximation using Giles and Beentjes (2024).

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
        dtype: Data type for computation.
        exact_max: Maximum number of loop iterations to perform for the bottom up exact inverse CDF method.
        order: Order of approximation (0, 1, or 2). Default is 2 for best accuracy.

    Returns:
        Array with the same broadcast shape as inputs, containing integer
        values k such that P(X <= k) >= u.
    """
    u_arr, n_arr, p_arr = jnp.broadcast_arrays(u, n, p)
    flat_u = u_arr.reshape(-1)
    flat_n = n_arr.reshape(-1)
    flat_p = p_arr.reshape(-1)
    flat_res = _binominvf_vmap(flat_u, flat_n, flat_p, exact_max, order)
    return flat_res.reshape(u_arr.shape)


@partial(jax.jit, static_argnames=["order", "exact_max", "dtype"])
def fast_approx_rbinom(
    key: Array,
    n: Array,
    p: Array,
    order: int = 2,
    exact_max: int = 5,
    dtype: np.dtype | None = None,
) -> Array:
    """
    Generate binomial random variables using a JAX implementation of the inverse incomplete beta function approximation.

    The implementation follows the methodology from Giles and Beentjes (2024). To more accurately handle cases where np is very small or the random draw is expected to be close to 0 or n, we apply the exact inverse CDF method in a manner similar to Giles (2016).

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
    assert (
        dtype is not None
    )  # Type guard: check_and_canonicalize_user_dtype only returns None if input is None
    if not (
        dtypes.issubdtype(dtype, np.floating) or dtypes.issubdtype(dtype, np.integer)
    ):
        raise ValueError(
            f"dtype argument to `fast_approx_rbinom` must be a float or integer dtype, got {dtype}"
        )

    # Get the dtype that JAX actually uses (may differ if jax_enable_x64=False)
    dtype = _get_available_dtype(dtype)
    assert (
        dtype is not None
    )  # Type guard: _get_available_dtype only returns None if input is None

    shape = jnp.broadcast_shapes(n.shape, p.shape)

    u = jax.random.uniform(key, shape)
    x = binominvf(u, n, p, exact_max, order=order)

    if jnp.issubdtype(dtype, jnp.integer):
        x = jnp.nan_to_num(x, nan=-1.0).astype(dtype)

    return x.astype(dtype)


@partial(jax.jit, static_argnames=["dtype"])
def fast_approx_rmultinom(
    key: Array, n: Array, p: Array, dtype: np.dtype | None = None
) -> Array:
    """
    Generate multinomial random variables using the inverse CDF method with fast_approx_rbinom.

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
            f"dtype argument to `fast_approx_rmultinom` must be a float or integer dtype, got {dtype}"
        )
    # Flatten inputs for broadcasting convenience
    n = jnp.asarray(n)
    p = jnp.asarray(p)
    # Handle shape checks
    if p.ndim < 1:
        raise ValueError("p must have at least 1 dimension (categories)")

    # p shape: (..., k), n shape: (...,)
    # Get shape as concrete Python values for batch_size computation
    p_shape = p.shape
    shape_batch = p_shape[:-1]
    num_cat = int(p_shape[-1])  # Convert to Python int

    # Compute batch_size as Python int (not JAX array)
    batch_size = math.prod(shape_batch) if shape_batch else 1

    # Broadcast n to match batch shape if needed
    n_broadcast = jnp.broadcast_to(n, shape_batch)

    # Normalize p so the last axis sums to 1
    p_sum = jnp.sum(p, axis=-1, keepdims=True)
    # Avoid division by zero: if p_sum == 0, set to 1 to avoid nans
    p_safe_sum = jnp.where(p_sum == 0, 1.0, p_sum)
    p = p / p_safe_sum

    def single_multinomial(key, n_i, p_i):
        """Sample a single multinomial row via sequential binomials."""
        # p_i: (k,)
        keys = jax.random.split(key, num_cat)
        n_remaining = n_i
        p_remain = jnp.array(1.0, dtype=p.dtype)
        out = []
        for j in range(num_cat - 1):
            p_cur = p_i[j] / p_remain
            p_cur = jnp.clip(p_cur, 0.0, 1.0)  # ensure numerically safe
            x = fast_approx_rbinom(
                keys[j], jnp.array(n_remaining), jnp.array(p_cur), dtype=dtype
            )
            out.append(x)
            n_remaining = n_remaining - x
            p_remain = p_remain - p_i[j]
        out.append(n_remaining)  # last category gets the remainder
        return jnp.stack(out, axis=-1)

    # Vectorize over leading dimensions
    sample_fn = jax.vmap(single_multinomial, in_axes=(0, 0, 0))

    # Split keys for each sample in the batch (batch_size must be Python int)
    keys = jax.random.split(key, batch_size)
    # Reshape the distributions for vectorization
    n_flat = n_broadcast.reshape((batch_size,))
    p_flat = p.reshape((batch_size, num_cat))
    samples = sample_fn(keys, n_flat, p_flat)

    return samples.reshape(shape_batch + (num_cat,)).astype(dtype)

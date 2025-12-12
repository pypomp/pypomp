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
from jax.scipy import special as jsp_special
from jax.scipy.stats import norm


def _newton_incomplete_beta(
    a: Array, b: Array, y: Array, x0: Array, max_iter: int = 10
) -> Array:
    """
    Refine initial guess for inverse incomplete beta using Newton iteration.

    Solves I_x(a, b) = y for x, where I_x is the regularized incomplete beta function.
    """
    dtype = x0.dtype

    x = jnp.clip(x0, jnp.float32(0.0), jnp.float32(1.0))

    for _ in range(max_iter):
        # Compute I_x(a, b) and its derivative
        # The derivative of I_x(a, b) with respect to x is:
        # x^(a-1) * (1-x)^(b-1) / B(a, b)
        ibeta = jsp_special.betainc(a, b, x)
        # PDF = x^(a-1) * (1-x)^(b-1) / B(a, b)
        # We can compute this using betainc derivative or directly
        log_pdf = (
            (a - 1) * jnp.log(jnp.maximum(x, jnp.finfo(dtype).tiny))
            + (b - 1) * jnp.log(jnp.maximum(1 - x, jnp.finfo(dtype).tiny))
            - jsp_special.betaln(a, b)
        )
        pdf = jnp.exp(log_pdf)

        # Newton step: x_new = x - (I_x(a,b) - y) / pdf
        f = ibeta - y
        x = x - f / jnp.maximum(pdf, jnp.finfo(dtype).tiny)

        # Clamp to valid range [0, 1]
        x = jnp.clip(x, jnp.float32(0.0), jnp.float32(1.0))

    return x


def _initial_guess_betaincinv(a: Array, b: Array, y: Array) -> Array:
    """
    Provide initial guess for inverse incomplete beta function.

    Uses approximation based on mean and mode of beta distribution.
    """
    dtype = a.dtype
    # Mean of beta distribution
    mean = a / (a + b)

    # Mode (when a > 1 and b > 1)
    mode = (a - jnp.float32(1.0)) / (a + b - jnp.float32(2.0))
    mode = jnp.where((a > jnp.float32(1.0)) & (b > jnp.float32(1.0)), mode, mean)

    # Use interpolation between 0, mean/mode, and 1 based on y
    x0 = jnp.where(
        y < jnp.float32(0.5),
        y * jnp.float32(2.0) * mode,
        jnp.float32(1.0)
        - (jnp.float32(1.0) - y) * jnp.float32(2.0) * (jnp.float32(1.0) - mode),
    )

    # Clamp to valid range
    x0 = jnp.clip(x0, jnp.finfo(dtype).tiny, jnp.float32(1.0) - jnp.finfo(dtype).tiny)
    return x0


def _betaincinv_scalar(a: Array, b: Array, y: Array) -> Array:
    """
    Compute inverse of regularized incomplete beta function I_x(a, b) = y.

    This is the core function that approximates the inverse incomplete beta.
    """
    dtype = jnp.float32
    a_arr = jnp.asarray(a, dtype=dtype)
    b_arr = jnp.asarray(b, dtype=dtype)
    y_arr = jnp.asarray(y, dtype=dtype)

    # Handle edge cases
    nan = jnp.array(jnp.nan, dtype=dtype)

    # Invalid inputs
    invalid_a = a_arr <= jnp.float32(0.0)
    invalid_b = b_arr <= jnp.float32(0.0)
    invalid_y = (y_arr < jnp.float32(0.0)) | (y_arr > jnp.float32(1.0))
    invalid = invalid_a | invalid_b | invalid_y

    # Edge cases for y
    y_is_zero = y_arr == jnp.float32(0.0)
    y_is_one = y_arr == jnp.float32(1.0)

    # Use safe values for computation
    a_safe = cast(Array, jnp.where(invalid_a, jnp.float32(1.0), a_arr))
    b_safe = cast(Array, jnp.where(invalid_b, jnp.float32(1.0), b_arr))
    y_safe = cast(Array, jnp.clip(y_arr, jnp.float32(0.0), jnp.float32(1.0)))

    # Initial approximation
    x0 = _initial_guess_betaincinv(a_safe, b_safe, y_safe)

    # Refine using Newton iteration
    x = _newton_incomplete_beta(a_safe, b_safe, y_safe, x0)

    # Handle edge cases
    x = cast(Array, jnp.where(invalid, nan, x))
    x = cast(Array, jnp.where(y_is_zero, jnp.float32(0.0), x))
    x = cast(Array, jnp.where(y_is_one, jnp.float32(1.0), x))

    return x


_betaincinv_vmap = jax.vmap(_betaincinv_scalar, in_axes=(0, 0, 0))


@jax.jit
def betaincinv_approx(a: Array, b: Array, y: Array) -> Array:
    """
    Vectorized inverse incomplete beta function approximation.

    Computes x such that I_x(a, b) = y, where I_x is the regularized
    incomplete beta function.

    Args:
        a: First shape parameter (must be positive).
        b: Second shape parameter (must be positive).
        y: Target value of the incomplete beta function (in [0, 1]).

    Returns:
        Array with the same broadcast shape as inputs, containing values
        in [0, 1] such that I_x(a, b) = y.
    """
    a_arr, b_arr, y_arr = jnp.broadcast_arrays(a, b, y)
    flat_a = a_arr.reshape(-1)
    flat_b = b_arr.reshape(-1)
    flat_y = y_arr.reshape(-1)
    flat_res = _betaincinv_vmap(flat_a, flat_b, flat_y)
    return flat_res.reshape(a_arr.shape)


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
    return (
        np_
        + sqrt_npq * w
        + (jnp.float32(2.0) + jnp.float32(2.0) * p + (q - p) * w2) / jnp.float32(6.0)
    )


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
    numerator_t2 = (jnp.float32(-2.0) + jnp.float32(14.0) * pq) * w + (
        jnp.float32(-1.0) - jnp.float32(2.0) * pq
    ) * w3
    denominator_t2 = jnp.float32(72.0) * sqrt_npq
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
    numerator_t3 = (
        (p - q)
        * (jnp.float32(2.0) + pq)
        * (jnp.float32(16.0) - jnp.float32(7.0) * w2 - jnp.float32(3.0) * w4)
    )
    denominator_t3 = jnp.float32(1620.0) * npq_
    term3 = numerator_t3 / jnp.maximum(denominator_t3, jnp.finfo(jnp.float32).tiny)
    return q_n1 + term3


def _binom_bottom_up(
    u: Array, n: Array, p: Array, approx: Array, max_k: int = 20
) -> Array:
    """
    Compute the exact inverse CDF for small k by accumulating the binomial CDF.
    Includes protection against float32 numerical stalling in the tail.
    """
    dtype = jnp.float32
    tiny = jnp.finfo(dtype).tiny
    epsilon = 1e-7  # Approx machine epsilon for float32

    q = jnp.clip(jnp.float32(1.0) - p, tiny, jnp.float32(1.0))
    p_safe = jnp.clip(p, jnp.float32(0.0), jnp.float32(1.0))
    q_safe = jnp.clip(q, tiny, jnp.float32(1.0))
    ratio_multiplier = p_safe / q_safe

    log_q = jnp.log(q_safe)
    pmf = jnp.where(
        n == jnp.float32(0.0),
        jnp.float32(1.0),
        jnp.exp(n * log_q),
    )
    cdf = pmf

    found = cdf >= u

    # Initialize result with approx (fallback), but if found at k=0, use 0.
    result = cast(Array, jnp.where(found, jnp.float32(0.0), approx))

    for i in range(1, max_k):
        k_curr_val = jnp.float32(i)
        k_curr = jnp.full_like(result, k_curr_val)
        k_prev = jnp.float32(i - 1)

        # Recurrence: P(k) = P(k-1) * ((n - k + 1) / k) * (p / q)
        num = jnp.maximum(n - k_prev, jnp.float32(0.0))
        den = k_curr_val

        pmf = pmf * (num / den) * ratio_multiplier
        cdf = cdf + pmf

        # Check for numerical stall:
        # If CDF is nearly 1.0 and PMF is negligible, the CDF won't increase further
        # in float32. We claim any remaining u values belong to this tail bucket.
        is_stalled = (cdf > 0.95) & (pmf < epsilon)

        found_now = (cdf >= u) | is_stalled

        is_new_discovery = jnp.logical_and(~found, found_now)
        result = cast(Array, jnp.where(is_new_discovery, k_curr, result))

        found = jnp.logical_or(found, found_now)

    return cast(Array, jnp.clip(result, jnp.float32(0.0), n))


def _binominvf_scalar(u: Array, n: Array, p: Array, order: int = 2) -> Array:
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

    invalid_n = n < jnp.float32(0.0)
    invalid_p = (p < jnp.float32(0.0)) | (p > jnp.float32(1.0))
    invalid_u = (u < jnp.float32(0.0)) | (u > jnp.float32(1.0))
    invalid = invalid_n | invalid_p | invalid_u

    # Special cases
    n_is_zero = n == jnp.float32(0.0)
    u_is_zero = u == jnp.float32(0.0)
    u_is_one = u == jnp.float32(1.0)
    p_is_zero = p == jnp.float32(0.0)
    p_is_one = p == jnp.float32(1.0)

    # For p = 0, all outcomes are 0
    # For p = 1, all outcomes are n
    # For u = 0, return 0
    # For u = 1, return n

    # Use symmetry: if p > 0.5, work with 1-p and flip result
    p_val = jnp.asarray(p, dtype=dtype)
    flip = p_val > jnp.float32(0.5)
    p_safe = cast(Array, jnp.where(flip, jnp.float32(1.0) - p_val, p_val))
    u_flipped = cast(Array, jnp.where(flip, jnp.float32(1.0) - u, u))

    n_safe = cast(Array, jnp.where(invalid_n, jnp.float32(1.0), n))
    p_safe = cast(Array, jnp.clip(p_safe, jnp.float32(0.0), jnp.float32(1.0)))
    u_safe = cast(
        Array,
        jnp.clip(u_flipped, jnp.float32(1e-9), jnp.float32(1.0) - jnp.float32(1e-9)),
    )  # Clip to avoid inf from norm.ppf

    # Pre-compute shared values for the approximation formulas
    q = jnp.float32(1.0) - p_safe
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
    order_idx = jnp.int32(safe_order)
    branches = [
        lambda x: _q_n0(*x),
        lambda x: _q_n1(*x),
        lambda x: _q_n2(*x),
    ]
    q_u = lax.switch(order_idx, branches, args)

    # The paper states \overline{C}^{-1}(u) = floor(C^{-1}(u))
    # Clip to valid range [0, n] and take floor
    k_approx = jnp.clip(jnp.floor(q_u), jnp.float32(0.0), n_safe)

    # Compute x from the bottom up if it is less than the cutoff
    cutoff = 5
    u_exact = jnp.clip(u_flipped, jnp.float32(0.0), jnp.float32(1.0))
    k_small = _binom_bottom_up(u_exact, n_safe, p_safe, k_approx, max_k=5)
    k_very_small = k_approx < cutoff
    np_very_small = np_ < 0.5
    use_bottom_up = k_very_small | np_very_small
    k_approx = cast(Array, jnp.where(use_bottom_up, k_small, k_approx))

    # Apply symmetry flip if needed
    k_flipped = cast(Array, jnp.where(flip, n_safe - k_approx, k_approx))

    k_result = k_flipped
    k_result = cast(Array, jnp.where(n_is_zero, jnp.float32(0.0), k_result))
    k_result = cast(Array, jnp.where(u_is_zero, jnp.float32(0.0), k_result))
    k_result = cast(Array, jnp.where(u_is_one, n_safe, k_result))
    k_result = cast(Array, jnp.where(p_is_zero, jnp.float32(0.0), k_result))
    k_result = cast(Array, jnp.where(p_is_one, n_safe, k_result))
    k_result = cast(Array, jnp.clip(k_result, jnp.float32(0.0), n_safe))

    k_result = cast(Array, jnp.where(invalid, nan, k_result))

    return k_result


_binominvf_vmap = jax.vmap(_binominvf_scalar, in_axes=(0, 0, 0, None))


@partial(jax.jit, static_argnames=["order"])
def binominvf(u: Array, n: Array, p: Array, order: int = 2) -> Array:
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
        order: Order of approximation (0, 1, or 2). Default is 2 for best accuracy.

    Returns:
        Array with the same broadcast shape as inputs, containing integer
        values k such that P(X <= k) >= u.
    """
    u_arr, n_arr, p_arr = jnp.broadcast_arrays(u, n, p)
    flat_u = u_arr.reshape(-1)
    flat_n = n_arr.reshape(-1)
    flat_p = p_arr.reshape(-1)
    flat_res = _binominvf_vmap(flat_u, flat_n, flat_p, order)
    return flat_res.reshape(u_arr.shape)


@partial(jax.jit, static_argnames=["order", "dtype"])
def fast_approx_rbinom(
    key: Array, n: Array, p: Array, order: int = 2, dtype: jnp.dtype = jnp.float32
) -> Array:
    """
    Generate binomial random variables using a JAX implementation of the inverse incomplete beta function approximation.

    The implementation follows the methodology from Giles and Beentjes (2024). To more accurately handle cases where np is very small or the random draw is expected to be close to 0 or n, we apply the exact inverse CDF method in a manner similar to Giles (2016).

    Args:
        key: PRNG key used as the random key.
        n: Number of trials for the binomial distribution.
        p: Success probability for the binomial distribution.
        order: Order of approximation (0, 1, or 2). Default is 2 for best accuracy.
        dtype: Data type of the output. Default is jnp.float32. If integer, returns -1 for invalid inputs instead of nan.

    Returns:
        Binomial random variables with the same shape as n and p.

    References:
        * Giles, Michael B., and Casper Beentjes. “Approximation of an Inverse of the Incomplete Beta Function.” In Mathematical Software – ICMS 2024, vol. 14749. Lecture Notes in Computer Science. Springer Nature Switzerland, 2024. https://doi.org/10.1007/978-3-031-64529-7_22.
        * Giles, Michael B. “Algorithm 955: Approximation of the Inverse Poisson Cumulative Distribution Function.” ACM Transactions on Mathematical Software 42, no. 1 (2016): 1–22. https://doi.org/10.1145/2699466.
    """
    shape = jnp.broadcast_shapes(n.shape, p.shape)

    u = jax.random.uniform(key, shape)
    x = binominvf(u, n, p, order=order)

    if jnp.issubdtype(dtype, jnp.integer):
        x = jnp.nan_to_num(x, nan=-1.0).astype(dtype)

    return x.astype(dtype)


@partial(jax.jit, static_argnames=["dtype"])
def fast_approx_rmultinom(
    key: Array, n: Array, p: Array, dtype: jnp.dtype = jnp.float32
) -> Array:
    """
    Generate multinomial random variables using the inverse CDF method with fast_approx_rbinom.

    Args:
        key: PRNG key used as the random key.
        n: Number of trials for the multinomial distribution. Shape: (...,)
        p: Probabilities for each category. Shape: (..., k), where k = num categories.
           Probabilities along the last axis must sum to 1.
        dtype: Data type of the output. Default is jnp.float32. If integer, returns -1 for invalid inputs instead of nan.
    Returns:
        Multinomial counts. Same shape as p, but dtype = n.dtype.
    """
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

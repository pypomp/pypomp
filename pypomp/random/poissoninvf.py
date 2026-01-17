"""
JAX implementation of the single-precision inverse Poisson CDF approximation.

The implementation ports NVIDIA's CURAND `poissinvf` CUDA device routine to
Python so it can be composed with `jax.jit`/`jax.vmap`.  The structure matches
the original algorithm: central-region polynomial approximation, Newton
iteration fallback, and a final bottom-up / top-down summation when the rate is
small.
"""

from __future__ import annotations

from typing import Tuple, cast

import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy import special as jsp_special

_RM_COEFFS: Tuple[float, ...] = (
    2.82298751e-07,
    -2.58136133e-06,
    1.02118025e-05,
    -2.37996199e-05,
    4.05347462e-05,
    -6.63730967e-05,
    1.24762566e-04,
    -2.56970731e-04,
    5.58953132e-04,
    -1.33129194e-03,
    3.70367937e-03,
    -1.38888706e-02,
    1.66666667e-01,
)

_T_COEFFS: Tuple[float, ...] = (
    1.86386867e-05,
    -2.07319499e-04,
    9.68945100e-04,
    -2.47340054e-03,
    3.79952985e-03,
    -3.86717047e-03,
    3.46960934e-03,
    -4.14125511e-03,
    5.86752093e-03,
    -8.38583787e-03,
    1.32793933e-02,
    -2.77755360e-02,
    3.33333333e-01,
)

_X_COEFFS: Tuple[float, ...] = (
    -1.45852240e-04,
    1.46121529e-03,
    -6.10328845e-03,
    1.38117964e-02,
    -1.86988746e-02,
    1.68155118e-02,
    -1.33947970e-02,
    1.35698573e-02,
    -1.55377333e-02,
    1.74065334e-02,
    -1.98011178e-02,
)

_SQRT2 = jnp.sqrt(jnp.float32(2.0))

_RM_COEFFS_ARR = jnp.array(_RM_COEFFS, dtype=jnp.float32)
_T_COEFFS_ARR = jnp.array(_T_COEFFS, dtype=jnp.float32)
_X_COEFFS_ARR = jnp.array(_X_COEFFS, dtype=jnp.float32)


def _central_region(s: Array, lam: Array) -> Array:
    rm = jnp.polyval(_RM_COEFFS_ARR, s)
    rm = s + s * (rm * s)

    t = jnp.polyval(_T_COEFFS_ARR, rm)
    x = jnp.polyval(_X_COEFFS_ARR, rm) / lam

    total = lam + (x + t) + lam * rm
    return jnp.floor(total)


def _newton_region(s: Array, lam: Array) -> Array:
    dtype = s.dtype

    MAX_LOOPS = 5
    r = jnp.maximum(0.1, 1.0 + s)
    r_prev = r
    first = jnp.array(True, dtype=jnp.bool_)
    counter = 0

    for _ in range(MAX_LOOPS):
        diff = jnp.abs(r - r_prev)
        not_done = jnp.logical_or(first, diff > 1e-5)
        not_max_loops = counter < MAX_LOOPS
        keep_going = jnp.logical_and(not_done, not_max_loops)

        t = jnp.log(r)
        s2 = jnp.sqrt(2.0 * ((1.0 - r) + r * t))
        s2 = jnp.where(r < 1.0, -s2, s2)
        next_r = r - (s2 - s) * s2 / t
        next_r = jnp.maximum(next_r, 0.1 * r)

        # Only update variables if condition is True
        r_new = jnp.where(keep_going, next_r, r)
        r_prev_new = jnp.where(keep_going, r, r_prev)
        first_new = jnp.array(False, dtype=jnp.bool_)
        counter_new = counter + 1

        r, r_prev, first, counter = r_new, r_prev_new, first_new, counter_new

    t = jnp.log(r)
    sqrt_term = jnp.sqrt(2.0 * r * ((1.0 - r) + r * t))
    log_correction = jnp.log(
        sqrt_term / jnp.maximum(jnp.abs(r - 1.0), jnp.finfo(dtype).tiny)
    )
    x = lam * r + log_correction / t
    x -= 0.0218 / (x + 0.065 * lam)
    return jnp.floor(x)


def _bottom_up(u: Array, lam: Array) -> Array:
    lami = 1.0 / lam

    t0 = jnp.exp(0.5 * lam)
    del0 = jnp.where(u > 0.5, t0 * (1e-6 * t0), 0.0)
    s0 = 1.0 - t0 * (u * t0) + del0

    def unrolled_computation(x_init, s0, del0, lami):
        MAX_LOOPS = 20

        # Initialize state
        x, s, delta = x_init, s0, del0
        t = jnp.float32(0.0)

        # Track if we are still running (equivalent to cond1)
        active = jnp.array(True)

        # JAX will unroll this loop during compilation
        for _ in range(MAX_LOOPS):
            # Check condition: s < 0.0
            current_cond = s < jnp.float32(0.0)

            # Determine if we should update in this step
            # We continue only if we were already active AND the condition holds
            keep_going = jnp.logical_and(active, current_cond)

            # Calculate candidates for next step
            x_next = x + jnp.float32(1.0)
            t_next = x_next * lami
            delta_next = t_next * delta
            s_next = t_next * s + jnp.float32(1.0)

            # Apply updates only if keep_going is True
            x = jnp.where(keep_going, x_next, x)
            s = jnp.where(keep_going, s_next, s)
            delta = jnp.where(keep_going, delta_next, delta)
            t = jnp.where(keep_going, t_next, t)

            # Update the active flag (once it turns False, it stays False)
            active = keep_going

        return x, s, delta

    x_init = 0.0
    x, s, delta = unrolled_computation(x_init, s0, del0, lami)

    def top_down_branch(state):
        x_val, delta_val = state
        # Setup
        delta_scaled = 1e6 * delta_val
        t_thresh = 1e7 * delta_scaled
        delta_scaled = (1.0 - u) * delta_scaled

        # Unrolled first loop (finding x_hi, delta_hi)
        MAX_LOOPS_2 = 20
        x_hi = x_val
        delta_hi = delta_scaled
        for _ in range(MAX_LOOPS_2):
            cond = delta_hi < t_thresh
            x_next = x_hi + 1.0
            delta_next = delta_hi * (x_next * lami)
            x_hi = jnp.where(cond, x_next, x_hi)
            delta_hi = jnp.where(cond, delta_next, delta_hi)

        # Unrolled second loop (finding x_lo)
        MAX_LOOPS_3 = 20
        x_lo = x_hi
        s_lo = delta_hi
        t_lo = jnp.float32(1.0)
        for _ in range(MAX_LOOPS_3):
            cond = s_lo > 0.0
            t_next = t_lo * (x_lo * lami)
            s_next = s_lo - t_next
            x_next = x_lo - 1.0
            x_lo = jnp.where(cond, x_next, x_lo)
            s_lo = jnp.where(cond, s_next, s_lo)
            t_lo = jnp.where(cond, t_next, t_lo)
        return x_lo

    return lax.cond(
        s < jnp.float32(2.0) * delta,
        top_down_branch,
        lambda state: state[0],
        operand=(x, delta),
    )


def _poissinvf_scalar(u: Array, lam: Array) -> Array:
    dtype = jnp.float32
    u = jnp.asarray(u, dtype=dtype)
    lam = jnp.asarray(lam, dtype=dtype)

    x0 = 0.0

    lam_invalid = lam <= 0.0
    lam_safe = cast(Array, jnp.where(lam_invalid, 1.0, lam))

    def large_lambda_case(_):
        s = jsp_special.ndtri(u) * lax.rsqrt(lam_safe)

        def central(_):
            return _central_region(s, lam_safe)

        def non_central(_):
            return lax.cond(
                s > -_SQRT2,
                lambda __: _newton_region(s, lam_safe),
                lambda __: x0,
                operand=0.0,
            )

        return lax.cond(
            jnp.logical_and(s > -0.6833501, s < 1.777993),
            central,
            non_central,
            operand=0.0,
        )

    large_lambda = lam_safe > 4.0
    x_large = lax.cond(
        large_lambda,
        large_lambda_case,
        lambda _: x0,
        operand=0.0,
    )

    def bottom_up_branch(_):
        return _bottom_up(u, lam_safe)

    bottom_up = x_large <= 10.0
    # not_large_bottom_up = jnp.logical_and(jnp.logical_not(large_lambda), bottom_up)
    # x = x_large
    # x = lax.cond(
    #     not_large_bottom_up,
    #     bottom_up_branch,
    #     lambda _: x_large,
    #     operand=jnp.float32(0.0),
    # )
    x = lax.cond(
        bottom_up,
        bottom_up_branch,
        lambda _: x_large,
        operand=0.0,
    )

    nan = jnp.array(jnp.nan, dtype=dtype)
    inf = jnp.array(jnp.inf, dtype=dtype)

    x = cast(Array, jnp.where(u < 0.0, nan, x))
    x = cast(Array, jnp.where(u == 0.0, 0.0, x))
    x = cast(Array, jnp.where(u == 1.0, inf, x))
    x = cast(Array, jnp.where(u > 1.0, nan, x))
    x = cast(Array, jnp.where(lam_invalid, nan, x))
    x = cast(Array, jnp.where(x < 0.0, 0.0, x))
    return x


_poissinvf_vmap = jax.vmap(_poissinvf_scalar)


@jax.jit
def poissinvf(u: Array, lam: Array) -> Array:
    """
    Vectorized inverse Poisson CDF approximation using JAX primitives.

    Args:
        u: Probabilities (scalar or array) in the interval [0, 1].
        lam: Corresponding Poisson rate(s), must be positive.

    Returns:
        DeviceArray with the same broadcast shape as `u` and `lam`.
    """

    u_arr, lam_arr = jnp.broadcast_arrays(u, lam)
    flat_u = u_arr.reshape(-1)
    flat_lam = lam_arr.reshape(-1)
    flat_res = _poissinvf_vmap(flat_u, flat_lam)
    return flat_res.reshape(u_arr.shape)


@jax.jit
def fast_approx_rpoisson(key: Array, lam: Array) -> Array:
    """
    Generate a Poisson random variable with given rate parameter.

    Follows the methodology from Giles (2016). We made some ad-hoc modifications to the algorithm to improve the speed. In particular, we put a cap on how many iterations the Newton-Raphson method and the exact inverse CDF method can take, and we adjusted the thresholds for applying the exact inverse CDF method.

    Args:
        key: a PRNG key used as the random key.
        lam: rate parameters for the Poisson distribution.

    Returns:
        A Poisson random variable.

    References:
        * Giles, Michael B. “Algorithm 955: Approximation of the Inverse Poisson Cumulative Distribution Function.” ACM Transactions on Mathematical Software 42, no. 1 (2016): 1–22. https://doi.org/10.1145/2699466.
    """
    shape = lam.shape
    u = jax.random.uniform(key, shape)
    # Clamp u to be slightly less than 1.0 to avoid inf output
    # Use nextafter to get the largest float < 1.0
    u_max = jnp.nextafter(jnp.array(1.0, dtype=u.dtype), jnp.array(0.0, dtype=u.dtype))
    u = jnp.minimum(u, u_max)
    x = poissinvf(u, lam)
    # Cap the output to a reasonable maximum to prevent overflow
    max_val = lam + 10.0 * jnp.sqrt(jnp.maximum(lam, 1.0))
    x = jnp.minimum(x, max_val)
    return x.astype(lam.dtype)

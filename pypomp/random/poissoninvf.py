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
from functools import partial

import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy import special as jsp_special
import numpy as np
from jax._src import dtypes

from ._dtype_helpers import check_and_canonicalize_user_dtype, _get_available_dtype


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

# These coefficient arrays are kept at float32 for efficiency
# They will be cast to the appropriate dtype during computation
_RM_COEFFS_ARR = jnp.array(_RM_COEFFS, dtype=jnp.float32)
_T_COEFFS_ARR = jnp.array(_T_COEFFS, dtype=jnp.float32)
_X_COEFFS_ARR = jnp.array(_X_COEFFS, dtype=jnp.float32)


def _central_region(s: Array, lam: Array, dtype) -> Array:
    # Cast coefficients to the working dtype
    rm_coeffs = _RM_COEFFS_ARR.astype(dtype)
    t_coeffs = _T_COEFFS_ARR.astype(dtype)
    x_coeffs = _X_COEFFS_ARR.astype(dtype)

    rm = jnp.polyval(rm_coeffs, s)
    rm = s + s * (rm * s)

    t = jnp.polyval(t_coeffs, rm)
    x = jnp.polyval(x_coeffs, rm) / lam

    total = lam + (x + t) + lam * rm
    return jnp.floor(total)


def _newton_region(s: Array, lam: Array, dtype) -> Array:
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


def _bottom_up(u: Array, lam: Array, dtype) -> Array:
    lami = 1.0 / lam

    t0 = jnp.exp(0.5 * lam)
    del0 = jnp.where(u > 0.5, t0 * (1e-6 * t0), 0.0)
    s0 = 1.0 - t0 * (u * t0) + del0

    def unrolled_computation(x_init, s0, del0, lami) -> Tuple[Array, Array, Array]:
        MAX_LOOPS = 20

        # Initialize state
        x, s, delta = x_init, s0, del0
        t = jnp.array(0.0, dtype=dtype)
        zero = jnp.array(0.0, dtype=dtype)
        one = jnp.array(1.0, dtype=dtype)

        # Track if we are still running (equivalent to cond1)
        active = jnp.array(True)

        # JAX will unroll this loop during compilation
        for _ in range(MAX_LOOPS):
            # Check condition: s < 0.0
            current_cond = s < zero

            # Determine if we should update in this step
            # We continue only if we were already active AND the condition holds
            keep_going = jnp.logical_and(active, current_cond)

            # Calculate candidates for next step
            x_next = x + one
            t_next = x_next * lami
            delta_next = t_next * delta
            s_next = t_next * s + one

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
    x = cast(Array, x)
    s = cast(Array, s)
    delta = cast(Array, delta)

    def top_down_branch(state):
        x_val, delta_val = state
        one = jnp.array(1.0, dtype=dtype)
        zero = jnp.array(0.0, dtype=dtype)

        # Setup
        delta_scaled = jnp.array(1e6, dtype=dtype) * delta_val
        t_thresh = jnp.array(1e7, dtype=dtype) * delta_scaled
        delta_scaled = (one - u) * delta_scaled

        # Unrolled first loop (finding x_hi, delta_hi)
        MAX_LOOPS_2 = 20
        x_hi = x_val
        delta_hi = delta_scaled
        for _ in range(MAX_LOOPS_2):
            cond = delta_hi < t_thresh
            x_next = x_hi + one
            delta_next = delta_hi * (x_next * lami)
            x_hi = jnp.where(cond, x_next, x_hi)
            delta_hi = jnp.where(cond, delta_next, delta_hi)

        # Unrolled second loop (finding x_lo)
        MAX_LOOPS_3 = 20
        x_lo = x_hi
        s_lo = delta_hi
        t_lo = one
        for _ in range(MAX_LOOPS_3):
            cond = s_lo > zero
            t_next = cast(Array, t_lo * (x_lo * lami))
            s_next = cast(Array, s_lo - t_next)
            x_next = cast(Array, x_lo - one)
            x_lo = cast(Array, jnp.where(cond, x_next, x_lo))
            s_lo = cast(Array, jnp.where(cond, s_next, s_lo))
            t_lo = cast(Array, jnp.where(cond, t_next, t_lo))
        return x_lo

    two = jnp.array(2.0, dtype=dtype)
    return lax.cond(
        s < two * delta,
        top_down_branch,
        lambda state: state[0],
        operand=(x, delta),
    )


def _poissinvf_scalar(u: Array, lam: Array, dtype) -> Array:
    u = jnp.asarray(u, dtype=dtype)
    lam = jnp.asarray(lam, dtype=dtype)

    zero = jnp.array(0.0, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    x0 = zero
    sqrt2 = jnp.sqrt(jnp.array(2.0, dtype=dtype))

    lam_invalid = lam <= zero
    lam_safe = cast(Array, jnp.where(lam_invalid, one, lam))

    def large_lambda_case(_):
        s = jsp_special.ndtri(u) * lax.rsqrt(lam_safe)

        def central(_):
            return _central_region(s, lam_safe, dtype)

        def non_central(_):
            return lax.cond(
                s > -sqrt2,
                lambda __: _newton_region(s, lam_safe, dtype),
                lambda __: x0,
                operand=zero,
            )

        return lax.cond(
            jnp.logical_and(
                s > jnp.array(-0.6833501, dtype=dtype),
                s < jnp.array(1.777993, dtype=dtype),
            ),
            central,
            non_central,
            operand=zero,
        )

    large_lambda = lam_safe > jnp.array(4.0, dtype=dtype)
    x_large = lax.cond(
        large_lambda,
        large_lambda_case,
        lambda _: x0,
        operand=zero,
    )

    def bottom_up_branch(_):
        return _bottom_up(u, lam_safe, dtype)

    bottom_up = x_large <= jnp.array(10.0, dtype=dtype)
    x = lax.cond(
        bottom_up,
        bottom_up_branch,
        lambda _: x_large,
        operand=zero,
    )

    nan = jnp.array(jnp.nan, dtype=dtype)
    inf = jnp.array(jnp.inf, dtype=dtype)

    x = cast(Array, jnp.where(u < zero, nan, x))
    x = cast(Array, jnp.where(u == zero, zero, x))
    x = cast(Array, jnp.where(u == one, inf, x))
    x = cast(Array, jnp.where(u > one, nan, x))
    x = cast(Array, jnp.where(lam_invalid, nan, x))
    x = cast(Array, jnp.where(x < zero, zero, x))
    return x


_poissinvf_vmap = jax.vmap(_poissinvf_scalar, in_axes=(0, 0, None))


@partial(jax.jit, static_argnames=["dtype"])
def poissinvf(u: Array, lam: Array, dtype=jnp.float32) -> Array:
    """
    Vectorized inverse Poisson CDF approximation using JAX primitives.

    Args:
        u: Probabilities (scalar or array) in the interval [0, 1].
        lam: Corresponding Poisson rate(s), must be positive.
        dtype: Data type for the computation (default float32).

    Returns:
        DeviceArray with the same broadcast shape as `u` and `lam`.
    """

    u_arr, lam_arr = jnp.broadcast_arrays(u, lam)
    flat_u = u_arr.reshape(-1)
    flat_lam = lam_arr.reshape(-1)
    flat_res = _poissinvf_vmap(flat_u, flat_lam, dtype)
    return flat_res.reshape(u_arr.shape)


def fast_approx_rpoisson(
    key: Array, lam: Array, dtype: np.dtype | None = None
) -> Array:
    """
    Generate a Poisson random variable with given rate parameter.

    Follows the methodology from Giles (2016). We made some ad-hoc modifications to the algorithm to improve the speed. In particular, we put a cap on how many iterations the Newton-Raphson method and the exact inverse CDF method can take, and we adjusted the thresholds for applying the exact inverse CDF method.

    Args:
        key: a PRNG key used as the random key.
        lam: rate parameters for the Poisson distribution.
        dtype: optional, an integer dtype for the returned values (default int64 if
            jax_enable_x64 is true, otherwise int32).

    Returns:
        A Poisson random variable.

    References:
        * Giles, Michael B. "Algorithm 955: Approximation of the Inverse Poisson Cumulative Distribution Function." ACM Transactions on Mathematical Software 42, no. 1 (2016): 1–22. https://doi.org/10.1145/2699466.
    """
    dtype = check_and_canonicalize_user_dtype(int if dtype is None else dtype)
    assert dtype is not None
    if not dtypes.issubdtype(dtype, np.integer):
        raise ValueError(
            f"dtype argument to `fast_approx_rpoisson` must be an integer dtype, got {dtype}"
        )

    # Get the dtype that JAX actually uses (may differ if jax_enable_x64=False)
    dtype = _get_available_dtype(dtype)
    assert dtype is not None

    # Determine the appropriate float dtype for internal computations
    # Use float64 if the integer dtype is 64-bit, otherwise float32
    if dtypes.issubdtype(dtype, np.int64):
        float_dtype = jnp.float64
    else:
        float_dtype = jnp.float32

    # Get the float dtype that JAX actually uses
    float_dtype = _get_available_dtype(float_dtype)
    assert float_dtype is not None

    lam = jnp.asarray(lam)
    shape = lam.shape
    u = jax.random.uniform(key, shape, dtype=float_dtype)
    # Clamp u to be slightly less than 1.0 to avoid inf output
    # Use nextafter to get the largest float < 1.0
    u_max = jnp.nextafter(
        jnp.array(1.0, dtype=float_dtype), jnp.array(0.0, dtype=float_dtype)
    )
    u = jnp.minimum(u, u_max)
    lam_float = lam.astype(float_dtype)
    x = poissinvf(u, lam_float, dtype=float_dtype)
    # Cap the output to a reasonable maximum to prevent overflow
    max_val = lam_float + jnp.array(10.0, dtype=float_dtype) * jnp.sqrt(
        jnp.maximum(lam_float, jnp.array(1.0, dtype=float_dtype))
    )
    x = jnp.minimum(x, max_val)
    return x.astype(dtype)

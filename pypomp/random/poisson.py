"""
JAX implementation of the single-precision inverse Poisson CDF approximation.

The implementation ports NVIDIA's CURAND `poissinvf` CUDA device routine to
Python so it can be composed with `jax.jit`/`jax.vmap`.  The structure matches
the original algorithm: central-region polynomial approximation, Newton
iteration fallback, and a final bottom-up / top-down summation when the rate is
small.
"""

from __future__ import annotations

from typing import Tuple, Any
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

# These will be cast to the appropriate dtype during computation
_RM_COEFFS_ARR = np.array(_RM_COEFFS, dtype=np.float64)
_T_COEFFS_ARR = np.array(_T_COEFFS, dtype=np.float64)
_X_COEFFS_ARR = np.array(_X_COEFFS, dtype=np.float64)


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


def _newton_region(s: Array, lam: Array, dtype, max_newton_loops: int) -> Array:
    r = jnp.maximum(0.1, 1.0 + s)
    r_prev = r
    first = jnp.array(True, dtype=jnp.bool_)
    counter = 0

    for _ in range(max_newton_loops):
        diff = jnp.abs(r - r_prev)
        not_done = jnp.logical_or(first, diff > 1e-5)
        not_max_loops = counter < max_newton_loops
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


def _bottom_up(u: Array, lam: Array, dtype, max_inverse_cdf_loops: int) -> Array:
    lami = 1.0 / lam

    t0 = jnp.exp(0.5 * lam)
    del0 = jnp.where(u > 0.5, t0 * (1e-6 * t0), 0.0)
    s0 = 1.0 - t0 * (u * t0) + del0

    def _find_quantile(
        x_init: Array, s0: Array, del0: Array, lami: Array
    ) -> Tuple[Array, Array, Array]:

        x = x_init
        s = s0
        delta = del0
        t = jnp.array(0.0, dtype=dtype)
        zero = jnp.array(0.0, dtype=dtype)
        one = jnp.array(1.0, dtype=dtype)

        active = jnp.array(True)

        for _ in range(max_inverse_cdf_loops):
            current_cond = s < zero

            keep_going = jnp.logical_and(active, current_cond)

            x_next = x + one
            t_next = x_next * lami
            delta_next = t_next * delta
            s_next = t_next * s + one

            x = jax.lax.select(keep_going, x_next, x)
            s = jax.lax.select(keep_going, s_next, s)
            delta = jax.lax.select(keep_going, delta_next, delta)
            t = jax.lax.select(keep_going, t_next, t)

            active = keep_going

        return x, s, delta

    x_init = jnp.array(0.0, dtype=dtype)
    x, s, delta = _find_quantile(x_init, s0, del0, lami)

    def _top_down_branch(state: Tuple[Array, Array]) -> Array:
        x_val, delta_val = state
        one = jnp.array(1.0, dtype=dtype)
        zero = jnp.array(0.0, dtype=dtype)

        delta_scaled = jnp.array(1e6, dtype=dtype) * delta_val
        t_thresh = jnp.array(1e7, dtype=dtype) * delta_scaled
        delta_scaled = (one - u) * delta_scaled

        x_hi = x_val
        delta_hi = delta_scaled
        for _ in range(max_inverse_cdf_loops):
            cond = delta_hi < t_thresh
            x_next = x_hi + one
            delta_next = delta_hi * (x_next * lami)
            x_hi = jnp.where(cond, x_next, x_hi)
            delta_hi = jnp.where(cond, delta_next, delta_hi)

        x_lo = x_hi
        s_lo = delta_hi
        t_lo = one
        for _ in range(max_inverse_cdf_loops):
            cond = s_lo > zero
            t_next = t_lo * (x_lo * lami)
            s_next = s_lo - t_next
            x_next = x_lo - one
            x_lo = jnp.where(cond, x_next, x_lo)
            s_lo = jnp.where(cond, s_next, s_lo)
            t_lo = jnp.where(cond, t_next, t_lo)
        return x_lo

    two = jnp.array(2.0, dtype=dtype)
    return lax.cond(
        s < two * delta,
        _top_down_branch,
        lambda state: state[0],
        operand=(x, delta),
    )


def _poissoninv_scalar(
    u: Array,
    lam: Array,
    dtype,
    max_newton_loops: int = 5,
    max_inverse_cdf_loops: int = 20,
) -> Array:
    u = jnp.asarray(u, dtype=dtype)
    lam = jnp.asarray(lam, dtype=dtype)

    zero = jnp.array(0.0, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    x0 = zero
    sqrt2 = jnp.sqrt(jnp.array(2.0, dtype=dtype))

    lam_invalid = lam <= zero
    lam_safe = jnp.where(lam_invalid, one, lam)

    def large_lambda_case(_: Any) -> Array:
        s = jsp_special.ndtri(u) * lax.rsqrt(lam_safe)

        def central(_: Any) -> Array:
            return _central_region(s, lam_safe, dtype)

        def non_central(_: Any) -> Array:
            return lax.cond(
                s > -sqrt2,
                lambda __: _newton_region(s, lam_safe, dtype, max_newton_loops),
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
    x_large: Array = lax.cond(
        large_lambda,
        large_lambda_case,
        lambda _: x0,
        operand=zero,
    )

    def bottom_up_branch(_: Any) -> Array:
        return _bottom_up(u, lam_safe, dtype, max_inverse_cdf_loops)

    bottom_up = x_large <= jnp.array(10.0, dtype=dtype)
    x: Array = lax.cond(
        bottom_up,
        bottom_up_branch,
        lambda _: x_large,
        operand=zero,
    )

    nan = jnp.array(jnp.nan, dtype=dtype)
    inf = jnp.array(jnp.inf, dtype=dtype)

    x = jnp.where(u < zero, nan, x)
    x = jnp.where(u == zero, zero, x)
    x = jnp.where(u == one, inf, x)
    x = jnp.where(u > one, nan, x)
    x = jnp.where(lam_invalid, nan, x)
    x = jnp.where(x < zero, zero, x)
    return x


_poissoninv_vmap = jax.vmap(_poissoninv_scalar, in_axes=(0, 0, None, None, None))


@partial(
    jax.jit, static_argnames=["dtype", "max_newton_loops", "max_inverse_cdf_loops"]
)
def poissoninv(
    u: Array,
    lam: Array,
    dtype: np.dtype | None = None,
    max_newton_loops: int = 5,
    max_inverse_cdf_loops: int = 20,
) -> Array:
    """
    Vectorized inverse Poisson CDF approximation using JAX primitives.

    Args:
        u: Probabilities (scalar or array) in the interval [0, 1].
        lam: Corresponding Poisson rate(s), must be positive.
        dtype: Data type for the computation and return value.
        max_newton_loops: Cap on iterations for the Newton-Raphson method.
        max_inverse_cdf_loops: Cap on iterations for the exact inverse CDF method.

    Returns:
        DeviceArray with the same broadcast shape as `u` and `lam`.
    """

    u_arr, lam_arr = jnp.broadcast_arrays(u, lam)
    if dtype is None:
        dtype = jnp.result_type(u_arr, lam_arr)
        if not dtypes.issubdtype(dtype, np.floating):
            dtype = jnp.float32

    dtype = check_and_canonicalize_user_dtype(dtype)
    assert dtype is not None
    if not (
        dtypes.issubdtype(dtype, np.floating) or dtypes.issubdtype(dtype, np.integer)
    ):
        raise ValueError(
            f"dtype argument to `poissoninv` must be a float or integer dtype, got {dtype}"
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

    if u_arr.ndim == 0:
        res = _poissoninv_scalar(
            u_arr, lam_arr, float_dtype, max_newton_loops, max_inverse_cdf_loops
        )
    else:
        flat_u = u_arr.reshape(-1)
        flat_lam = lam_arr.reshape(-1)
        flat_res = _poissoninv_vmap(
            flat_u, flat_lam, float_dtype, max_newton_loops, max_inverse_cdf_loops
        )
        res = flat_res.reshape(u_arr.shape)

    if dtypes.issubdtype(dtype, np.integer):
        res = jnp.where(jnp.isnan(res) | jnp.isinf(res), -1.0, res)
        return res.astype(dtype)
    return res.astype(dtype)


@partial(
    jax.jit, static_argnames=["dtype", "max_newton_loops", "max_inverse_cdf_loops"]
)
def fast_poisson(
    key: Array,
    lam: Array,
    dtype: np.dtype | None = None,
    max_newton_loops: int = 5,
    max_inverse_cdf_loops: int = 20,
) -> Array:
    """
    Generate a Poisson random variable with given rate parameter using an approximate inverse CDF method in order to run fast on GPUs.

    Follows the methodology from Giles (2016). We made some ad-hoc modifications to the algorithm to improve its speed. In particular, we put a cap on how many iterations the Newton-Raphson method and the exact inverse CDF method can take, and we adjusted the thresholds for applying the exact inverse CDF method. Our implementation of the method does not produce exact Poisson random variables, but it is very close to exact.

    Args:
        key: a PRNG key used as the random key.
        lam: rate parameters for the Poisson distribution.
        dtype: optional, an integer dtype for the returned values (default int64 if
            jax_enable_x64 is true, otherwise int32).
        max_newton_loops: Cap on iterations for the Newton-Raphson method.
        max_inverse_cdf_loops: Cap on iterations for the exact inverse CDF method.

    Returns:
        A Poisson random variable.

    References:
        * Giles, Michael B. "Algorithm 955: Approximation of the Inverse Poisson Cumulative Distribution Function." ACM Transactions on Mathematical Software 42, no. 1 (2016): 1–22. https://doi.org/10.1145/2699466.
    """
    dtype = check_and_canonicalize_user_dtype(int if dtype is None else dtype)
    assert dtype is not None
    if not dtypes.issubdtype(dtype, np.integer):
        raise ValueError(
            f"dtype argument to `fast_poisson` must be an integer dtype, got {dtype}"
        )

    dtype = _get_available_dtype(dtype)
    assert dtype is not None

    if dtypes.issubdtype(dtype, np.int64):
        float_dtype = jnp.float64
    else:
        float_dtype = jnp.float32

    float_dtype = _get_available_dtype(float_dtype)
    assert float_dtype is not None

    lam = jnp.asarray(lam)
    lam_float = lam.astype(float_dtype)
    invalid = lam_float < 0.0

    shape = lam.shape
    u = jax.random.uniform(key, shape, dtype=float_dtype)
    # Clamp u to be slightly less than 1.0 to avoid inf output
    # Use nextafter to get the largest float < 1.0
    u_max = jnp.nextafter(
        jnp.array(1.0, dtype=float_dtype), jnp.array(0.0, dtype=float_dtype)
    )
    u = jnp.minimum(u, u_max)
    x = poissoninv(
        u,
        lam_float,
        dtype=float_dtype,
        max_newton_loops=max_newton_loops,
        max_inverse_cdf_loops=max_inverse_cdf_loops,
    )
    # Cap the output to a reasonable maximum to prevent overflow
    max_val = lam_float + jnp.array(10.0, dtype=float_dtype) * jnp.sqrt(
        jnp.maximum(lam_float, jnp.array(1.0, dtype=float_dtype))
    )
    x = jnp.minimum(x, max_val)
    # For integer dtype, follow the convention of returning -1 for invalid inputs
    return jnp.where(invalid, -1, x.astype(dtype))

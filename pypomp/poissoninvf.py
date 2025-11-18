"""
JAX implementation of the single-precision inverse Poisson CDF approximation.

The implementation ports NVIDIA's CURAND `poissinvf` CUDA device routine to
Python so it can be composed with `jax.jit`/`jax.vmap`.  The structure matches
the original algorithm: central-region polynomial approximation, Newton
iteration fallback, and a final bottom-up / top-down summation when the rate is
small.
"""

from __future__ import annotations

from typing import Sequence, Tuple, cast

import jax
from jax import Array, lax
import jax.numpy as jnp
from jax.scipy import special as jsp_special
from functools import partial

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


def _horner(coeffs: Sequence[float], x: Array) -> Array:
    acc = jnp.array(coeffs[0], dtype=jnp.float32)
    for c in coeffs[1:]:
        acc = jnp.array(c, dtype=jnp.float32) + acc * x
    return acc


def _central_region(s: Array, lam: Array) -> Array:
    rm = _horner(_RM_COEFFS, s)
    rm = s + s * (rm * s)

    t = _horner(_T_COEFFS, rm)
    x = _horner(_X_COEFFS, rm) / lam

    total = lam + (x + t) + lam * rm
    return jnp.floor(total)


def _newton_region(s: Array, lam: Array) -> Array:
    dtype = s.dtype

    def cond_fun(state):
        MAX_LOOPS = 5
        r, r_prev, first, counter = state
        diff = jnp.abs(r - r_prev)
        not_done = jnp.logical_or(first, diff > jnp.float32(1e-5))
        not_max_loops = counter < MAX_LOOPS
        return jnp.logical_and(not_done, not_max_loops)

    def body_fun(state):
        r, r_prev, first, counter = state
        t = jnp.log(r)
        s2 = jnp.sqrt(jnp.float32(2.0) * ((jnp.float32(1.0) - r) + r * t))
        s2 = jnp.where(r < jnp.float32(1.0), -s2, s2)
        next_r = r - (s2 - s) * s2 / t
        next_r = jnp.maximum(next_r, jnp.float32(0.1) * r)
        return (next_r, r, jnp.array(False, dtype=jnp.bool_), counter + 1)

    r0 = jnp.maximum(jnp.float32(0.1), jnp.float32(1.0) + s)
    r, _, _, _ = lax.while_loop(
        cond_fun, body_fun, (r0, r0, jnp.array(True, dtype=jnp.bool_), 0)
    )

    t = jnp.log(r)
    sqrt_term = jnp.sqrt(jnp.float32(2.0) * r * ((jnp.float32(1.0) - r) + r * t))
    log_correction = jnp.log(
        sqrt_term / jnp.maximum(jnp.abs(r - jnp.float32(1.0)), jnp.finfo(dtype).tiny)
    )
    x = lam * r + log_correction / t
    x -= jnp.float32(0.0218) / (x + jnp.float32(0.065) * lam)
    return jnp.floor(x)


def _bottom_up(u: Array, lam: Array) -> Array:
    lami = jnp.float32(1.0) / lam

    t0 = jnp.exp(jnp.float32(0.5) * lam)
    del0 = jnp.where(
        u > jnp.float32(0.5), t0 * (jnp.float32(1e-6) * t0), jnp.float32(0.0)
    )
    s0 = jnp.float32(1.0) - t0 * (u * t0) + del0

    def cond1(state):
        MAX_LOOPS = 10
        _, s, _, _, counter = state
        not_done = s < jnp.float32(0.0)
        not_max_loops = counter < MAX_LOOPS
        return jnp.logical_and(not_done, not_max_loops)

    def body1(state):
        x, s, delta, _, counter = state
        x_next = x + jnp.float32(1.0)
        t_next = x_next * lami
        delta_next = t_next * delta
        s_next = t_next * s + jnp.float32(1.0)
        return (x_next, s_next, delta_next, t_next, counter + 1)

    x_init = jnp.float32(0.0)
    x, s, delta, _, _ = lax.while_loop(
        cond1, body1, (x_init, s0, del0, jnp.float32(0.0), 0)
    )

    def top_down_branch(state):
        x_val, delta_val = state
        delta_scaled = jnp.float32(1e6) * delta_val
        t_thresh = jnp.float32(1e7) * delta_scaled
        delta_scaled = (jnp.float32(1.0) - u) * delta_scaled

        def cond2(inner_state):
            MAX_LOOPS = 10
            _, delta_inner, counter = inner_state
            not_done = delta_inner < t_thresh
            not_max_loops = counter < MAX_LOOPS
            return jnp.logical_and(not_done, not_max_loops)

        def body2(inner_state):
            x_inner, delta_inner, counter = inner_state
            x_next = x_inner + jnp.float32(1.0)
            delta_next = delta_inner * (x_next * lami)
            return (x_next, delta_next, counter + 1)

        x_hi, delta_hi, _ = lax.while_loop(cond2, body2, (x_val, delta_scaled, 0))

        def cond3(inner_state):
            MAX_LOOPS = 10
            _, s_inner, _, counter = inner_state
            not_done = s_inner > jnp.float32(0.0)
            not_max_loops = counter < MAX_LOOPS
            return jnp.logical_and(not_done, not_max_loops)

        def body3(inner_state):
            x_inner, s_inner, t_inner, counter = inner_state
            t_next = t_inner * (x_inner * lami)
            s_next = s_inner - t_next
            x_next = x_inner - jnp.float32(1.0)
            return (x_next, s_next, t_next, counter + 1)

        x_lo, _, _, _ = lax.while_loop(
            cond3, body3, (x_hi, delta_hi, jnp.float32(1.0), 0)
        )
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

    x0 = jnp.float32(0.0)

    lam_invalid = lam <= jnp.float32(0.0)
    lam_safe = cast(Array, jnp.where(lam_invalid, jnp.float32(1.0), lam))

    def large_lambda_case(_):
        s = jsp_special.ndtri(u) * lax.rsqrt(lam_safe)

        def central(_):
            return _central_region(s, lam_safe)

        def non_central(_):
            return lax.cond(
                s > -_SQRT2,
                lambda __: _newton_region(s, lam_safe),
                lambda __: x0,
                operand=jnp.float32(0.0),
            )

        return lax.cond(
            jnp.logical_and(s > jnp.float32(-0.6833501), s < jnp.float32(1.777993)),
            central,
            non_central,
            operand=jnp.float32(0.0),
        )

    large_lambda = lam_safe > jnp.float32(4.0)
    x_large = lax.cond(
        large_lambda,
        large_lambda_case,
        lambda _: x0,
        operand=jnp.float32(0.0),
    )

    def bottom_up_branch(_):
        return _bottom_up(u, lam_safe)

    bottom_up = x_large < jnp.float32(10.0)
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
        operand=jnp.float32(0.0),
    )

    nan = jnp.array(jnp.nan, dtype=dtype)
    inf = jnp.array(jnp.inf, dtype=dtype)

    x = cast(Array, jnp.where(u < jnp.float32(0.0), nan, x))
    x = cast(Array, jnp.where(u == jnp.float32(0.0), jnp.float32(0.0), x))
    x = cast(Array, jnp.where(u == jnp.float32(1.0), inf, x))
    x = cast(Array, jnp.where(u > jnp.float32(1.0), nan, x))
    x = cast(Array, jnp.where(lam_invalid, nan, x))
    x = cast(Array, jnp.where(x < 0.0, jnp.float32(0.0), x))
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
def rpoisson(key: Array, lam: Array) -> Array:
    """
    Generate a Poisson random variable with given rate parameter.

    Args:
        key: a PRNG key used as the random key.
        lam: rate parameters for the Poisson distribution.

    Returns:
        A Poisson random variable.
    """
    shape = lam.shape
    u = jax.random.uniform(key, shape)
    x = poissinvf(u, lam)
    return x.astype(lam.dtype)

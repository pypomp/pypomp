"""
JAX implementation of the single-precision inverse Gamma CDF approximation.

The implementation uses the asymptotic inversion method described in Temme (1992).
"""

from __future__ import annotations

from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import ndtri
import numpy as np
from jax._src import dtypes

from ._dtype_helpers import check_and_canonicalize_user_dtype, _get_available_dtype


@partial(jax.jit, static_argnames=["adjustment_size", "dtype", "newton_steps"])
def fast_gamma(
    key: jax.Array,
    alpha: jax.Array,
    dtype: np.dtype | None = None,
    adjustment_size: int = 3,
    newton_steps: int = 3,
) -> jax.Array:
    """
    Generate a Gamma random variable using an approximate inverse CDF method in order to run fast on GPUs.

    The implementation follows the methodology from Temme (1992). To extend the method to small alpha values, we apply a multi-step trick. The method does not produce exact Gamma random variables, but it is very close to exact.

    Args:
        key: a PRNG key used as the random key.
        alpha: shape parameters for the Gamma(alpha, 1) distribution.
        dtype: optional, a float dtype for the returned values (default float64 if
            jax_enable_x64 is true, otherwise float32).
        adjustment_size: number of uniform adjustments to apply.
            The function generates Gamma(alpha + adjustment_size) and reduces
            it to Gamma(alpha) using adjustment_size uniform adjustments. The larger the value, the more accurate the approximation at low alpha values (e.g.,
            alpha < 2).
        newton_steps: number of Newton-Raphson iterations to perform for refining
            the CDF inverse approximation.

    Returns:
        A jax.Array with the same shape as alpha.

    References:
        * Temme, N. M. "Asymptotic Inversion of Incomplete Gamma Functions." Mathematics of Computation 58, no. 198 (1992): 755–64. https://doi.org/10.2307/2153214.
    """
    dtype = check_and_canonicalize_user_dtype(float if dtype is None else dtype)
    assert dtype is not None

    if not dtypes.issubdtype(dtype, np.floating):
        raise ValueError(
            f"dtype argument to `fast_gamma` must be a float dtype, got {dtype}"
        )

    dtype = _get_available_dtype(dtype)
    assert dtype is not None

    shape = alpha.shape
    alpha_dtype = jnp.asarray(alpha, dtype=dtype)

    key_base, key_adj = jax.random.split(key)

    # Apply the multi-step Gamma(alpha + adjustment_size) trick for better accuracy
    alpha_base = alpha_dtype + jnp.full(shape, adjustment_size, dtype=dtype)

    u_base = jax.random.uniform(key_base, shape, dtype=dtype)
    u_base = jnp.clip(u_base, 1e-7, 1.0 - 1e-7)
    x = gammainv(u_base, alpha_base, dtype=dtype, newton_steps=newton_steps)

    u_adj = jax.random.uniform(key_adj, (adjustment_size,) + shape, dtype=dtype)
    u_adj = jnp.clip(u_adj, 1e-7, 1.0 - 1e-7)

    adjustment_indices = jnp.arange(adjustment_size - 1, -1, -1, dtype=dtype)

    adjustment_indices = adjustment_indices.reshape(
        (adjustment_size,) + (1,) * len(shape)
    )
    adjustment_powers = jnp.array(1.0, dtype=dtype) / (alpha_dtype + adjustment_indices)
    adjustments = jnp.power(u_adj, adjustment_powers)

    x = x * jnp.prod(adjustments, axis=0)

    return x.astype(dtype)


@partial(jax.jit, static_argnames=["dtype", "newton_steps"])
def gammainv(u: Array, alpha: Array, dtype=jnp.float32, newton_steps: int = 3) -> Array:
    """
    Vectorized inverse Gamma CDF approximation using JAX primitives.

    Args:
        u: Probabilities (scalar or array) in the interval [0, 1].
        alpha: Corresponding Gamma shape parameter(s), must be positive.
        dtype: Data type for computation (default float32).
        newton_steps: Number of Newton-Raphson iterations to perform (default: 3).

    Returns:
        DeviceArray with the same broadcast shape as `u` and `alpha`.
    """
    u, alpha = jnp.broadcast_arrays(u, alpha)

    u = jnp.asarray(u, dtype=dtype)
    alpha = jnp.asarray(alpha, dtype=dtype)

    zero = jnp.array(0.0, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)

    alpha_invalid = alpha <= zero
    alpha_safe = jnp.where(alpha_invalid, one, alpha)

    eta_0 = ndtri(u) / jnp.sqrt(alpha_safe)

    eps = _compute_epsilon(eta_0, dtype)

    correction = (
        (eps[0] / alpha_safe)
        + (eps[1] / alpha_safe**2)
        + (eps[2] / alpha_safe**3)
        + (eps[3] / alpha_safe**4)
    )

    eta = eta_0 + correction

    lam = _solve_lambda_from_eta(eta, dtype, newton_steps=newton_steps)

    x = alpha_safe * lam

    nan = jnp.array(jnp.nan, dtype=dtype)
    inf = jnp.array(jnp.inf, dtype=dtype)

    x = jnp.where(u < zero, nan, x)
    x = jnp.where(u == zero, zero, x)
    x = jnp.where(u == one, inf, x)
    x = jnp.where(u > one, nan, x)
    x = jnp.where(alpha_invalid, nan, x)
    x = jnp.where(x < zero, zero, x)
    return x


_LAM_GUESS_COEFFS: tuple[float, ...] = (
    -1.0 / 270.0,
    1.0 / 36.0,
    1.0 / 3.0,
    1.0,
    1.0,
)

_LAM_GUESS_COEFFS_ARR = np.array(_LAM_GUESS_COEFFS, dtype=np.float64)


def _solve_lambda_from_eta(eta: Array, dtype, newton_steps: int = 3) -> Array:
    """
    Inverts the relation 1/2 * eta^2 = lambda - 1 - ln(lambda) in log-space.

    Uses a log-space Newton-Raphson solver to prevent left-tail underflow,
    NaN gradients, and to guarantee stable convergence in fewer steps.
    """
    zero = jnp.array(0.0, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    half = jnp.array(0.5, dtype=dtype)

    eta2 = eta**2

    lam_guess = jnp.polyval(_LAM_GUESS_COEFFS_ARR.astype(dtype), eta)

    z_guess = jnp.where(
        lam_guess <= jnp.array(0.01, dtype=dtype),
        -half * eta2,
        jnp.log(jnp.maximum(lam_guess, jnp.array(1e-10, dtype=dtype))),
    )

    def newton_step(z_curr):
        # Use expm1 to avoid precision loss near z=0 (lambda=1)
        expm1_z = jnp.expm1(z_curr)
        val = expm1_z - z_curr - half * eta2
        grad = expm1_z
        # Avoid division by zero at z=0 (eta=0)
        grad_is_small = jnp.abs(grad) < jnp.array(1e-6, dtype=dtype)
        safe_grad = jnp.where(grad_is_small, one, grad)
        step = val / safe_grad
        step = jnp.where(grad_is_small, zero, step)
        return z_curr - step

    z = z_guess
    for _ in range(newton_steps):
        z = newton_step(z)

    return jnp.exp(z)


_E1_COEFFS: tuple[float, ...] = (
    -3224618478943.0 / 170264214140233973760000.0,
    12699400547.0 / 153146779782796800000.0,
    -756882301459.0 / 445517904822681600000.0,
    -449.0 / 1595917323000.0,
    119937661.0 / 30505427656704000.0,
    -2152217.0 / 127673385840000.0,
    2745493.0 / 84737299046400.0,
    1231.0 / 15913705500.0,
    -454973.0 / 498845952000.0,
    37.0 / 9797760.0,
    -101.0 / 16329600.0,
    -11.0 / 382725.0,
    5.0 / 18144.0,
    -7.0 / 6480.0,
    1.0 / 1620.0,
    1.0 / 36.0,
    -1.0 / 3.0,
)

_E2_COEFFS: tuple[float, ...] = (
    52310527831.0 / 343186061137920000.0,
    -311266223.0 / 899963447040000.0,
    -100824673.0 / 571976768563200.0,
    919081.0 / 185177664000.0,
    -9281803.0 / 436490208000.0,
    10217.0 / 251942400.0,
    109.0 / 1749600.0,
    -1579.0 / 2099520.0,
    533.0 / 204120.0,
    -7.0 / 2592.0,
    -7.0 / 405.0,
)

_E3_COEFFS: tuple[float, ...] = (
    987512909021.0 / 514779091706880000.0,
    -69980826653.0 / 39598391669760000.0,
    -1359578327.0 / 129994720128000.0,
    14408797.0 / 246903552000.0,
    -18442139.0 / 130947062400.0,
    346793.0 / 5290790400.0,
    29233.0 / 36741600.0,
    -63149.0 / 20995200.0,
    449.0 / 102060.0,
)

_E4_COEFFS: tuple[float, ...] = (
    636178018081.0 / 48260539847520000.0,
    -16004851139.0 / 26398927779840000.0,
    -16968489929.0 / 194992080192000.0,
    1981235233.0 / 6666395904000.0,
    -449882243.0 / 982102968000.0,
    -269383.0 / 4232632320.0,
    319.0 / 183708.0,
)

_E1_COEFFS_ARR = np.array(_E1_COEFFS, dtype=np.float64)
_E2_COEFFS_ARR = np.array(_E2_COEFFS, dtype=np.float64)
_E3_COEFFS_ARR = np.array(_E3_COEFFS, dtype=np.float64)
_E4_COEFFS_ARR = np.array(_E4_COEFFS, dtype=np.float64)


def _compute_epsilon(eta: Array, dtype) -> tuple[Array, Array, Array, Array]:
    """
    Computes epsilon_1 through epsilon_4 using Horner's method.
    """
    e1 = jnp.polyval(_E1_COEFFS_ARR.astype(dtype), eta)
    e2 = jnp.polyval(_E2_COEFFS_ARR.astype(dtype), eta)
    e3 = jnp.polyval(_E3_COEFFS_ARR.astype(dtype), eta)
    e4 = jnp.polyval(_E4_COEFFS_ARR.astype(dtype), eta)
    return e1, e2, e3, e4

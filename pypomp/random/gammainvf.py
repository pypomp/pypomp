"""
JAX implementation of the single-precision inverse Gamma CDF approximation.

The implementation uses the asymptotic inversion method described in Temme (1992).
"""

from __future__ import annotations

from typing import cast
from functools import partial

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import ndtri


def _gammainvf_scalar(u: Array, alpha: Array) -> Array:
    """
    Scalar inverse Gamma CDF using the Temme (1992) asymptotic inversion method.

    Args:
        u: Probability in the interval [0, 1].
        alpha: Shape parameter for the Gamma(alpha, 1) distribution, must be positive.

    Returns:
        Inverse CDF value.
    """
    dtype = jnp.float32
    u = jnp.asarray(u, dtype=dtype)
    alpha = jnp.asarray(alpha, dtype=dtype)

    alpha_invalid = alpha <= jnp.float32(0.0)
    alpha_safe = cast(Array, jnp.where(alpha_invalid, jnp.float32(1.0), alpha))

    # 1. Calculate eta_0 (The starting approximation)
    # Eq (3.2) implies eta_0 is related to the inverse error function.
    # In standard statistical terms, eta_0 * sqrt(alpha) is the z-score
    # corresponding to probability u.
    # eta_0 = Phi^-1(u) / sqrt(alpha)
    eta_0 = ndtri(u) / jnp.sqrt(alpha_safe)

    # 2. Calculate the perturbation epsilon(eta)
    # Using the polynomial expansions from Section 5.
    eps = _compute_epsilon(eta_0)

    # 3. Calculate the refined eta
    # Eq (3.3): eta = eta_0 + epsilon
    # Eq (3.4): epsilon ~ e1/a + e2/a^2 + e3/a^3 + e4/a^4
    # Note: The epsilon functions from Section 5 are technically e_i(eta).
    # The paper notes we can approximate e_i(eta) with e_i(eta_0).

    correction = (
        (eps[0] / alpha_safe)
        + (eps[1] / alpha_safe**2)
        + (eps[2] / alpha_safe**3)
        + (eps[3] / alpha_safe**4)
    )

    eta = eta_0 + correction

    # 4. Convert eta back to lambda (where x = alpha * lambda)
    # We need to invert the relation: 1/2 * eta^2 = lambda - 1 - ln(lambda)
    # with the condition sign(eta) == sign(lambda - 1).
    lam = _solve_lambda_from_eta(eta)

    x = alpha_safe * lam

    nan = jnp.array(jnp.nan, dtype=dtype)
    inf = jnp.array(jnp.inf, dtype=dtype)

    x = cast(Array, jnp.where(u < jnp.float32(0.0), nan, x))
    x = cast(Array, jnp.where(u == jnp.float32(0.0), jnp.float32(0.0), x))
    x = cast(Array, jnp.where(u == jnp.float32(1.0), inf, x))
    x = cast(Array, jnp.where(u > jnp.float32(1.0), nan, x))
    x = cast(Array, jnp.where(alpha_invalid, nan, x))
    x = cast(Array, jnp.where(x < 0.0, jnp.float32(0.0), x))
    return x


_gammainvf_vmap = jax.vmap(_gammainvf_scalar)


@jax.jit
def gammainvf(u: Array, alpha: Array) -> Array:
    """
    Vectorized inverse Gamma CDF approximation using JAX primitives.

    Args:
        u: Probabilities (scalar or array) in the interval [0, 1].
        alpha: Corresponding Gamma shape parameter(s), must be positive.

    Returns:
        DeviceArray with the same broadcast shape as `u` and `alpha`.
    """
    u_arr, alpha_arr = jnp.broadcast_arrays(u, alpha)
    flat_u = u_arr.reshape(-1)
    flat_alpha = alpha_arr.reshape(-1)
    flat_res = _gammainvf_vmap(flat_u, flat_alpha)
    return flat_res.reshape(u_arr.shape)


@partial(jax.jit, static_argnames=["adjustment_size"])
def fast_approx_rgamma(
    key: jax.Array, alpha: jax.Array, adjustment_size: int = 3
) -> jax.Array:
    """
    Generate a Gamma random variable with given shape parameter.

    The implementation follows the methodology from Temme (1992). To extend the method to small alpha values, we apply a multi-step trick.

    Args:
        key: a PRNG key used as the random key.
        alpha: shape parameters for the Gamma(alpha, 1) distribution.
        adjustment_size: number of uniform adjustments to apply (default: 3).
            The function generates Gamma(alpha + adjustment_size) and reduces
            it to Gamma(alpha) using adjustment_size uniform adjustments. The larger the value, the more accurate the approximation at low alpha values (e.g.,
            alpha < 2).

    Returns:
        A jax.Array with the same shape as alpha.

    References:
        * Temme, N. M. “Asymptotic Inversion of Incomplete Gamma Functions.” Mathematics of Computation 58, no. 198 (1992): 755–64. https://doi.org/10.2307/2153214.
    """
    shape = alpha.shape
    alpha_orig_dtype = alpha.dtype
    alpha_f32 = jnp.asarray(alpha, dtype=jnp.float32)

    key_base, key_adj = jax.random.split(key)

    # Apply the multi-step Gamma(alpha + adjustment_size) trick for better accuracy
    alpha_base = alpha_f32 + jnp.full(shape, adjustment_size, dtype=jnp.float32)

    u_base = jax.random.uniform(key_base, shape)
    x = gammainvf(u_base, alpha_base)

    u_adj = jax.random.uniform(key_adj, (adjustment_size,) + shape)

    adjustment_indices = jnp.arange(adjustment_size - 1, -1, -1, dtype=jnp.float32)

    adjustment_indices = adjustment_indices.reshape(
        (adjustment_size,) + (1,) * len(shape)
    )
    adjustment_powers = 1.0 / (alpha_f32 + adjustment_indices)
    adjustments = jnp.power(u_adj, adjustment_powers)

    # Multiply all adjustments together
    x = x * jnp.prod(adjustments, axis=0)

    return x.astype(alpha_orig_dtype)


def _compute_epsilon(eta):
    """
    Computes epsilon_1 through epsilon_4 using the Taylor expansions
    provided in Section 5 of Temme (1992).
    """
    # Coefficients extracted from Section 5 text

    (
        eta2,
        eta3,
        eta4,
        eta5,
        eta6,
        eta7,
        eta8,
        eta9,
        eta10,
        eta11,
        eta12,
        eta13,
        eta14,
        eta15,
        eta16,
    ) = [eta**i for i in range(2, 17)]

    # epsilon 1
    e1 = (
        -1.0 / 3.0
        + (1.0 / 36.0) * eta
        + (1.0 / 1620.0) * eta2
        - (7.0 / 6480.0) * eta3
        + (5.0 / 18144.0) * eta4
        - (11.0 / 382725.0) * eta5
        - (101.0 / 16329600.0) * eta6
        + (37.0 / 9797760.0) * eta7
        - (454973.0 / 498845952000.0) * eta8
        + (1231.0 / 15913705500.0) * eta9
        + (2745493.0 / 84737299046400.0) * eta10
        - (2152217.0 / 127673385840000.0) * eta11
        + (119937661.0 / 30505427656704000.0) * eta12
        - (449.0 / 1595917323000.0) * eta13
        - (756882301459.0 / 445517904822681600000.0) * eta14
        + (12699400547.0 / 153146779782796800000.0) * eta15
        - (3224618478943.0 / 170264214140233973760000.0) * eta16
    )

    # epsilon 2
    e2 = (
        -7.0 / 405.0
        - (7.0 / 2592.0) * eta
        + (533.0 / 204120.0) * eta2
        - (1579.0 / 2099520.0) * eta3
        + (109.0 / 1749600.0) * eta4
        + (10217.0 / 251942400.0) * eta**5
        - (9281803.0 / 436490208000.0) * eta**6
        + (919081.0 / 185177664000.0) * eta7
        - (100824673.0 / 571976768563200.0) * eta8
        - (311266223.0 / 899963447040000.0) * eta9
        + (52310527831.0 / 343186061137920000.0) * eta10
    )

    # epsilon 3
    e3 = (
        (449.0 / 102060.0)
        - (63149.0 / 20995200.0) * eta
        + (29233.0 / 36741600.0) * eta2
        + (346793.0 / 5290790400.0) * eta3
        - (18442139.0 / 130947062400.0) * eta4
        + (14408797.0 / 246903552000.0) * eta5
        - (1359578327.0 / 129994720128000.0) * eta6
        - (69980826653.0 / 39598391669760000.0) * eta7
        + (987512909021.0 / 514779091706880000.0) * eta8
    )

    # epsilon 4
    e4 = (
        (319.0 / 183708.0)
        - (269383.0 / 4232632320.0) * eta
        - (449882243.0 / 982102968000.0) * eta2
        + (1981235233.0 / 6666395904000.0) * eta3
        - (16968489929.0 / 194992080192000.0) * eta4
        - (16004851139.0 / 26398927779840000.0) * eta5
        + (636178018081.0 / 48260539847520000.0) * eta6
    )

    return e1, e2, e3, e4


def _solve_lambda_from_eta(eta):
    """
    Inverts the relation 1/2 * eta^2 = lambda - 1 - ln(lambda).

    Uses the series expansion from Section 6 as an initial guess,
    followed by Newton-Raphson iterations as suggested in the paper.
    """
    # Series approximation from Section 6
    # lambda = 1 + eta + 1/3 eta^2 + 1/36 eta^3 - 1/270 eta^4 + ...
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4

    lam_guess = (
        1.0 + eta + (1.0 / 3.0) * eta2 + (1.0 / 36.0) * eta3 - (1.0 / 270.0) * eta4
    )

    # For very large negative eta (left tail), the series might be unstable (lambda < 0).
    # Since lambda must be > 0, we clamp the guess.
    # For eta << -1, lambda is small, dominated by -ln(lambda) ~ eta^2/2 -> lambda ~ exp(-eta^2/2)
    # This prevents NaN in the log step of Newton-Raphson.
    safe_guess = jnp.where(lam_guess <= 0.01, jnp.exp(-0.5 * eta2), lam_guess)

    # Newton-Raphson refinement
    # f(lambda) = lambda - 1 - ln(lambda) - eta^2/2
    # f'(lambda) = 1 - 1/lambda
    # step = f(lambda) / f'(lambda)
    #      = (lambda - 1 - ln(lambda) - eta^2/2) / ((lambda - 1) / lambda)
    #      = lambda * (lambda - 1 - ln(lambda) - eta^2/2) / (lambda - 1)

    def newton_step(lam_curr):
        val = lam_curr - 1.0 - jnp.log(lam_curr) - 0.5 * eta2
        grad = 1.0 - 1.0 / lam_curr
        # Avoid division by zero at lambda=1 (eta=0)
        # At lambda=1, the limit of val/grad is 0, so update should be 0.
        safe_grad = jnp.where(jnp.abs(grad) < 1e-6, 1.0, grad)
        step = val / safe_grad
        # Mask the step if we are at the singularity to avoid instability
        step = jnp.where(jnp.abs(grad) < 1e-6, 0.0, step)
        lam_new = lam_curr - step
        # Ensure lambda stays positive to avoid NaN in log
        # Use a small positive epsilon to prevent numerical issues
        # This should be relevant only very rarely
        # TODO: implement a better solution
        lam_new = jnp.maximum(lam_new, jnp.float32(1e-10))
        return lam_new

    # 3 iterations is usually sufficient for double precision with this good initial guess
    lam = safe_guess
    lam = newton_step(lam)
    lam = newton_step(lam)
    lam = newton_step(lam)

    return lam

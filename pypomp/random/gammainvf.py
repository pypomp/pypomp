"""
JAX implementation of the single-precision inverse Gamma CDF approximation.

The implementation uses the asymptotic inversion method described in Temme (1992).
"""

from __future__ import annotations

from typing import cast

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


@jax.jit
def rgamma(key: Array, alpha: Array) -> Array:
    """
    Generate a Gamma random variable with given shape parameter.

    Args:
        key: a PRNG key used as the random key.
        alpha: shape parameters for the Gamma(alpha, 1) distribution.

    Returns:
        A Gamma random variable.
    """
    shape = alpha.shape
    alpha_orig_dtype = alpha.dtype
    alpha_f32 = jnp.asarray(alpha, dtype=jnp.float32)

    # Split key for uniform draws
    key_base, key_adj = jax.random.split(key)

    needs_extra_trick = alpha_f32 <= 1.0
    needs_primary_trick = alpha_f32 <= 2.0

    # Apply the Gamma(alpha + 1) trick once for 1 <= alpha < 2 and twice for alpha < 1
    alpha_base = jnp.where(
        needs_extra_trick,
        alpha_f32 + 2.0,
        jnp.where(needs_primary_trick, alpha_f32 + 1.0, alpha_f32),
    )

    u_base = jax.random.uniform(key_base, shape)
    x = gammainvf(u_base, alpha_base)

    # First adjustment reduces from alpha + 2 -> alpha + 1 when needed (alpha < 1)
    u_adj = jax.random.uniform(key_adj, (2,) + shape)
    u_extra = u_adj[0]
    extra_adjustment = jnp.where(
        needs_extra_trick,
        jnp.power(u_extra, 1.0 / (alpha_f32 + 1.0)),
        1.0,
    )
    x = x * extra_adjustment

    # Primary adjustment reduces to the target alpha whenever alpha < 2
    u_primary = u_adj[1]
    primary_adjustment = jnp.where(
        needs_primary_trick,
        jnp.power(u_primary, 1.0 / alpha_f32),
        1.0,
    )
    x = x * primary_adjustment

    return x.astype(alpha_orig_dtype)


def _compute_epsilon(eta):
    """
    Computes epsilon_1 through epsilon_4 using the Taylor expansions
    provided in Section 5 of Temme (1992).
    """
    # Coefficients extracted from Section 5 text

    # To avoid manual error in transcription of high order fractions, we use the
    # explicit lower order terms which dominate, and approximations for higher orders.
    # It is safer to implement the polynomials explicitly with the provided fractions
    # for the first few significant terms.

    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta**4

    # epsilon 1
    e1 = (
        -1.0 / 3.0
        + (1.0 / 36.0) * eta
        + (1.0 / 1620.0) * eta2
        - (7.0 / 6480.0) * eta3
        + (5.0 / 18144.0) * eta4
    )
    # Higher order terms have diminishing returns for sampling efficiency

    # epsilon 2
    e2 = (
        -7.0 / 405.0
        - (7.0 / 2592.0) * eta
        + (533.0 / 204120.0) * eta2
        - (1579.0 / 2099520.0) * eta3
        + (109.0 / 1749600.0) * eta4
    )

    # epsilon 3
    e3 = (
        (449.0 / 102060.0)
        - (63149.0 / 20995200.0) * eta
        + (29233.0 / 36741600.0) * eta2
        + (346793.0 / 5290790400.0) * eta3
    )

    # epsilon 4
    e4 = (
        (319.0 / 183708.0)
        - (269383.0 / 4232632320.0) * eta
        - (449882243.0 / 982102968000.0) * eta2
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
        return lam_curr - step

    # 3 iterations is usually sufficient for double precision with this good initial guess
    lam = safe_guess
    lam = newton_step(lam)
    lam = newton_step(lam)
    lam = newton_step(lam)

    return lam

"""Normal-interval log-probabilities for the discretized measles ``dmeas`` functions.

The measurement models give each count ``y`` the mass of a normal over
``[y - 0.5, y + 0.5]``. Subtracting CDFs (or log-CDFs) loses that mass when the
interval is far above the mean: in float32 both CDFs round to 1 once ``z_lo`` is
above about 5.4, so the likelihood collapses to the tolerance floor even though
double precision still resolves it.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr

LOG_HALF = math.log(0.5)
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _log1mexp(x: jax.Array) -> jax.Array:
    """log(1 - exp(x)) for x < 0, accurate near 0 and for large -x (Maechler 2012)."""
    near_zero = x > LOG_HALF
    a = jnp.log(-jnp.expm1(jnp.where(near_zero, x, -1.0)))
    b = jnp.log1p(-jnp.exp(jnp.where(near_zero, -1.0, x)))
    return jnp.where(near_zero, a, b)


@jax.custom_jvp
def log_ndtr_diff(z_hi: jax.Array, z_lo: jax.Array) -> jax.Array:
    """log(Phi(z_hi) - Phi(z_lo)) for z_hi >= z_lo, accurate in both tails.

    Uses Phi(b) - Phi(a) = Phi(-a) - Phi(-b) to reflect an interval whose midpoint
    is above 0 into the lower half, where log_ndtr keeps relative precision.
    Returns -inf where the difference is 0 at working precision.
    """
    flip = z_hi + z_lo > 0
    hi = jnp.where(flip, -z_lo, z_hi)
    lo = jnp.where(flip, -z_hi, z_lo)
    log_hi = log_ndtr(hi)
    d = log_ndtr(lo) - log_hi
    # d is 0 when hi == lo at working precision and NaN when both underflow to -inf.
    ok = d < 0
    return jnp.where(ok, log_hi + _log1mexp(jnp.where(ok, d, -1.0)), -jnp.inf)


def _phi_ratio(z: jax.Array, log_p: jax.Array) -> jax.Array:
    """phi(z) / exp(log_p), capped at the dtype's max; 0 where log_p is -inf."""
    finite = jnp.isfinite(log_p)
    log_r = -0.5 * z * z - LOG_SQRT_2PI - jnp.where(finite, log_p, 0.0)
    log_max = jnp.log(jnp.finfo(log_p.dtype).max)
    return jnp.where(finite, jnp.exp(jnp.minimum(log_r, log_max)), 0.0)


@log_ndtr_diff.defjvp
def _log_ndtr_diff_jvp(
    primals: tuple[jax.Array, jax.Array], tangents: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """JVP of log_ndtr_diff that never multiplies a zero tangent by inf."""
    z_hi, z_lo = primals
    t_hi, t_lo = tangents
    out = log_ndtr_diff(z_hi, z_lo)
    return out, _phi_ratio(z_hi, out) * t_hi - _phi_ratio(z_lo, out) * t_lo

"""Samplers for vectorized measles ``rproc`` functions, driven by pre-drawn uniforms.

Each vectorized ``rproc`` draws all of a step's uniforms with a single PRNG call
and feeds them through these inverse-CDF transforms, avoiding per-particle key
splits and keeping the number of kernels per Euler step low.
"""

import jax.numpy as jnp
import numpy as np

from pypomp.random import binominv, gammainv, poissoninv

GAMMA_ADJ = 3  # Same adjustment count as fast_gamma's default.
N_GAMMA_UNIFORMS = 1 + GAMMA_ADJ
U_MAX = float(np.nextafter(np.float32(1.0), np.float32(0.0)))


def gamma(u, alpha):
    """fast_gamma from ``N_GAMMA_UNIFORMS`` rows of uniforms ``u``."""
    u = jnp.clip(u, 1e-7, 1.0 - 1e-7)
    alpha = jnp.broadcast_to(alpha, u.shape[1:])
    x = gammainv(u[0], alpha + GAMMA_ADJ)
    idx = jnp.arange(GAMMA_ADJ - 1, -1, -1, dtype=u.dtype)[:, None]
    return x * jnp.prod(jnp.power(u[1:], 1.0 / (alpha + idx)), axis=0)


def poisson(u, lam):
    """fast_poisson, from pre-drawn uniforms."""
    lam = jnp.broadcast_to(lam, u.shape)
    x = poissoninv(jnp.minimum(u, U_MAX), lam)
    return jnp.minimum(x, lam + 10.0 * jnp.sqrt(jnp.maximum(lam, 1.0)))


def binom(u, n, p):
    """fast_binomial, from pre-drawn uniforms."""
    u = jnp.clip(u, jnp.finfo(u.dtype).tiny, U_MAX)
    return binominv(u, n, jnp.broadcast_to(jnp.clip(p, 0.0, 1.0), u.shape))


def euler_probs(r0, r1, dt):
    """Probabilities of staying, leaving via rate r0, and via rate r1 over one step."""
    r_sum = r0 + r1
    p_stay = jnp.exp(-r_sum * dt)
    scale = (1.0 - p_stay) / r_sum
    return p_stay, r0 * scale, r1 * scale


def multinom_exits(u0, u1, n, p0, p1):
    """Counts leaving a class of size n with exit probabilities p0 and p1."""
    x0 = binom(u0, n, p0)
    p_rem = 1.0 - p0
    x1 = binom(u1, n - x0, p1 / jnp.where(p_rem > 0.0, p_rem, 1.0))
    return x0, x1


def euler_exits(u0, u1, n, r0, r1, dt):
    """Euler-multinomial counts leaving a class via rates r0 and r1 over one step."""
    _, p0, p1 = euler_probs(r0, r1, dt)
    return multinom_exits(u0, u1, n, p0, p1)

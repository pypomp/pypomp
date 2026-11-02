"""Polynomial evaluation for the samplers, unrolled whatever JAX's default."""

import jax
import jax.numpy as jnp
import numpy as np


def polyval(coeffs: np.ndarray, x: jax.Array) -> jax.Array:
    """``jnp.polyval(coeffs, x)``, with Horner's rule fully unrolled.

    ``coeffs`` is ordered from the highest degree down, as for ``jnp.polyval``.

    JAX 0.12's ``jnp.polyval`` calls ``lax.polynomial``, which writes out Horner's
    rule for at most 8 coefficients and evaluates longer polynomials in a ``scan``
    of 8 coefficients per step. With two or more steps, as for the 17 coefficients
    of ``gammainv``'s first Temme coefficient, XLA keeps that scan as a loop, which
    stops it fusing the sampler into one kernel: ``gammainv`` was 2.7 times slower
    on an A40 for 5,000 draws per call. JAX 0.11 unrolled up to 16 coefficients.
    """
    return jnp.polyval(coeffs, x, unroll=len(coeffs))

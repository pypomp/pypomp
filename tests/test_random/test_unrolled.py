"""The inverse CDFs lower to straight-line code, with no loops.

A loop in a sampler's HLO keeps XLA from fusing it into one kernel. JAX 0.12's
``jnp.polyval`` evaluates polynomials of more than 8 coefficients in a scan, which
made ``gammainv`` 2.7 times slower on a GPU until pypomp unrolled it.

The ``fast_*`` samplers aren't checked: ``jax.random.uniform``'s threefry lowers to
a loop of its own, which XLA unrolls.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp.random as ppr
from pypomp.random._polyval import polyval

N = 16
U = jnp.linspace(0.01, 0.99, N)

CASES = {
    "gammainv": (ppr.gammainv, (U, jnp.full(N, 3.4))),
    "poissoninv": (ppr.poissoninv, (U, jnp.full(N, 190.0))),
    "binominv": (ppr.binominv, (U, jnp.full(N, 500.0), jnp.full(N, 0.08))),
}


@pytest.mark.parametrize("name", CASES)
def test_inverse_cdf_lowers_without_loops(name: str) -> None:
    fn, args = CASES[name]
    text = jax.jit(fn).lower(*args).as_text()
    assert "stablehlo.while" not in text


def test_polyval_matches_numpy() -> None:
    coeffs = np.linspace(-1.0, 1.0, 17)
    x = np.linspace(-0.5, 0.5, 11)
    np.testing.assert_allclose(
        polyval(coeffs, jnp.asarray(x)), np.polyval(coeffs, x), rtol=1e-5, atol=1e-6
    )

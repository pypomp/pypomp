"""Float32 measles dmeas against a float64 scipy reference, far tails included.

The allowed error is ``8 * eps32 * kappa``, where ``kappa`` is how far the float64
value moves when y +/- 0.5, m and s are perturbed by one float32 ulp.
"""

import itertools
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from pypomp.models.measles import (
    model_001,
    model_001b,
    model_001d,
    model_002,
    model_002d,
    model_003,
)

EPS32 = float(np.finfo(np.float32).eps)
# 00x return log(P + 1e-18); the others floor v at tol**2 and return max(log P, log 1e-12).
FLOOR_MODELS = [model_001d, model_002d, model_003]
MODELS = pytest.mark.parametrize(
    "mod",
    [model_001, model_001b, model_002, *FLOOR_MODELS],
    ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)


def _interval_prob(z_hi: np.ndarray, z_lo: np.ndarray) -> np.ndarray:
    """float64 Phi(z_hi) - Phi(z_lo), from whichever tail avoids cancellation."""
    upper = norm.sf(z_lo) - norm.sf(z_hi)
    lower = norm.cdf(z_hi) - norm.cdf(z_lo)
    middle = 1.0 - norm.cdf(z_lo) - norm.sf(z_hi)
    return np.where(z_lo > 0, upper, np.where(z_hi < 0, lower, middle))


def _grid() -> tuple[np.ndarray, ...]:
    """Float32-exact (y, C, rho, psi): a coarse grid plus counts 5-9 sd from the mean."""
    rhos, psis = [0.1, 0.5, 0.9], [0.05, 0.2, 1.0]
    pts = list(
        itertools.product(
            [0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4],
            [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4],
            rhos,
            psis,
        )
    )
    for C, rho, psi, k in itertools.product(
        [10.0, 100.0, 1e3, 1e4], rhos, psis, [-9, -7, -5, 5, 6, 7, 8, 9]
    ):
        m = rho * C
        s = np.sqrt(m * (1.0 - rho + psi**2 * m))
        pts.append((max(np.round(m + k * s), 0.0), C, rho, psi))
    cols = zip(*pts, strict=True)
    return tuple(np.asarray(c, np.float32).astype(np.float64) for c in cols)


def _reference(
    y: np.ndarray, C: np.ndarray, rho: np.ndarray, psi: np.ndarray, floor: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """float64 dmeas, its kappa, and z_lo."""
    tol = 1e-12 if floor else 1e-18
    m = rho * (np.maximum(C, 0.0) if floor else C)
    v = m * (1.0 - rho + psi**2 * m)
    s = np.sqrt(np.maximum(v, tol**2)) if floor else np.sqrt(v) + tol
    z_hi, z_lo = (y + 0.5 - m) / s, (y - 0.5 - m) / s
    pos = y > tol
    p = np.where(pos, _interval_prob(z_hi, z_lo), norm.cdf(z_hi))
    with np.errstate(divide="ignore"):
        ll = np.maximum(np.log(p), np.log(tol)) if floor else np.log(p + tol)
    dll_dz = (norm.pdf(z_hi) + np.where(pos, norm.pdf(z_lo), 0.0)) / (p + tol)
    dz = 1.0 + np.maximum(np.abs(z_hi), np.abs(z_lo)) + (y + np.abs(m) + 1.0) / s
    return ll, 1.0 + np.abs(ll) + dll_dz * dz, z_lo


def _dmeas(
    mod: ModuleType, y: jax.Array, C: jax.Array, rho: jax.Array, psi: jax.Array
) -> jax.Array:
    """The model's float32 dmeas."""
    theta = {"rho": rho, "psi": psi}
    return mod.dmeas({"cases": y}, {"C": C}, theta, None, None)


@MODELS
def test_dmeas_matches_float64(mod):
    y, C, rho, psi = _grid()
    ref, kappa, z_lo = _reference(y, C, rho, psi, mod in FLOOR_MODELS)
    args = (jnp.asarray(a, jnp.float32) for a in (y, C, rho, psi))
    got = np.asarray(jax.jit(_dmeas, static_argnums=0)(mod, *args), np.float64)

    assert np.all(np.isfinite(got))
    i = np.argmax(np.abs(got - ref) / kappa)
    assert np.abs(got - ref)[i] <= 8 * EPS32 * kappa[i], (
        f"y={y[i]} C={C[i]} rho={rho[i]} psi={psi[i]}: {got[i]} vs {ref[i]}"
    )
    # Upper-tail points that float64 resolves above the floor, where the old
    # float32 code returned log(tol).
    assert np.sum((z_lo > 5.5) & (ref > np.log(1e-12) + 1.0)) >= 10


@MODELS
def test_dmeas_gradient(mod):
    # y = 60 is 5.6 sd above the mean of 25.
    y, rho, psi, h = np.array(60.0), np.array(0.5), np.array(0.2), 1e-3
    hi, lo = (
        _reference(y, np.array(c), rho, psi, mod in FLOOR_MODELS)[0]
        for c in (50.0 + h, 50.0 - h)
    )
    y32, rho32, psi32 = (jnp.asarray(a, jnp.float32) for a in (y, rho, psi))
    grad = jax.grad(lambda C: _dmeas(mod, y32, C, rho32, psi32))(jnp.float32(50.0))
    np.testing.assert_allclose(float(grad), (hi - lo) / (2 * h), rtol=1e-4)

    # Finite on the grid too; the 00x models are NaN at C == 0 through sqrt(v).
    grid = _grid()
    y, C, rho, psi = (jnp.asarray(a[grid[1] > 0], jnp.float32) for a in grid)
    grads = jax.jit(jax.grad(lambda C: jnp.sum(_dmeas(mod, y, C, rho, psi))))(C)
    assert np.all(np.isfinite(np.asarray(grads)))

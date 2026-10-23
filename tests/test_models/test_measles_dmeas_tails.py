"""Float32 measles dmeas against a float64 scipy reference, including far tails.

Subtracting normal CDFs that are both near 1 cancels to 0 in float32 once y is
about 5.4 sd above the mean, which used to floor the likelihood at the tolerance
where double precision (R pomp, NumPy) still resolves it.

Allowed error: ``K * eps32 * kappa``, where ``kappa`` is how much the float64
reference moves under relative eps32 perturbations of its inputs (the z-scores,
or y +/- 0.5, m and s for the full dmeas), plus one ulp of the result.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import log_ndtr as log_ndtr64
from scipy.stats import norm

from pypomp.models.measles import (
    model_001,
    model_001b,
    model_001d,
    model_002,
    model_002d,
    model_003,
)
from pypomp.models.measles._dmeas import log_ndtr_diff

EPS32 = float(np.finfo(np.float32).eps)
K = 8.0

# The 00x models return log(P + 1e-18); the "d" models and 003 floor v at
# tol**2 and return max(log P, log 1e-12).
TOL_MODELS = [model_001, model_001b, model_002]
FLOOR_MODELS = [model_001d, model_002d, model_003]


def _name(mod):
    return mod.__name__.rsplit(".", 1)[-1]


def _f32_exact(x):
    """Round to float32 and return as float64, so both sides see the same inputs."""
    return np.asarray(x, np.float32).astype(np.float64)


def _interval_prob64(z_hi, z_lo):
    """float64 Phi(z_hi) - Phi(z_lo), from whichever tail avoids cancellation."""
    upper = norm.sf(z_lo) - norm.sf(z_hi)
    lower = norm.cdf(z_hi) - norm.cdf(z_lo)
    middle = 1.0 - norm.cdf(z_lo) - norm.sf(z_hi)
    return np.where(z_lo > 0, upper, np.where(z_hi < 0, lower, middle))


def _ratios(ref, z_hi, z_lo):
    """phi(z) / (Phi(z_hi) - Phi(z_lo)) for both ends, from the float64 log value."""
    return np.exp(norm.logpdf(z_hi) - ref), np.exp(norm.logpdf(z_lo) - ref)


def _kappa(ref, z_hi, z_lo):
    r_hi, r_lo = _ratios(ref, z_hi, z_lo)
    return 1.0 + np.abs(ref) + r_hi * (1 + np.abs(z_hi)) + r_lo * (1 + np.abs(z_lo))


_helper = jax.jit(log_ndtr_diff)
_helper_grad = jax.jit(jax.vmap(jax.grad(log_ndtr_diff, argnums=(0, 1))))


def _run_helper(z_hi, z_lo):
    z_hi, z_lo = jnp.asarray(z_hi, jnp.float32), jnp.asarray(z_lo, jnp.float32)
    out = _helper(z_hi, z_lo)
    g_hi, g_lo = _helper_grad(z_hi, z_lo)
    return tuple(np.asarray(a, np.float64) for a in (out, g_hi, g_lo))


def _check_helper(ref, z_hi, z_lo):
    got, g_hi, g_lo = _run_helper(z_hi, z_lo)
    kappa = _kappa(ref, z_hi, z_lo)
    assert np.all(np.isfinite(got))
    err = np.abs(got - ref)
    assert np.all(err <= K * EPS32 * kappa), np.max(err / (EPS32 * kappa))

    # Gradients: phi / P inherits P's error plus rounding of z**2 / 2.
    r_hi, r_lo = _ratios(ref, z_hi, z_lo)
    rtol = K * EPS32 * (kappa + 0.5 * np.maximum(z_hi**2, z_lo**2))
    np.testing.assert_array_less(np.abs(g_hi - r_hi), rtol * r_hi + 1e-30)
    np.testing.assert_array_less(np.abs(g_lo + r_lo), rtol * r_lo + 1e-30)


def test_log_ndtr_diff_matches_float64():
    z_lo = np.linspace(-37.0, 37.0, 297)
    widths = [1e-3, 1e-2, 0.1, 1.0, 10.0]
    z_lo, w = (a.ravel() for a in np.meshgrid(z_lo, widths))
    z_lo = _f32_exact(z_lo)
    z_hi = _f32_exact(z_lo + w)

    p = _interval_prob64(z_hi, z_lo)
    keep = p > 1e-300  # float64 resolves the difference itself
    z_hi, z_lo, p = z_hi[keep], z_lo[keep], p[keep]
    _check_helper(np.log(p), z_hi, z_lo)
    # The grid reaches the region the old float32 code collapsed (both CDFs == 1).
    assert np.sum(z_lo > 6.0) > 100


def test_log_ndtr_diff_deep_tails():
    """Beyond float64's range for the probability itself, compare log-space float64."""
    z = np.array([40.0, 60.0, 100.0, 300.0, 1000.0])
    for w, sign in itertools.product([0.01, 1.0], [1.0, -1.0]):
        a, b = _f32_exact(sign * z), _f32_exact(sign * z + w)
        z_lo, z_hi = np.minimum(a, b), np.maximum(a, b)
        # Reflect into the lower tail, where float64 log_ndtr is accurate.
        hi = np.where(sign > 0, -z_lo, z_hi)
        lo = np.where(sign > 0, -z_hi, z_lo)
        ref = log_ndtr64(hi) + np.log1p(-np.exp(log_ndtr64(lo) - log_ndtr64(hi)))
        _check_helper(ref, z_hi, z_lo)


def test_log_ndtr_diff_degenerate_inputs():
    zh = jnp.array([5.0, -1e21, 1e21, 3.0, 40.0, -40.0], jnp.float32)
    zl = jnp.array([5.0, -1.1e21, 0.9e21, -jnp.inf, -jnp.inf, -jnp.inf], jnp.float32)
    got, g_hi, g_lo = _run_helper(zh, zl)
    # Zero-width and underflowing intervals give -inf, not NaN, with zero gradient.
    assert np.all(got[:3] == -np.inf)
    assert np.all(g_hi[:3] == 0.0) and np.all(g_lo[:3] == 0.0)
    # An infinite lower end reduces to log_ndtr.
    ref = log_ndtr64(np.asarray(zh[3:], np.float64))
    np.testing.assert_allclose(got[3:], ref, rtol=K * EPS32, atol=K * EPS32)
    assert np.all(np.isfinite(g_hi)) and np.all(g_lo[3:] == 0.0)


def _dmeas_grid():
    ys = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4]
    Cs = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4]
    rhos = [0.1, 0.5, 0.9]
    psis = [0.05, 0.2, 1.0]
    pts = list(itertools.product(ys, Cs, rhos, psis))
    # Counts k sd from the mean, where the old float32 code floored the likelihood.
    for C, rho, psi, k in itertools.product(
        [10.0, 100.0, 1e3, 1e4], rhos, psis, [-9, -7, -5, 5, 6, 7, 8, 9]
    ):
        m = rho * C
        s = np.sqrt(m * (1.0 - rho + psi**2 * m))
        pts.append((max(np.round(m + k * s), 0.0), C, rho, psi))
    cols = zip(*pts, strict=True)
    return tuple(_f32_exact(c) for c in cols)


def _dmeas_ref(y, C, rho, psi, floor_model):
    """float64 version of the measles dmeas; returns (loglik, kappa, z_lo)."""
    if floor_model:
        tol = 1e-12
        m = rho * np.maximum(C, 0.0)
        s = np.sqrt(np.maximum(m * (1.0 - rho + psi**2 * m), tol**2))
    else:
        tol = 1e-18
        m = rho * C
        s = np.sqrt(m * (1.0 - rho + psi**2 * m)) + tol
    z_hi = (y + 0.5 - m) / s
    z_lo = (y - 0.5 - m) / s
    pos = y > tol
    p = np.where(pos, _interval_prob64(z_hi, z_lo), norm.cdf(z_hi))
    with np.errstate(divide="ignore"):
        ref = np.maximum(np.log(p), np.log(tol)) if floor_model else np.log(p + tol)

    # Sensitivity to relative eps perturbations of y +/- 0.5, m and s.
    g = (norm.pdf(z_hi) + np.where(pos, norm.pdf(z_lo), 0.0)) / (p + tol)
    dz = 1.0 + np.maximum(np.abs(z_hi), np.abs(z_lo))
    dz = dz + (np.abs(y) + np.abs(m) + 1.0) / s
    kappa = 1.0 + np.abs(ref) + g * dz
    return ref, kappa, z_lo


def _run_dmeas(mod, y, C, rho, psi):
    def f32(a):
        return jnp.asarray(a, jnp.float32)

    out = jax.jit(mod.dmeas)(
        {"cases": f32(y)}, {"C": f32(C)}, {"rho": f32(rho), "psi": f32(psi)}, None, None
    )
    return np.asarray(out, np.float64)


@pytest.mark.parametrize("mod", TOL_MODELS + FLOOR_MODELS, ids=_name)
def test_dmeas_matches_float64(mod):
    floor_model = mod in FLOOR_MODELS
    y, C, rho, psi = _dmeas_grid()
    ref, kappa, z_lo = _dmeas_ref(y, C, rho, psi, floor_model)
    got = _run_dmeas(mod, y, C, rho, psi)

    assert np.all(np.isfinite(got))
    err = np.abs(got - ref)
    worst = np.argmax(err / (EPS32 * kappa))
    assert np.all(err <= K * EPS32 * kappa), (
        f"y={y[worst]} C={C[worst]} rho={rho[worst]} psi={psi[worst]}: "
        f"got {got[worst]}, float64 {ref[worst]}"
    )
    # The grid covers upper-tail points that float64 resolves above the floor,
    # which the old float32 code floored at log(tol).
    floor = np.log(1e-12 if floor_model else 1e-18)
    assert np.sum((z_lo > 5.5) & (ref > floor + 1.0)) >= 10


@pytest.mark.parametrize("mod", TOL_MODELS + FLOOR_MODELS, ids=_name)
def test_dmeas_special_cases(mod):
    theta = {"rho": jnp.float32(0.5), "psi": jnp.float32(0.2)}
    nan_y = mod.dmeas(
        {"cases": jnp.float32(np.nan)}, {"C": jnp.float32(100.0)}, theta, None, None
    )
    assert float(nan_y) == 0.0
    if mod in TOL_MODELS:
        neg_C = mod.dmeas(
            {"cases": jnp.float32(10.0)}, {"C": jnp.float32(-1.0)}, theta, None, None
        )
        assert float(neg_C) == -np.inf


@pytest.mark.parametrize("mod", TOL_MODELS + FLOOR_MODELS, ids=_name)
def test_dmeas_gradients_finite(mod):
    y, C, rho, psi = _dmeas_grid()
    keep = C > 0  # the 00x models already have NaN gradients at C == 0 (sqrt(v))
    y, C, rho, psi = (jnp.asarray(a[keep], jnp.float32) for a in (y, C, rho, psi))

    def ll(C, rho, psi):
        theta = {"rho": rho, "psi": psi}
        return jnp.sum(mod.dmeas({"cases": y}, {"C": C}, theta, None, None))

    grads = jax.jit(jax.grad(ll, argnums=(0, 1, 2)))(C, rho, psi)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g)))

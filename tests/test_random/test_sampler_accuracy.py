"""Distributional accuracy of the approximate samplers in ``pypomp.random``.

These samplers are approximations (Giles & Beentjes for binomial, Temme for
gamma, a ported CURAND Poisson), so correctness is checked against the exact
``scipy.stats`` distributions rather than against another sampler. Wasserstein
distance is used alongside a quantile mismatch rate because it is sensitive to
the tail and location errors these approximations actually make.

Marked heavy: each check draws large samples.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import pypomp.random as ppr

pytestmark = pytest.mark.heavy


def test_poisson_quantile_mismatch_rate() -> None:
    from pypomp.random.poisson import poissoninv

    n_samples = 100_000
    key = jax.random.key(12345)
    u = jax.random.uniform(key, (n_samples,), dtype=jnp.float32)
    # Clip to avoid extreme tail float inaccuracies
    u_clip = jnp.clip(u, 1e-9, 1.0 - 1e-9)
    u_np = np.array(u_clip)

    lam_vals = [0.1, 1.0, 10.0, 100.0, 500.0]
    for lam in lam_vals:
        exact = stats.poisson.ppf(u_np, lam)
        lam_arr = jnp.full((n_samples,), lam, dtype=jnp.float32)
        fast = np.array(poissoninv(u_clip, lam_arr, dtype=jnp.float32))

        mismatches = np.sum(exact != fast)
        mismatch_rate = mismatches / n_samples
        assert mismatch_rate < 0.0001, (
            f"Poisson lam={lam} mismatch rate too high: {mismatch_rate:.4%}"
        )
        print(f"Poisson lam={lam} mismatch rate: {mismatch_rate:.4%}")


def test_binomial_quantile_mismatch_rate() -> None:
    from pypomp.random.binom import binominv

    n_samples = 100_000
    key = jax.random.key(12345)
    u = jax.random.uniform(key, (n_samples,), dtype=jnp.float32)
    u_clip = jnp.clip(u, 1e-9, 1.0 - 1e-9)
    u_np = np.array(u_clip)

    binom_params = [
        (10, 0.1),
        (10, 0.5),
        (10, 0.9),
        (100, 0.01),
        (100, 0.5),
        (100, 0.99),
        (1000, 0.1),
        (1000, 0.5),
        (1000, 0.9),
    ]
    for n_val, p_val in binom_params:
        exact = stats.binom.ppf(u_np, n_val, p_val)
        n_arr = jnp.full((n_samples,), n_val, dtype=jnp.float32)
        p_arr = jnp.full((n_samples,), p_val, dtype=jnp.float32)
        fast = np.array(
            binominv(u_clip, n_arr, p_arr, exact_max=5, order=2, dtype=jnp.float32)
        )

        mismatches = np.sum(exact != fast)
        mismatch_rate = mismatches / n_samples
        assert mismatch_rate < 0.0015, (
            f"Binomial n={n_val}, p={p_val} mismatch rate too high: {mismatch_rate:.4%}"
        )
        print(f"Binomial n={n_val}, p={p_val} mismatch rate: {mismatch_rate:.4%}")


def test_gamma_quantile_mismatch_rate() -> None:
    from pypomp.random.gamma import gammainv

    n_samples = 100_000
    key = jax.random.key(12345)
    u = jax.random.uniform(key, (n_samples,), dtype=jnp.float32)
    u_clip = jnp.clip(u, 1e-9, 1.0 - 1e-9)
    u_np = np.array(u_clip)

    alpha_vals = [3.0, 5.0, 10.0, 50.0, 100.0]
    for alpha in alpha_vals:
        exact = stats.gamma.ppf(u_np, alpha)
        alpha_arr = jnp.full((n_samples,), alpha, dtype=jnp.float32)
        fast = np.array(gammainv(u_clip, alpha_arr, dtype=jnp.float32, newton_steps=3))

        abs_diff = np.abs(exact - fast)
        rel_diff = abs_diff / np.maximum(exact, 1e-5)

        # Consider a mismatch if relative difference > 1e-3 AND absolute difference > 1e-4
        mismatches = np.sum((rel_diff > 1e-3) & (abs_diff > 1e-4))
        mismatch_rate = mismatches / n_samples

        assert mismatch_rate < 0.0001, (
            f"Gamma alpha={alpha} mismatch rate too high: {mismatch_rate:.6%}"
        )
        print(f"Gamma alpha={alpha} mismatch rate: {mismatch_rate:.6%}")


def test_poisson_quantile_wasserstein_distance() -> None:
    from pypomp.random.poisson import poissoninv

    n_samples = 100_000
    u = np.linspace(1e-5, 1 - 1e-5, n_samples)
    u_jnp = jnp.array(u)

    lam_vals = [0.1, 1.0, 10.0, 100.0, 500.0]
    for lam in lam_vals:
        fast = np.array(
            poissoninv(u_jnp, jnp.full((n_samples,), lam), dtype=jnp.float32)
        )
        exact = stats.poisson.ppf(u, lam)
        w1 = np.mean(np.abs(fast - exact))
        assert w1 < 0.0001, f"Poisson lam={lam} Wasserstein distance too high: {w1:.6f}"
        print(f"Poisson lam={lam} Wasserstein distance: {w1:.6f}")


def test_binomial_quantile_wasserstein_distance() -> None:
    from pypomp.random.binom import binominv

    n_samples = 100_000
    u = np.linspace(1e-5, 1 - 1e-5, n_samples)
    u_jnp = jnp.array(u)

    binom_params = [
        (10, 0.1),
        (10, 0.5),
        (10, 0.9),
        (100, 0.01),
        (100, 0.5),
        (100, 0.99),
        (1000, 0.1),
        (1000, 0.5),
        (1000, 0.9),
    ]
    for n_val, p_val in binom_params:
        fast = np.array(
            binominv(
                u_jnp,
                jnp.full((n_samples,), n_val),
                jnp.full((n_samples,), p_val),
                exact_max=5,
                order=2,
                dtype=jnp.float32,
            )
        )
        exact = stats.binom.ppf(u, n_val, p_val)
        w1 = np.mean(np.abs(fast - exact))
        assert w1 < 0.001, (
            f"Binomial n={n_val}, p={p_val} Wasserstein distance too high: {w1:.6f}"
        )
        print(f"Binomial n={n_val}, p={p_val} Wasserstein distance: {w1:.6f}")


def test_gamma_quantile_wasserstein_distance() -> None:
    from pypomp.random.gamma import gammainv

    n_samples = 100_000
    u = np.linspace(1e-5, 1 - 1e-5, n_samples)
    u_jnp = jnp.array(u)

    alpha_vals = [3.0, 5.0, 10.0, 50.0, 100.0]
    for alpha in alpha_vals:
        fast = np.array(
            gammainv(
                u_jnp, jnp.full((n_samples,), alpha), dtype=jnp.float32, newton_steps=3
            )
        )
        exact = stats.gamma.ppf(u, alpha)
        w1 = np.mean(np.abs(fast - exact))
        assert w1 < 0.0001, (
            f"Gamma alpha={alpha} Wasserstein distance too high: {w1:.6f}"
        )
        print(f"Gamma alpha={alpha} Wasserstein distance: {w1:.6f}")


def test_nbinomial_empirical_wasserstein_distance() -> None:
    n_samples = 100_000
    u = np.linspace(1e-5, 1 - 1e-5, n_samples)

    key = jax.random.key(54321)
    nbinom_params = [(1.0, 0.1), (1.0, 0.5), (5.0, 0.5), (20.0, 0.9), (100.0, 0.5)]
    for n_val, p_val in nbinom_params:
        key, subkey = jax.random.split(key)
        samples = np.sort(
            np.array(
                ppr.fast_nbinomial(
                    subkey,
                    jnp.full((n_samples,), n_val),
                    p=jnp.full((n_samples,), p_val),
                )
            )
        )
        exact = stats.nbinom.ppf(u, n_val, p_val)
        w1 = np.mean(np.abs(samples - exact))
        # Allow headroom for sampling variance (since sample has standard deviation up to ~14)
        assert w1 < 0.10, (
            f"NBinomial n={n_val}, p={p_val} empirical Wasserstein distance too high: {w1:.6f}"
        )
        print(
            f"NBinomial n={n_val}, p={p_val} empirical Wasserstein distance: {w1:.6f}"
        )


# ---------------------------------------------------------------------------
# End-to-end sampler accuracy
#
# The quantile checks above exercise the inverse-CDF functions directly. These
# drive the public samplers instead, so the accuracy parameters they pass
# through (max_newton_loops, max_inverse_cdf_loops) are covered too.
# ---------------------------------------------------------------------------

# Samples per check. The 1-Wasserstein sampling-noise floor scales as
# 1/sqrt(N_SAMPLES), which is ~0.002 here in units of the distribution's sd.
N_SAMPLES = 200_000

# Observed normalised distances for the current samplers are all <= 0.007,
# so this leaves roughly 3x headroom over the worst regime while still
# failing on any gross distributional error.
W1_TOL = 0.02


def _normalised_w1(samples: jax.Array, dist: Any) -> float:
    """1-Wasserstein distance to ``dist``, in units of its standard deviation.

    Computed in quantile space: for one dimension the 1-Wasserstein distance is
    the mean absolute gap between the sorted sample and the exact quantiles at
    the same plotting positions.
    """
    sorted_samples = np.sort(np.asarray(samples, dtype=np.float64))
    exact_quantiles = dist.ppf((np.arange(N_SAMPLES) + 0.5) / N_SAMPLES)
    scale = dist.std()
    return float(
        np.mean(np.abs(sorted_samples - exact_quantiles))
        / (scale if scale > 0 else 1.0)
    )


def _check_sampler(name: str, samples: jax.Array, dist: Any, params: str) -> None:
    w1 = _normalised_w1(samples, dist)
    assert w1 < W1_TOL, f"{name} {params}: normalised W1={w1:.5f} exceeds {W1_TOL}"


@pytest.mark.parametrize("lam", [0.1, 1.0, 4.0, 10.0, 20.0, 100.0, 500.0])
def test_fast_poisson_wasserstein(lam: float) -> None:
    """fast_poisson matches the exact Poisson across its algorithmic regimes."""
    key = jax.random.key(2026)
    samples = ppr.fast_poisson(key, jnp.full((N_SAMPLES,), lam, dtype=jnp.float32))
    _check_sampler("Poisson", samples, stats.poisson(lam), f"lam={lam}")


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 10.0, 100.0])
def test_fast_gamma_wasserstein(a: float) -> None:
    """fast_gamma matches the exact gamma, including the a < 1 regime."""
    key = jax.random.key(2026)
    samples = ppr.fast_gamma(key, jnp.full((N_SAMPLES,), a, dtype=jnp.float32))
    _check_sampler("Gamma", samples, stats.gamma(a), f"a={a}")


@pytest.mark.parametrize("n, p", [(10, 0.5), (100, 0.01), (100, 0.5), (1000, 0.9)])
def test_fast_binomial_wasserstein(n: int, p: float) -> None:
    """fast_binomial matches the exact binomial across n/p regimes."""
    key = jax.random.key(2026)
    samples = ppr.fast_binomial(
        key,
        jnp.full((N_SAMPLES,), n, dtype=jnp.float32),
        jnp.full((N_SAMPLES,), p, dtype=jnp.float32),
    )
    _check_sampler("Binomial", samples, stats.binom(n, p), f"n={n}, p={p}")


@pytest.mark.parametrize("r, p", [(1.0, 0.5), (5.0, 0.3), (20.0, 0.7)])
def test_fast_nbinomial_wasserstein(r: float, p: float) -> None:
    """fast_nbinomial matches the exact negative binomial."""
    key = jax.random.key(2026)
    samples = ppr.fast_nbinomial(
        key,
        jnp.full((N_SAMPLES,), r, dtype=jnp.float32),
        jnp.full((N_SAMPLES,), p, dtype=jnp.float32),
    )
    _check_sampler("NBinomial", samples, stats.nbinom(r, p), f"r={r}, p={p}")

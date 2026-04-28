import numpy as np
import pytest

from pypomp.mcap import MCAPResult, mcap, _qchisq


def _quadratic_profile(rng, n=60, center=0.3, curvature=0.5, noise=0.05):
    x = np.linspace(center - 2.0, center + 2.0, n)
    y = -curvature * (x - center) ** 2 + rng.normal(0.0, noise, size=n)
    return x, y


def test_mcap_returns_result_with_expected_fields():
    rng = np.random.default_rng(0)
    x, y = _quadratic_profile(rng)
    result = mcap(x, y)

    assert isinstance(result, MCAPResult)
    assert result.level == 0.95
    assert set(result.fit.keys()) == {"parameter", "smoothed", "quadratic"}
    assert result.fit["parameter"].shape == result.fit["smoothed"].shape
    assert result.fit["parameter"].shape == result.fit["quadratic"].shape
    assert result.vcov.shape == (2, 2)
    assert set(result.quadratic_coef.keys()) == {"a", "b", "c"}


def test_mcap_recovers_mle_and_finite_ci_on_quadratic():
    rng = np.random.default_rng(1)
    true_center = 0.3
    x, y = _quadratic_profile(rng, center=true_center)
    result = mcap(x, y)

    # MLE should be close to the true peak
    assert abs(result.mle - true_center) < 0.2

    lo, hi = result.ci
    assert lo is not None and hi is not None
    assert lo < result.mle < hi

    # se_total combines stat and MC components
    assert result.se_stat > 0
    assert result.se_mc >= 0
    assert result.se_total >= result.se_stat

    # delta should be positive when curvature is well-defined
    assert result.delta > 0


def test_mcap_grid_size_matches_n_grid():
    rng = np.random.default_rng(2)
    x, y = _quadratic_profile(rng)
    n_grid = 250
    result = mcap(x, y, n_grid=n_grid)
    assert result.fit["parameter"].shape == (n_grid,)


def test_mcap_loess_degree_1_runs():
    rng = np.random.default_rng(3)
    x, y = _quadratic_profile(rng)
    result = mcap(x, y, loess_degree=1)
    assert isinstance(result, MCAPResult)
    assert np.isfinite(result.mle)


def test_mcap_constant_parameter_returns_finite_result():
    # All x identical exercises:
    #   - _loess_smooth_1d degenerate-scale branch (returns flat line)
    #   - _fit_local_quadratic maxdist == 0 branch (uniform weights)
    #   - _fit_local_quadratic singular-matrix fallback (pinv)
    rng = np.random.default_rng(4)
    x = np.full(20, 2.0)
    y = rng.normal(0.0, 1.0, size=20)

    result = mcap(x, y)
    assert isinstance(result, MCAPResult)
    assert result.mle == 2.0
    # smoothed profile is flat at mean(y)
    assert np.allclose(result.fit["smoothed"], float(np.mean(y)))


def test_mcap_higher_level_widens_delta():
    rng = np.random.default_rng(5)
    x, y = _quadratic_profile(rng)
    r95 = mcap(x, y, level=0.95)
    r99 = mcap(x, y, level=0.99)
    assert r99.delta > r95.delta


def test_qchisq_matches_scipy():
    # Sanity check on the private helper; values are well-known.
    assert _qchisq(0.95, df=1) == pytest.approx(3.841458, rel=1e-4)
    assert _qchisq(0.99, df=1) == pytest.approx(6.634897, rel=1e-4)

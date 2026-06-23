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


def test_mcap_monotonic_profile_signals_failed_fit():
    # When the parameter range does not bracket a local maximum, the local
    # quadratic fit returns non-positive curvature (a <= 0). This is a real
    # MCAP failure mode; the function falls back to (None, None) for the CI
    # and to the smoothed MLE for quadratic_max.
    rng = np.random.default_rng(4)
    x = np.linspace(0.0, 1.0, 40)
    y = x + rng.normal(0.0, 0.05, size=40)

    import warnings

    with warnings.catch_warnings():
        # se_stat2 = 1/(2a) is negative when a < 0; sqrt(se_stat2) -> NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mcap(x, y)

    assert result.quadratic_coef["a"] < 0.0
    assert result.quadratic_max == result.mle
    assert result.delta < 0.0
    assert result.ci == (None, None)


def test_mcap_small_sample_uses_zero_residual_variance():
    # With only 5 points, the local fit's effective degrees of freedom can be
    # <= 0, exercising the s2 = 0.0 branch in _fit_local_quadratic.
    rng = np.random.default_rng(0)
    x = np.linspace(-1.0, 1.0, 5)
    y = -0.5 * x**2 + rng.normal(0.0, 0.05, size=5)
    result = mcap(x, y, span=0.75)
    assert isinstance(result, MCAPResult)
    assert np.isfinite(result.mle)


def test_qchisq_matches_scipy():
    # Sanity check on the private helper; values are well-known.
    assert _qchisq(0.95, df=1) == pytest.approx(3.841458, rel=1e-4)
    assert _qchisq(0.99, df=1) == pytest.approx(6.634897, rel=1e-4)


def test_mcap_r_comparison():
    # Hardcoded test data used to compare with R pomp::mcap
    x = np.linspace(0.1, 0.5, 10)
    y = -0.5 * (x - 0.3) ** 2 + np.array(
        [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.03, -0.01]
    )

    # R pomp::mcap reference outputs:
    # mle: 0.23293293
    # ci: (0.1, 0.5)
    # delta: 1.93755414
    # se_stat: 0.35499986
    # se_mc: 0.03322532
    # se_total: 0.35655129
    # quadratic_max: 0.23196971

    result = mcap(x, y)

    # MLE and quadratic peak should be close to R values
    assert abs(result.mle - 0.23293293) < 0.05
    assert abs(result.quadratic_max - 0.23196971) < 0.05

    # Confidence intervals should match
    assert result.ci[0] is not None and result.ci[1] is not None
    assert abs(result.ci[0] - 0.1) < 1e-5
    assert abs(result.ci[1] - 0.5) < 1e-5

    # Delta and SE components should match within a reasonable tolerance
    # due to loess implementation differences
    assert abs(result.delta - 1.93755414) < 0.1
    assert abs(result.se_stat - 0.35499986) < 0.1
    assert abs(result.se_mc - 0.03322532) < 0.05
    assert abs(result.se_total - 0.35655129) < 0.1


def test_mcap_loess_linalg_error_fallback():
    # Construct input data that triggers mad = 0 (perfect fit in local window)
    # leading to LinAlgError in standard loess_1d robust iterations.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Calling mcap on this should trigger the fallback, raise the RuntimeWarning,
    # and produce a valid, non-degenerate smoothed profile.
    with pytest.warns(RuntimeWarning, match="LinAlgError in loess_1d"):
        result = mcap(x, y, span=0.75, loess_degree=1)

    # Verify the smoothed profile is NOT degenerate (all zeroes)
    assert not np.allclose(result.fit["smoothed"], 0.0)
    assert np.max(result.fit["smoothed"]) > 3.0

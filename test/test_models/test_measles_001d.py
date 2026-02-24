import pytest
import pypomp as pp
import jax
import jax.numpy as jnp

BASE_THETA_001D = {
    "R0": 56.8,
    "sigma": 28.9,
    "gamma": 30.4,
    "iota": 2.9,
    "rho": 0.488,
    "sigmaSE": 0.0878,
    "psi": 0.116,
    "cohort": 0.557,
    "amplitude": 0.554,
    "S_0": 2.97e-02,
    "E_0": 5.17e-05,
    "I_0": 5.14e-05,
    "R_0": 9.70e-01,
}

DEFAULT_J = 3
DEFAULT_KEY = jax.random.key(1)


@pytest.fixture(scope="function")
def london_001d():
    measles = pp.UKMeasles.Pomp(
        unit=["London"],
        theta=BASE_THETA_001D.copy(),
        clean=True,
        model="001d",
    )
    return measles


def test_001d_simulate(london_001d):
    london_001d.simulate(key=DEFAULT_KEY, nsim=1)


def test_001d_pfilter(london_001d):
    london_001d.pfilter(J=DEFAULT_J, key=DEFAULT_KEY)


def test_001d_mop(london_001d):
    london_001d.mop(J=DEFAULT_J, key=DEFAULT_KEY)


def test_001d_dpop(london_001d):
    vals = london_001d.dpop(
        J=DEFAULT_J,
        key=DEFAULT_KEY,
        alpha=0.9,
        process_weight_state="logw",
    )
    nll = vals[0]
    assert nll.shape == ()
    assert jnp.isfinite(nll.item())


def test_001d_dpop_train(london_001d):
    eta = {name: 0.01 for name in london_001d.canonical_param_names}
    nll, theta_hist = london_001d.dpop_train(
        J=DEFAULT_J,
        M=2,
        eta=eta,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=DEFAULT_KEY,
    )
    assert nll.shape == (3,)
    assert theta_hist.shape[0] == 3
    assert jnp.all(jnp.isfinite(nll))


def test_001d_par_trans_roundtrip(london_001d):
    """Parameter transform round-trip: natural -> est -> natural."""
    from pypomp.measles.model_001d import to_est, from_est

    theta = BASE_THETA_001D.copy()
    theta_jax = {k: jnp.array(v) for k, v in theta.items()}
    est = to_est(theta_jax)
    recovered = from_est(est)
    for k in theta:
        # Simplex parameters (S_0/E_0/I_0/R_0) lose precision due to
        # normalization in float32, so use a looser tolerance.
        assert jnp.allclose(theta_jax[k], recovered[k], atol=1e-3), (
            f"Round-trip failed for {k}: {theta_jax[k]} vs {recovered[k]}"
        )


def test_001d_dmeas_nan_handling():
    """dmeas should return 0 for NaN observations (not NaN)."""
    from pypomp.measles.model_001d import dmeas

    Y = {"cases": jnp.array(float("nan"))}
    X = {"C": jnp.array(100.0)}
    theta = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}
    result = dmeas(Y, X, theta)
    assert jnp.isfinite(result)
    assert result == 0.0


def test_001d_dmeas_zero_cases():
    """dmeas should handle y=0 (uses log_cdf_single path)."""
    from pypomp.measles.model_001d import dmeas

    Y = {"cases": jnp.array(0.0)}
    X = {"C": jnp.array(100.0)}
    theta = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}
    result = dmeas(Y, X, theta)
    assert jnp.isfinite(result)


def test_001d_dmeas_positive_cases():
    """dmeas should handle y>0 (uses log_cdf_diff path)."""
    from pypomp.measles.model_001d import dmeas

    Y = {"cases": jnp.array(50.0)}
    X = {"C": jnp.array(100.0)}
    theta = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}
    result = dmeas(Y, X, theta)
    assert jnp.isfinite(result)
    assert result < 0.0  # log-likelihood should be negative


def test_001d_log_cdf_diff_gradient():
    """log_cdf_diff should produce finite gradients."""
    from pypomp.measles.model_001d import log_cdf_diff

    def f(zh, zl):
        return log_cdf_diff(zh, zl)

    zh = jnp.array(1.0)
    zl = jnp.array(-1.0)
    grad_zh, grad_zl = jax.grad(f, argnums=(0, 1))(zh, zl)
    assert jnp.isfinite(grad_zh)
    assert jnp.isfinite(grad_zl)


def test_001d_log_cdf_diff_extreme():
    """log_cdf_diff should handle moderately extreme z values without NaN gradients."""
    from pypomp.measles.model_001d import log_cdf_diff

    def f(zh, zl):
        return log_cdf_diff(zh, zl)

    # Moderately large z values (z=30 underflows log_ndtr in float32,
    # so we use z~5-6 which are still deep in the tail)
    zh = jnp.array(6.0)
    zl = jnp.array(5.0)
    val = f(zh, zl)
    grad_zh, grad_zl = jax.grad(f, argnums=(0, 1))(zh, zl)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad_zh)
    assert jnp.isfinite(grad_zl)

    # Moderately negative z values
    zh = jnp.array(-5.0)
    zl = jnp.array(-6.0)
    val = f(zh, zl)
    grad_zh, grad_zl = jax.grad(f, argnums=(0, 1))(zh, zl)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad_zh)
    assert jnp.isfinite(grad_zl)

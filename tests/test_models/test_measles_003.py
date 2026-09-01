"""
Focused unit tests for pypomp.models.measles.model_003.

model_003 ("continuous process model" variant) is exercised generically via
test_other_models in test_measles.py (simulate/pfilter), but several
module-level helper functions are not reached through that path:

- dmeas_continuous: an alternate Gaussian measurement density that is not
  wired up as the model's `dmeas` (the model uses the discretized `dmeas`
  defined later in the module), so it must be tested directly.
- _log_phi / _log_cdf_diff_jvp: only exercised when a gradient is taken
  through log_cdf_diff (e.g. under `mif`/`train`-style estimation), which
  the generic simulate/pfilter smoke test never triggers.
- to_est / from_est: the parameter transforms, only invoked when converting
  between natural and estimation scales (e.g. during `mif`), not by a bare
  pfilter/simulate call.
"""

import jax
import jax.numpy as jnp
import pytest

from pypomp.models.measles.model_003 import (
    dmeas,
    dmeas_continuous,
    from_est,
    log_cdf_diff,
    log_cdf_single,
    to_est,
)
from pypomp.types import ObservationDict, ParamDict, StateDict

BASE_THETA_003 = {
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


def test_003_dmeas_continuous_basic():
    """dmeas_continuous should return a finite log-density for typical inputs."""
    Y = {"cases": jnp.array(50.0)}
    X = {"C": jnp.array(100.0)}
    theta = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}
    ll = dmeas_continuous(Y, X, theta)
    assert jnp.isfinite(ll)
    assert ll < 0.0


def test_003_dmeas_continuous_nan_observation():
    """NaN observations should be masked to a log-likelihood of 0, not NaN."""
    Y = {"cases": jnp.array(float("nan"))}
    X = {"C": jnp.array(100.0)}
    theta = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}
    ll = dmeas_continuous(Y, X, theta)
    assert jnp.isfinite(ll)
    assert ll == 0.0


def test_003_dmeas_continuous_negative_C():
    """Negative true-case counts C should be penalized with -inf log-likelihood."""
    Y = {"cases": jnp.array(10.0)}
    X = {"C": jnp.array(-5.0)}
    theta = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}
    ll = dmeas_continuous(Y, X, theta)
    assert ll == -jnp.inf


def test_003_dmeas_continuous_vectorized():
    """dmeas_continuous should broadcast over arrays of observations/states."""
    Y = {"cases": jnp.array([0.0, 20.0, 100.0])}
    X = {"C": jnp.array([5.0, 25.0, 90.0])}
    theta = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}
    ll = dmeas_continuous(Y, X, theta)
    assert ll.shape == (3,)
    assert jnp.all(jnp.isfinite(ll))


def test_003_dmeas_zero_and_positive_cases():
    """The discrete dmeas used by the model should be finite for both branches."""
    theta: ParamDict = {"rho": jnp.array(0.5), "psi": jnp.array(0.1)}

    Y0: ObservationDict = {"cases": jnp.array(0.0)}
    X: StateDict = {"C": jnp.array(100.0)}
    ll0 = dmeas(Y0, X, theta)
    assert jnp.isfinite(ll0)

    Ypos: ObservationDict = {"cases": jnp.array(50.0)}
    llpos = dmeas(Ypos, X, theta)
    assert jnp.isfinite(llpos)
    assert llpos < 0.0


def test_003_log_cdf_diff_gradient():
    """log_cdf_diff should produce finite gradients (exercises _log_phi and the
    custom jvp rule)."""

    def f(zh, zl):
        return log_cdf_diff(zh, zl)

    zh = jnp.array(1.0)
    zl = jnp.array(-1.0)
    val = f(zh, zl)
    grad_zh, grad_zl = jax.grad(f, argnums=(0, 1))(zh, zl)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad_zh)
    assert jnp.isfinite(grad_zl)


def test_003_log_cdf_diff_extreme():
    """log_cdf_diff / its custom jvp should stay finite deep in the tails."""

    def f(zh, zl):
        return log_cdf_diff(zh, zl)

    zh = jnp.array(6.0)
    zl = jnp.array(5.0)
    val = f(zh, zl)
    grad_zh, grad_zl = jax.grad(f, argnums=(0, 1))(zh, zl)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad_zh)
    assert jnp.isfinite(grad_zl)

    zh = jnp.array(-5.0)
    zl = jnp.array(-6.0)
    val = f(zh, zl)
    grad_zh, grad_zl = jax.grad(f, argnums=(0, 1))(zh, zl)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad_zh)
    assert jnp.isfinite(grad_zl)


def test_003_log_cdf_single_gradient():
    """log_cdf_single (the y=0 boundary case) should also have a finite gradient."""

    def f(z):
        return log_cdf_single(z)

    z = jnp.array(0.3)
    val = f(z)
    grad_z = jax.grad(f)(z)
    assert jnp.isfinite(val)
    assert jnp.isfinite(grad_z)


def test_003_to_est_from_est_roundtrip():
    """to_est/from_est should be inverses of each other on natural-scale params.

    The S_0/E_0/I_0/R_0 initial proportions are renormalized to sum to 1 by
    the transform (they represent compositional proportions), so the
    round-trip is checked against the pre-normalized values for those.
    """
    theta_nat: ParamDict = {k: jnp.array(v) for k, v in BASE_THETA_003.items()}

    est = to_est(theta_nat)
    assert set(est.keys()) == set(theta_nat.keys())

    nat2 = from_est(est)

    seir_keys = {"S_0", "E_0", "I_0", "R_0"}
    seir_sum = sum(theta_nat[k] for k in seir_keys)
    for k, v in theta_nat.items():
        expected = v / seir_sum if k in seir_keys else v
        assert jnp.allclose(nat2[k], expected, rtol=1e-5, atol=1e-8), (
            f"round-trip mismatch for {k}"
        )


def test_003_to_est_log_transforms():
    """Spot-check that to_est applies the documented transforms."""
    theta_nat: ParamDict = {k: jnp.array(v) for k, v in BASE_THETA_003.items()}
    est = to_est(theta_nat)

    assert jnp.allclose(est["R0"], jnp.log(theta_nat["R0"]))
    assert jnp.allclose(est["sigma"], jnp.log(theta_nat["sigma"]))

    # S_0/E_0/I_0/R_0 are log-ratio (compositional) transformed: exponentiating
    # and renormalizing should recover the original proportions.
    seir_est = jnp.array([est["S_0"], est["E_0"], est["I_0"], est["R_0"]])
    seir_nat = jnp.exp(seir_est)
    seir_nat = seir_nat / jnp.sum(seir_nat)
    seir_orig = jnp.array(
        [theta_nat["S_0"], theta_nat["E_0"], theta_nat["I_0"], theta_nat["R_0"]]
    )
    seir_orig = seir_orig / jnp.sum(seir_orig)
    assert jnp.allclose(seir_nat, seir_orig, rtol=1e-5)


@pytest.fixture(scope="module")
def london_003():
    import pypomp as pp

    theta = BASE_THETA_003.copy()
    measles = pp.models.UKMeasles.pomp(
        unit="London",
        theta=pp.PompParameters(theta),
        model="003",
        last_year=1951,  # Use less data for faster testing
    )
    return measles


def test_003_simulate_shapes(london_003):
    """Sanity-check that model 003 simulation runs and produces finite states."""
    key = jax.random.key(3)
    states_df, obs_df = london_003.simulate(key=key, nsim=1)
    assert len(states_df) > 0
    assert len(obs_df) > 0
    for col in ["S", "E", "I", "R", "W", "C"]:
        assert jnp.all(jnp.isfinite(jnp.asarray(states_df[col])))

import jax
import pandas as pd
import pytest

import pypomp as pp


@pytest.fixture(scope="function")
def simple():
    LG = pp.models.lg()
    return LG


@pytest.mark.parametrize("ntheta, nsim", [(1, 1), (1, 3), (3, 1), (3, 3)])
def test_simulate(ntheta, nsim, simple):
    LG = simple
    key = jax.random.key(111)
    ys = LG.ys
    theta = LG.theta * ntheta
    X_sims, Y_sims = LG.simulate(nsim=nsim, key=key, theta=theta)

    assert isinstance(X_sims, pd.DataFrame)
    assert isinstance(Y_sims, pd.DataFrame)
    assert X_sims.shape == ((len(ys) + 1) * nsim * len(theta), 5)
    assert Y_sims.shape == (len(ys) * nsim * len(theta), 5)


def test_simulate_invalid_theta_keys(simple):
    """theta with non-canonical keys should raise an error."""
    LG = simple
    key = jax.random.key(111)
    bad_theta = {"not_a_param": 1.0}

    with pytest.raises(
        ValueError,
        match="theta parameter names must match canonical_param_names up to reordering",
    ):
        LG.simulate(nsim=1, key=key, theta=pp.PompParameters(bad_theta))


def test_simulate_as_pomp(simple):
    LG = simple
    key = jax.random.key(0)

    # Test normal as_pomp
    new_pomp = LG.simulate(nsim=1, key=key, as_pomp=True)
    assert isinstance(new_pomp, pp.Pomp)
    assert new_pomp.ys.shape == LG.ys.shape
    assert not new_pomp.ys.equals(LG.ys)
    assert new_pomp.theta.num_replicates() == 1

    # Test as_pomp with nsim > 1 (should warn and force nsim=1)
    with pytest.warns(UserWarning, match="as_pomp is True, but nsim > 1"):
        new_pomp_warn = LG.simulate(nsim=5, key=key, as_pomp=True)
    assert isinstance(new_pomp_warn, pp.Pomp)
    assert new_pomp_warn.theta.num_replicates() == 1

    # Verify that the simulated Pomp can be simulated again
    result = new_pomp.simulate(nsim=1, key=key)
    assert isinstance(result, tuple)
    X, Y = result
    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.DataFrame)

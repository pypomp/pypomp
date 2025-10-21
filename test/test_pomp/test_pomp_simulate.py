import jax
import pytest
import pypomp as pp
import pandas as pd


@pytest.fixture(scope="function")
def simple():
    LG = pp.LG()
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


def test_simulate_param_order_invariance(simple):
    LG = simple
    key = jax.random.key(1234)
    theta = LG.theta
    nsim = 1
    X_sims, Y_sims = LG.simulate(nsim=nsim, key=key, theta=theta)

    param_keys = list(theta[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta]

    X2, Y2 = LG.simulate(nsim=nsim, key=key, theta=permuted_theta)
    pd.testing.assert_frame_equal(X_sims, X2)
    pd.testing.assert_frame_equal(Y_sims, Y2)

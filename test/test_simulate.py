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

import jax
import pytest
import pypomp as pp


@pytest.fixture(scope="function")
def simple():
    LG = pp.LG()
    key = jax.random.key(111)
    J = 5
    ys = LG.ys
    theta = LG.theta
    covars = LG.covars
    nsim = 1
    return LG, key, J, ys, theta, covars, nsim


def test_internal_basic(simple):
    LG, key, J, ys, theta, covars, nsim = simple
    val = LG.simulate(nsim=nsim, key=key)

    assert isinstance(val, list)
    assert isinstance(val[0], dict)
    assert "X_sims" in val[0]
    assert "Y_sims" in val[0]
    assert val[0]["X_sims"].shape == (len(ys) + 1, LG.rmeas.ydim, nsim)

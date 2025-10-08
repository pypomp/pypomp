import jax
import pytest
import pypomp as pp


@pytest.fixture(scope="function")
def simple_setup():
    spx_model = pp.spx()
    J = 3
    key = jax.random.key(111)
    theta = spx_model.theta
    return spx_model, J, key, theta


@pytest.fixture(scope="function")
def simple(simple_setup):
    spx_model, J, key, theta = simple_setup
    spx_model.results_history.clear()
    spx_model.theta = theta
    return spx_model, J, key


def test_spx_pfilter_basic(simple):
    spx_model, J, key = simple
    spx_model.pfilter(J=J, key=key, reps=1)
    assert isinstance(spx_model.results_history, list)
    assert len(spx_model.results_history) > 0


def test_spx_mif_basic(simple):
    spx_model, J, key = simple
    spx_model.mif(
        sigmas=0.02,
        sigmas_init=0.1,
        J=J,
        key=key,
        M=1,
        a=0.5,
    )
    assert isinstance(spx_model.results_history, list)
    assert len(spx_model.results_history) > 0

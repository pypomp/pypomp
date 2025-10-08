import jax
import jax.numpy as jnp
import pytest
import pypomp as pp


@pytest.fixture(scope="function")
def simple():
    LG = pp.LG()
    J = 2
    ys = LG.ys
    theta = LG.theta
    covars = LG.covars
    sigmas = 0.02
    key = jax.random.key(111)
    return (LG, J, ys, theta, covars, sigmas, key)


def test_class_basic(simple):
    LG, J, ys, theta, covars, sigmas, key = simple
    val = LG.mop(J=J, key=key)
    assert val[0].shape == ()
    assert jnp.isfinite(val[0].item())
    assert val[0].dtype == jnp.float32

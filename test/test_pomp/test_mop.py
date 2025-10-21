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


def test_mop_param_order_invariance(simple):
    LG, J, ys, theta, covars, sigmas, key = simple
    val1 = LG.mop(J=J, key=key, theta=theta)
    param_keys = list(theta[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta]
    val2 = LG.mop(J=J, key=key, theta=permuted_theta)
    assert jnp.allclose(val1[0], val2[0], atol=1e-7), (
        f"MOP result changed after theta reordering: {val1[0]} vs {val2[0]}"
    )

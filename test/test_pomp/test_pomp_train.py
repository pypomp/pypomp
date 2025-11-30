import jax
import pytest
import numpy as np
import pypomp as pp
import jax.numpy as jnp
from pypomp.train import dpop_sgd_decay


@pytest.fixture(scope="function")
def simple():
    # Set default values for tests
    LG = pp.LG()
    ys = LG.ys
    covars = None
    theta = LG.theta
    J = 2
    key = jax.random.key(111)
    M = 2
    return LG, ys, covars, theta, J, key, M


@pytest.mark.parametrize("optimizer", ["SGD", "Newton", "WeightedNewton", "BFGS"])
def test_class_GD_basic(optimizer, simple):
    LG, ys, covars, theta, J, key, M = simple
    LG.train(
        J=J,
        M=M,
        eta=0.2,
        optimizer=optimizer,
        scale=True,
        key=key,
    )
    GD_out = LG.results_history[-1]
    traces = GD_out.traces_da
    # Check shape for first replicate
    assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
    # +1 for logLik column
    # Check that "logLik" is in variable coordinate
    assert "logLik" in list(traces.coords["variable"].values)
    # Check that all parameter names are in variable coordinate
    for param in LG.theta[0].keys():
        assert param in list(traces.coords["variable"].values)
    assert all(isinstance(v, float) for v in LG.theta[0].values())


def test_class_GD_ls(simple):
    LG, ys, covars, theta, J, key, M = simple
    LG.train(
        J=J, M=M, eta=0.2, optimizer="SGD", scale=True, ls=True, n_monitors=1, key=key
    )
    GD_out = LG.results_history[-1]
    traces = GD_out.traces_da
    assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
    assert "logLik" in list(traces.coords["variable"].values)


def test_invalid_GD_input(simple):
    LG, ys, covars, theta, J, key, M = simple
    with pytest.raises(ValueError):
        # Check that an error is thrown when J is not positive
        LG.train(J=0, M=M, eta=0.2, scale=True, ls=True, key=key)


def test_train_param_order_invariance(simple):
    LG, ys, covars, theta, J, key, M = simple
    LG.train(
        J=J,
        M=M,
        eta=0.2,
        optimizer="Newton",
        scale=True,
        key=key,
        theta=theta,
    )
    out1 = LG.results_history[-1].traces_da.values
    param_keys = list(theta[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta]
    LG.train(
        J=J,
        M=M,
        eta=0.2,
        optimizer="Newton",
        scale=True,
        n_monitors=0,
        key=key,
        theta=permuted_theta,
    )
    out2 = LG.results_history[-1].traces_da.values
    np.testing.assert_allclose(out1, out2, atol=1e-7)

@pytest.fixture(scope="function")
def simple_sis_for_dpop():
    """
    Build a small SIS Pomp model for testing the DPOP SGD-with-decay optimizer.

    We keep T and J small so that the test is fast.
    """
    key_model = jax.random.key(2025)
    # This assumes you have pp.SIS exposed in pypomp.__init__
    model = pp.SIS(T=50, key=key_model)
    return model

def test_pomp_dpop_sgd_decay_param_order_invariance(simple_sis_for_dpop):
    """
    """
    model = simple_sis_for_dpop

    # Use small J,M so the test is fast, but KEEP THEM IDENTICAL across runs.
    J = 50
    M = 4

    # First run: default theta ordering
    key1 = jax.random.key(123)
    nll1, theta_hist1 = model.dpop_sgd_decay(
        J=J,
        M=M,
        eta0=0.01,
        decay=0.1,
        alpha=0.8,
        key=key1,
    )

    # Build a permuted theta with reversed key order
    theta_orig = model.theta  # list[dict]
    param_keys = list(theta_orig[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta_orig]

    # Second run: same random key & same hyper-parameters, but permuted theta
    key2 = jax.random.key(123)
    nll2, theta_hist2 = model.dpop_sgd_decay(
        J=J,
        M=M,
        eta0=0.01,
        decay=0.1,
        alpha=0.8,
        key=key2,
        theta=permuted_theta,
    )

    # Histories should match exactly up to numerical precision
    np.testing.assert_allclose(nll1, nll2, atol=1e-7)
    np.testing.assert_allclose(theta_hist1, theta_hist2, atol=1e-7)

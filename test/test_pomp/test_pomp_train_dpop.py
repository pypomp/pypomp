import jax
import pytest
import numpy as np
import pypomp as pp
import jax.numpy as jnp

J_DEFAULT = 2
M_DEFAULT = 2


@pytest.fixture(scope="function")
def simple_sir_for_dpop():
    """
    Build a small SIR Pomp model for testing the DPOP optimizers.
    """
    model = pp.sir()
    return model


def test_dpop_train_adam(simple_sir_for_dpop):
    """
    Test dpop_train with Adam optimizer.
    """
    model = simple_sir_for_dpop
    eta = {name: 0.01 for name in model.canonical_param_names}
    nll, theta_hist = model.dpop_train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(42),
    )
    assert nll.shape == (M_DEFAULT + 1,)
    assert theta_hist.shape[0] == M_DEFAULT + 1
    assert jnp.all(jnp.isfinite(nll))


def test_dpop_train_sgd(simple_sir_for_dpop):
    """
    Test dpop_train with SGD optimizer and decay.
    """
    model = simple_sir_for_dpop
    eta = {name: 0.01 for name in model.canonical_param_names}
    nll, theta_hist = model.dpop_train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer="SGD",
        alpha=0.8,
        decay=0.1,
        process_weight_state="logw",
        key=jax.random.key(42),
    )
    assert nll.shape == (M_DEFAULT + 1,)
    assert theta_hist.shape[0] == M_DEFAULT + 1
    assert jnp.all(jnp.isfinite(nll))


def test_dpop_train_adam_with_decay(simple_sir_for_dpop):
    """
    Test dpop_train with Adam optimizer and LR decay.
    """
    model = simple_sir_for_dpop
    eta = {name: 0.01 for name in model.canonical_param_names}
    nll, theta_hist = model.dpop_train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer="Adam",
        alpha=0.8,
        decay=0.1,
        process_weight_state="logw",
        key=jax.random.key(42),
    )
    assert nll.shape == (M_DEFAULT + 1,)
    assert theta_hist.shape[0] == M_DEFAULT + 1
    assert jnp.all(jnp.isfinite(nll))


def test_dpop_train_scalar_eta(simple_sir_for_dpop):
    """
    Test dpop_train with a scalar eta (uniform LR for all params).
    """
    model = simple_sir_for_dpop
    nll, theta_hist = model.dpop_train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=0.01,
        optimizer="SGD",
        alpha=0.8,
        decay=0.1,
        process_weight_state="logw",
        key=jax.random.key(42),
    )
    assert nll.shape == (M_DEFAULT + 1,)
    assert theta_hist.shape[0] == M_DEFAULT + 1
    assert jnp.all(jnp.isfinite(nll))


def test_dpop_train_param_order_invariance(simple_sir_for_dpop):
    """
    Check that dpop_train is invariant to the ordering of
    parameter dictionary keys (in natural space).
    """
    model = simple_sir_for_dpop

    J = J_DEFAULT
    M = M_DEFAULT
    eta = {name: 0.01 for name in model.canonical_param_names}

    # First run: default theta ordering
    key1 = jax.random.key(123)
    nll1, theta_hist1 = model.dpop_train(
        J=J,
        M=M,
        eta=eta,
        optimizer="SGD",
        decay=0.1,
        alpha=0.8,
        key=key1,
        process_weight_state="logw",
    )

    # Build a permuted theta with reversed key order
    theta_orig = model.theta  # list[dict]
    param_keys = list(theta_orig[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta_orig]

    # Second run: same random key & hyper-parameters, but permuted theta
    key2 = jax.random.key(123)
    nll2, theta_hist2 = model.dpop_train(
        J=J,
        M=M,
        eta=eta,
        optimizer="SGD",
        decay=0.1,
        alpha=0.8,
        key=key2,
        theta=permuted_theta,
        process_weight_state="logw",
    )

    # Histories should match exactly up to numerical precision
    np.testing.assert_allclose(nll1, nll2, atol=1e-7)
    np.testing.assert_allclose(theta_hist1, theta_hist2, atol=1e-7)

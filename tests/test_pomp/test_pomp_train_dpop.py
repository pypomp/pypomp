import jax
import pytest
import numpy as np
import pypomp as pp
import jax.numpy as jnp

J_DEFAULT = 2
M_DEFAULT = 2


@pytest.fixture(scope="module")
def simple_sir_for_dpop():
    """
    Build a small SIR Pomp model for testing the DPOP optimizers.
    """
    # Mirror the shrinkage in test_pomp_dpop.py::simple_sir to keep setup fast.
    model = pp.models.sir(delta_t=0.1, times=np.array([0.2, 0.4]))
    return model


@pytest.mark.parametrize(
    "optimizer, decay, eta_type",
    [
        ("Adam", 0.0, "dict"),
        ("SGD", 0.1, "dict"),
        ("Adam", 0.1, "dict"),
        ("SGD", 0.1, "scalar"),
    ],
)
def test_dpop_train_variants(simple_sir_for_dpop, optimizer, decay, eta_type):
    """
    Test dpop_train with various optimizer configurations.
    """
    model = simple_sir_for_dpop
    if eta_type == "dict":
        eta = {name: 0.01 for name in model.canonical_param_names}
    else:
        eta = 0.01

    nll, theta_hist = model.dpop_train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer=optimizer,
        alpha=0.8,
        decay=decay,
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

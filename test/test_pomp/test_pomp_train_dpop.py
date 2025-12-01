import jax
import pytest
import numpy as np
import pypomp as pp
import jax.numpy as jnp
from pypomp.train_dpop import dpop_sgd_decay


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

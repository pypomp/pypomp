# test_sis_dpop.py

import jax
import jax.numpy as jnp
import pytest
import pypomp as pp


@pytest.fixture(scope="function")
def simple_sis():
    """
    Build a simple SIS Pomp model for testing DPOP.

    Returns
    -------
    model : pp.SIS
        SIS Pomp object with internally simulated data.
    J : int
        Number of particles (small here for speed).
    ys : pandas.DataFrame
        Observations used by the model.
    theta : list[dict]
        Model parameters (natural space) as stored in model.theta.
    covars : pandas.DataFrame | None
        Covariates used by the model (None for this SIS).
    key : jax.random.PRNGKey
        Base random key for tests.
    """
    # Use a fixed key so tests are reproducible
    key_model = jax.random.key(2025)
    # Construct the SIS Pomp model with internally simulated data
    model = pp.SIS(T=50, key=key_model)

    # Keep J small so tests run fast
    J = 2
    ys = model.ys
    theta = model.theta
    covars = model.covars
    key = jax.random.key(111)

    return model, J, ys, theta, covars, key


def test_dpop_basic(simple_sis):
    """
    Basic sanity check for DPOP on the SIS model.

    - dpop() should return a finite scalar negative log-likelihood
      (per replicate) of floating dtype.
    """
    model, J, ys, theta, covars, key = simple_sis

    # Call DPOP with a moderate cooling factor alpha
    vals = model.dpop(J=J, alpha=0.9, key=key)
    nll0 = vals[0]

    # Must be a scalar
    assert nll0.shape == ()
    # Must be finite
    assert jnp.isfinite(nll0.item())
    # Must be floating type (we don't hard-code float32 in case backend changes)
    assert jnp.issubdtype(nll0.dtype, jnp.floating)


def test_dpop_param_order_invariance(simple_sis):
    """
    Check that DPOP result does not depend on the order of parameter
    dictionary keys.

    We reorder theta[0].keys() and pass permuted theta into model.dpop;
    the DPOP negative log-likelihood should remain the same (up to
    numerical tolerance).
    """
    model, J, ys, theta, covars, key = simple_sis

    # Baseline DPOP value with original theta ordering
    val1 = model.dpop(J=J, alpha=0.9, key=key, theta=theta)
    nll1 = val1[0]

    # Reverse the key order in each theta dict
    param_keys = list(theta[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta]

    # Use a fresh key to avoid accidental correlation in randomness
    key2 = jax.random.key(111)
    val2 = model.dpop(J=J, alpha=0.9, key=key2, theta=permuted_theta)
    nll2 = val2[0]

    # The two NLLs should match up to numerical tolerance
    assert jnp.allclose(nll1, nll2, atol=1e-7), (
        f"DPOP result changed after theta reordering: {nll1} vs {nll2}"
    )


def test_dpop_default_vs_explicit_process_weight_index(simple_sis):
    """
    Using the default process_weight_index (via accumvars) should produce
    the same DPOP value as passing the 'logw' state index explicitly.
    """
    model, J, ys, theta, covars, _ = simple_sis

    # Use the same base seed for both calls so the random stream matches.
    key1 = jax.random.key(999)
    key2 = jax.random.key(999)

    # 1) Default behavior: process_weight_index inferred from accumvars
    nll_default = model.dpop(J=J, alpha=0.9, key=key1)[0]

    # 2) Explicit process_weight_index = index of "logw" in statenames
    logw_index = model.statenames.index("logw")
    nll_explicit = model.dpop(
        J=J,
        alpha=0.9,
        key=key2,
        process_weight_index=logw_index,
    )[0]

    assert jnp.allclose(nll_default, nll_explicit, atol=1e-7), (
        f"DPOP default vs explicit index mismatch: {nll_default} vs {nll_explicit}"
    )

import jax
import jax.numpy as jnp
import pytest
import pypomp as pp


@pytest.fixture(scope="function")
def simple_sir():
    """
    Build a simple SIR Pomp model for testing DPOP.

    Returns
    -------
    model : Pomp
        SIR Pomp object with internally simulated data.
    J : int
        Number of particles (small here for speed).
    ys : pandas.DataFrame
        Observations used by the model.
    theta : list[dict]
        Model parameters (natural space) as stored in model.theta.
    covars : pandas.DataFrame | None
        Covariates used by the model.
    key : jax.random.key
        Base random key for tests.
    """
    model = pp.sir()

    # Keep J small so tests run fast
    J = 2
    ys = model.ys
    theta = model.theta
    covars = model.covars
    key = jax.random.key(111)

    return model, J, ys, theta, covars, key


def test_dpop_basic(simple_sir):
    """
    Basic sanity check for DPOP on the SIR model.

    - dpop() should return a finite scalar negative log-likelihood
      (per replicate) of floating dtype.
    """
    model, J, ys, theta, covars, key = simple_sir

    # Call DPOP with a moderate cooling factor alpha
    vals = model.dpop(
        J=J,
        alpha=0.9,
        key=key,
        process_weight_state="logw",
    )
    nll0 = vals[0]

    # Must be a scalar
    assert nll0.shape == ()
    # Must be finite
    assert jnp.isfinite(nll0.item())
    # Must be floating type (we don't hard-code float32 in case backend changes)
    assert jnp.issubdtype(nll0.dtype, jnp.floating)


def test_dpop_param_order_invariance(simple_sir):
    """
    Check that DPOP result does not depend on the order of parameter
    dictionary keys.

    We reorder theta[0].keys() and pass permuted theta into model.dpop;
    the DPOP negative log-likelihood should remain the same (up to
    numerical tolerance).
    """
    model, J, ys, theta, covars, key = simple_sir

    # Baseline DPOP value with original theta ordering
    val1 = model.dpop(
        J=J,
        alpha=0.9,
        key=key,
        theta=theta,
        process_weight_state="logw",
    )
    nll1 = val1[0]

    # Reverse the key order in each theta dict
    param_keys = list(theta[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta]

    # Use a fresh key to avoid accidental correlation in randomness
    key2 = jax.random.key(111)
    val2 = model.dpop(
        J=J,
        alpha=0.9,
        key=key2,
        theta=permuted_theta,
        process_weight_state="logw",
    )
    nll2 = val2[0]

    # The two NLLs should match up to numerical tolerance
    assert jnp.allclose(nll1, nll2, atol=1e-7), (
        f"DPOP result changed after theta reordering: {nll1} vs {nll2}"
    )


def test_dpop_explicit_process_weight_state_is_deterministic(simple_sir):
    """
    Calling dpop() with the same process_weight_state ('logw') and the
    same random seed should give identical results.
    """
    model, J, ys, theta, covars, _ = simple_sir

    # Use the same base seed for both calls so the random stream matches.
    key1 = jax.random.key(999)
    key2 = jax.random.key(999)

    nll_1 = model.dpop(
        J=J,
        alpha=0.9,
        key=key1,
        process_weight_state="logw",
    )[0]

    nll_2 = model.dpop(
        J=J,
        alpha=0.9,
        key=key2,
        process_weight_state="logw",
    )[0]

    assert jnp.allclose(nll_1, nll_2, atol=1e-7), (
        f"DPOP results with process_weight_state='logw' do not match: "
        f"{nll_1} vs {nll_2}"
    )

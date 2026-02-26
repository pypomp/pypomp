# test_ctmc_multinom.py
"""
Unit tests for the Euler-multinomial CTMC utilities.

We test:
- basic shape and mass-conservation properties of `reulermultinom`,
- consistency between `sample_and_log_prob` and `deulermultinom`,
- behaviour of `deulermultinom` in the "no event" case.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pypomp.ctmc_multinom import (
    reulermultinom,
    deulermultinom,
    sample_and_log_prob,
)


def test_reulermultinom_shape_and_total():
    """
    Basic sanity check for `reulermultinom`:

    - output shape should be (K+1,) when shape=(),
    - output shape should be (B, K+1) when shape=(B,),
    - counts should be non-negative and sum to n in every draw.
    """
    key = jax.random.key(0)
    n = 50
    rates = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)  # K = 3
    dt = 0.5

    # Single draw
    x_single = reulermultinom(key, n=n, rates=rates, dt=dt)
    assert x_single.shape == (rates.shape[0] + 1,)
    assert jnp.all(x_single >= 0)
    assert jnp.isclose(jnp.sum(x_single), n)

    # Batch of draws
    key_batch = jax.random.key(1)
    B = 10
    x_batch = reulermultinom(key_batch, n=n, rates=rates, dt=dt, shape=(B,))
    assert x_batch.shape == (B, rates.shape[0] + 1)
    assert jnp.all(x_batch >= 0)

    totals = jnp.sum(x_batch, axis=-1)
    assert jnp.all(jnp.isclose(totals, n))


def test_sample_and_log_prob_matches_deulermultinom():
    """
    Check that `sample_and_log_prob` and `deulermultinom` are consistent.

    For a given (N, rates, dt) and random key:
    - `sample_and_log_prob` returns a sample and its log-probability,
    - `deulermultinom(sample, N, rates, dt)` should match that log-prob.
    """
    key = jax.random.key(2)
    N = 20
    rates = jnp.array([0.4, 0.6], dtype=jnp.float32)
    dt = 0.3

    sample, logw, key_out = sample_and_log_prob(N=N, rates=rates, dt=dt, key=key)

    # Shape checks
    assert sample.shape == rates.shape
    assert isinstance(logw, jnp.ndarray)
    assert logw.shape == ()  # scalar

    # Key should be updated
    assert not jnp.array_equal(key, key_out)

    # Consistency check: recompute log-prob using deulermultinom
    logw2 = deulermultinom(x=sample, n=N, rates=rates, dt=dt)
    np.testing.assert_allclose(
        np.array(logw),
        np.array(logw2),
        atol=1e-6,
        err_msg="sample_and_log_prob logw does not match deulermultinom",
    )


def test_deulermultinom_handles_zero_events():
    """
    Check that `deulermultinom` behaves sensibly in the 'no event' case.

    If x is all zeros, the probability mass should correspond to having
    no events in that Euler step and the log-probability should be finite.
    """
    N = 10
    rates = jnp.array([0.2, 0.3], dtype=jnp.float32)
    dt = 0.1
    x_zero = jnp.zeros_like(rates)

    logw_zero = deulermultinom(x=x_zero, n=N, rates=rates, dt=dt)

    # log-prob should be finite (no NaNs / infs)
    assert jnp.isfinite(logw_zero), "log-prob for zero increment should be finite"

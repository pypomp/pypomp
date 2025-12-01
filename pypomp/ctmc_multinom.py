"""
ctmc_multinom.py

Utility functions for Euler-multinomial approximations of CTMC transitions.

These helpers are model-agnostic and can be reused across different
process models that use a multinomial Euler scheme.

Multinomial samples are drawn using a fast inverse-CDF based sampler
(`pypomp.random.binominvf.rmultinomial`), and log-probabilities are
computed via an explicit multinomial log-pmf formula.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from pypomp.random.binominvf import rmultinomial as fast_rmultinomial


def _euler_multinomial_probs(
    rates: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Compute multinomial probabilities for an Euler CTMC step.

    Given transition rates (length K) and time step dt, we construct a
    (K+1)-dimensional probability vector:

        p = [p0, p1, ..., pK]

    where:
        p0  = P("no event" over dt)   = exp(-sum(rates) * dt)
        pi  ∝ rate_i, i = 1..K, scaled such that sum(p) = 1.
    """
    rates = jnp.asarray(rates)
    sumrates = jnp.sum(rates)

    # Probability of "no event"
    logp0 = -sumrates * dt
    p0 = jnp.exp(logp0)

    # When sumrates == 0, this reduces to p0 = 1, others = 0
    scale = jnp.where(
        sumrates > 0.0,
        (1.0 - p0) / jnp.maximum(sumrates, 1.0e-15),
        0.0,
    )

    probs_others = rates * scale
    probs = jnp.concatenate(
        [jnp.array([p0], dtype=rates.dtype), probs_others],
        axis=0,
    )

    # Normalize defensively to ensure numerical sum to 1.
    probs = probs / jnp.maximum(jnp.sum(probs), 1.0e-15)
    return probs


def _multinomial_logpmf(
    counts: jnp.ndarray,
    n: jnp.ndarray,
    probs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Multinomial log-pmf:

        log P(X = x | n, p) =
            log(n!) - sum_i log(x_i!) + sum_i x_i log(p_i)

    where sum_i x_i = n and sum_i p_i = 1.
    """
    counts = jnp.asarray(counts, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    probs = jnp.asarray(probs, dtype=jnp.float64)

    # Clip probabilities to avoid log(0).
    probs_safe = jnp.clip(probs, 1.0e-12, 1.0)

    return (
        gammaln(n + 1.0)
        - jnp.sum(gammaln(counts + 1.0))
        + jnp.sum(counts * jnp.log(probs_safe))
    )


def reulermultinom(
    key: jax.Array,
    n: jnp.ndarray,
    rates: jnp.ndarray,
    dt: float,
    shape=(),
) -> jnp.ndarray:
    """
    Draw multinomial Euler increments for a single compartment.

    Args:
        key: JAX PRNGKey.
        n: Total number of individuals in the compartment (scalar or array).
        rates: 1D array of transition rates (length K).
        dt: Euler time step.
        shape: Optional leading sample shape. If non-empty, a batch of
            samples is drawn with that leading shape.

    Returns:
        Multinomial sample array of shape ``shape + (K + 1,)``, where the
        first entry corresponds to 'no event' and the remaining entries
        correspond to the K event types.
    """
    probs = _euler_multinomial_probs(rates, dt)  # shape (K+1,)

    # Scalar sample (shape == ())
    if shape == () or shape == ():
        return fast_rmultinomial(key, n, probs)

    # Batched samples for a given leading shape
    shape = tuple(shape)
    num_samples = int(jnp.prod(jnp.array(shape, dtype=jnp.int32)))

    # Create keys for each sample
    keys = jax.random.split(key, num_samples)

    # Broadcast n and probs to match the batch
    n_b = jnp.broadcast_to(n, (num_samples,))
    probs_b = jnp.broadcast_to(probs, (num_samples, probs.shape[0]))

    # Vectorized multinomial sampling
    samples_flat = jax.vmap(fast_rmultinomial, in_axes=(0, 0, 0))(
        keys, n_b, probs_b
    )

    out_shape = shape + (probs.shape[0],)
    return samples_flat.reshape(out_shape)


def deulermultinom(
    x: jnp.ndarray,
    n: jnp.ndarray,
    rates: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Evaluate the log-probability of a given multinomial Euler increment.

    Args:
        x: 1D array of event counts for the K event types (excluding 'no event'),
           e.g. transitions to each target state.
        n: Total number of individuals in the compartment.
        rates: 1D array of transition rates (length K).
        dt: Euler time step.

    Returns:
        Scalar log-probability of observing the increment ``x`` under
        the Euler-multinomial scheme.
    """
    x = jnp.asarray(x)
    n = jnp.asarray(n)

    probs = _euler_multinomial_probs(rates, dt)  # shape (K+1,)

    # Reconstruct the "no event" count and full count vector:
    #   x_full = [x0, x1, ..., xK], where x0 = n - sum(x_i).
    x_sum = jnp.sum(x)
    x0 = n - x_sum
    x_full = jnp.concatenate(
        [jnp.array([x0], dtype=x.dtype), x],
        axis=0,
    )

    return _multinomial_logpmf(x_full, n, probs)


def sample_and_log_prob(
    N: jnp.ndarray,
    rates: jnp.ndarray,
    dt: float,
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """
    Draw Euler-multinomial increments and return both the sample and its log-prob.

    Args:
        N: Total number of individuals in the compartment.
        rates: 1D array of transition rates (length K).
        dt: Euler time step.
        key: JAX PRNGKey.

    Returns:
        sample: 1D array of length K with the event counts (excluding 'no event').
        logw: Scalar log-probability of the sampled increment.
        key: Updated PRNGKey.
    """
    probs = _euler_multinomial_probs(rates, dt)  # shape (K+1,)

    N = jnp.asarray(N, dtype=probs.dtype)

    # Draw one multinomial sample over (K+1) categories:
    # [no event, event_1, ..., event_K].
    sample_full = fast_rmultinomial(key, N, probs)  # shape (K+1,)

    # Drop the "no event" component; keep only the K event types.
    sample = sample_full[1:]

    # Compute log-probability of the full sample (including no-event).
    logw = _multinomial_logpmf(sample_full, N, probs)

    # Update the PRNG key
    key, _ = jax.random.split(key)

    return sample, logw, key

"""
ctmc_multinom.py
Utility functions for Euler-multinomial approximations of CTMC transitions.

These helpers are model-agnostic and can be reused across different
process models that use a multinomial Euler scheme.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


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
        n: Total number of individuals in the compartment (scalar).
        rates: 1D array of transition rates (length K).
        dt: Euler time step.
        shape: Optional leading sample shape.

    Returns:
        Multinomial sample array of shape ``shape + (K + 1,)``, where the
        first entry corresponds to 'no event' and the remaining entries
        correspond to the K event types.
    """
    sumrates = jnp.sum(rates)
    logp0 = -sumrates * dt  # log probability of no event
    logits_others = jnp.log(-jnp.expm1(logp0)) + jnp.log(rates) - jnp.log(sumrates)
    logits = jnp.insert(logits_others, 0, logp0)

    n = jnp.asarray(n, dtype=logits.dtype)
    dist = tfd.Multinomial(total_count=n, logits=logits)
    return dist.sample(seed=key, sample_shape=shape)


def deulermultinom(
    x: jnp.ndarray,
    n: jnp.ndarray,
    rates: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Evaluate the log-probability of a given multinomial Euler increment.

    Args:
        x: 1D array of event counts for the K event types (excluding 'no event').
        n: Total number of individuals in the compartment.
        rates: 1D array of transition rates (length K).
        dt: Euler time step.

    Returns:
        Scalar log-probability of observing the increment ``x`` under
        the Euler-multinomial scheme.
    """
    sumrates = jnp.sum(rates)
    logp0 = -sumrates * dt
    logits_others = jnp.log(-jnp.expm1(logp0)) + jnp.log(rates) - jnp.log(sumrates)
    logits = jnp.insert(logits_others, 0, logp0)

    n = jnp.asarray(n, dtype=logits.dtype)
    # prepend the 'no event' count
    x_full = jnp.insert(x, 0, n - jnp.sum(x))
    dist = tfd.Multinomial(total_count=n, logits=logits)
    return dist.log_prob(x_full)


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
    sumrates = jnp.sum(rates)
    logp0 = -sumrates * dt
    logits_others = jnp.log(-jnp.expm1(logp0)) + jnp.log(rates) - jnp.log(sumrates)
    logits = jnp.insert(logits_others, 0, logp0)

    N = jnp.asarray(N, dtype=logits.dtype)
    dist = tfd.Multinomial(total_count=N, logits=logits)

    sample_full = dist.sample(seed=key)          # shape (K+1,)
    sample = sample_full[1:]                     # drop 'no event' component
    logw = dist.log_prob(sample_full)           # log-prob of this increment

    key, _ = jax.random.split(key)
    return sample, logw, key

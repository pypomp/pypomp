"""
JIT-compiled PMCMC (particle Markov chain Monte Carlo).

Implements the particle marginal Metropolis-Hastings sampler of
Andrieu, Doucet & Holenstein (2010).  The outer MCMC loop runs inside
``jax.lax.scan`` for a fixed number of iterations ``Nmcmc``, and a
particle filter (``_pfilter_internal``) provides a noisy but unbiased
estimate of the marginal log-likelihood at each proposed parameter.

The acceptance rule, prior evaluation and proposal step are all
expressed in pure JAX so the entire chain compiles into a single
XLA program.
"""

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit

from pypomp.proposals import Proposal
from .pfilter import _pfilter_internal
from .types import PmcmcConfig, PmcmcInputs


@partial(
    jit,
    static_argnames=("config",),
)
def _pmcmc_internal(
    theta_arr: jax.Array,
    proposal: Proposal,
    config: PmcmcConfig,
    inputs: PmcmcInputs,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run a single PMCMC chain of length ``Nmcmc`` starting at ``theta_arr``.

    Args:
        theta_arr: Starting parameter vector, shape ``(d,)``.
        proposal: Proposal object exposing ``init_state`` and ``step``.
            See :mod:`pypomp.proposals`.
        config: PMCMC configuration.
        inputs: PMCMC inputs.
        key: PRNG key.

    Returns:
        ``(loglik_trace, log_prior_trace, theta_trace, accepts)`` where:

        * ``loglik_trace`` has shape ``(Nmcmc + 1,)`` (iteration 0 is the
          initial evaluation).
        * ``log_prior_trace`` has shape ``(Nmcmc + 1,)``.
        * ``theta_trace`` has shape ``(Nmcmc + 1, d)``.
        * ``accepts`` is a scalar count of accepted proposals.
    """
    # 1. Prepare particle-filter evaluation function.
    run_pfilter_fn = jax.tree_util.Partial(
        _pmcmc_run_pfilter,
        config=config,
        inputs=inputs,
    )

    # 2. Initial evaluation at starting theta.
    key, init_pf_key = jax.random.split(key)
    loglik0 = run_pfilter_fn(theta_arr, init_pf_key)
    log_prior0 = config.dprior(theta_arr)
    prop_state0 = proposal.init_state(theta_arr)

    init_carry = (
        theta_arr,  # current accepted theta
        loglik0,  # current accepted loglik
        log_prior0,  # current accepted log prior
        prop_state0,  # adaptive proposal state
        jnp.array(0, dtype=jnp.int32),  # accepts counter
        key,  # PRNG key
    )

    # 3. Setup scan step function.
    step_fn = jax.tree_util.Partial(
        _pmcmc_step,
        proposal,
        config,
        run_pfilter_fn,
    )

    # 4. Run the scan loop.
    final_carry, (ll_trace, lp_trace, theta_trace) = jax.lax.scan(
        step_fn, init_carry, jnp.arange(1, config.Nmcmc + 1, dtype=jnp.int32)
    )
    final_accepts = final_carry[4]

    # Prepend iteration-0 (the initial evaluation) to each trace.
    ll_trace = jnp.concatenate((jnp.asarray([loglik0]), ll_trace))
    lp_trace = jnp.concatenate((jnp.asarray([log_prior0]), lp_trace))
    theta_trace = jnp.concatenate((theta_arr[None, :], theta_trace), axis=0)

    return ll_trace, lp_trace, theta_trace, final_accepts


def _pmcmc_step(
    proposal: Proposal,
    config: PmcmcConfig,
    run_pfilter_fn: Callable,
    carry: tuple[jax.Array, jax.Array, jax.Array, Any, jax.Array, jax.Array],
    n: int | jax.Array,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, Any, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]:
    """Run one step of the PMCMC chain."""
    theta_cur, ll_cur, lp_cur, prop_state, accepts, key = carry
    key, prop_key, pf_key, accept_key = jax.random.split(key, 4)

    # 1. Propose (and update proposal state)
    theta_prop, new_prop_state = proposal.step(
        prop_state, theta_cur, prop_key, n, accepts
    )

    # 2. Evaluate prior and likelihood at proposed parameter.
    lp_prop = config.dprior(theta_prop)
    ll_prop = run_pfilter_fn(theta_prop, pf_key)

    # 3. Accept or reject.
    log_alpha = ll_prop + lp_prop - ll_cur - lp_cur
    u = jax.random.uniform(accept_key)
    accept = jnp.isfinite(log_alpha) & jnp.isfinite(lp_prop) & (jnp.log(u) < log_alpha)

    # 4. Update.
    new_theta = jax.lax.select(accept, theta_prop, theta_cur)
    new_ll = jax.lax.select(accept, ll_prop, ll_cur)
    new_lp = jax.lax.select(accept, lp_prop, lp_cur)
    new_accepts = accepts + accept.astype(jnp.int32)

    out = (new_ll, new_lp, new_theta)
    new_carry = (new_theta, new_ll, new_lp, new_prop_state, new_accepts, key)
    return new_carry, out


def _pmcmc_run_pfilter(
    theta: jax.Array,
    k: jax.Array,
    config: PmcmcConfig,
    inputs: PmcmcInputs,
) -> jax.Array:
    """Run the particle filter to get the log-likelihood."""
    out = _pfilter_internal(
        theta,
        k,
        config.to_pfilter_config(),
        inputs.to_pfilter_inputs(),
    )
    return -out["neg_loglik"]


# vmap over the chain (replicate) dimension: theta_arr (0), key (0), everything else None.
# proposal flows through unchanged (its leaves are not chain-dependent).
_vmapped_pmcmc_internal = jax.vmap(
    _pmcmc_internal,
    in_axes=(
        0,  # theta_arr per chain
        None,  # proposal
        None,  # config
        None,  # inputs
        0,  # key per chain
    ),
)

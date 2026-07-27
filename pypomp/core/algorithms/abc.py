"""
JIT-compiled ABC-MCMC.

Implements Approximate Bayesian Computation with a Metropolis-Hastings
outer loop.  At each iteration the algorithm:

1. Draws a proposal ``theta_prop`` from the user-supplied proposal.
2. Evaluates the log-prior at ``theta_prop`` and accepts the MH prior
   ratio (symmetric proposal: ``log(u) < log_prior_prop - log_prior``).
3. Simulates a single synthetic dataset under ``theta_prop`` and
   computes ``distance_prop``, the squared scaled Euclidean distance
   between observed and simulated probe vectors.
4. Accepts iff both the prior ratio and ``distance_prop < epsilon**2``
   pass.

The outer loop is a single ``jax.lax.scan``; ``jax.vmap`` over the
chain dimension runs ``n_chains`` chains in parallel.

The user provides a probe function ``probe_fn(y) -> (n_probes,)``
operating on a dict mapping observation names to ``(n_obs,)`` JAX
arrays, plus a matching ``(n_probes,)`` ``scale_arr`` for normalising
distances.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import jit

from pypomp.proposals import Proposal

from .contexts import AbcContext
from .simulate import _simulate_internal

SHOULD_TRANS = True  # Should transformations be applied to the parameters?


@jit
def _abc_internal(
    theta_arr: jax.Array,
    proposal: Proposal,
    context: AbcContext,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run one ABC-MCMC chain of length ``M`` starting at ``theta_arr``.

    Returns ``(distance_trace, log_prior_trace, theta_trace, accepts)``:

    * ``distance_trace``: shape ``(M + 1,)``.
    * ``log_prior_trace``: shape ``(M + 1,)``.
    * ``theta_trace``: shape ``(M + 1, d)``.
    * ``accepts``: scalar count of accepted proposals.
    """
    # 1. Prepare simulation distance function.
    sim_distance_fn = jax.tree_util.Partial(
        _abc_sim_distance,
        context=context,
    )

    # 2. Initial evaluation.
    key, init_sim_key = jax.random.split(key)
    dist0 = sim_distance_fn(theta_arr, init_sim_key)
    lp0 = context.dprior(theta_arr, SHOULD_TRANS)

    prop_state0 = proposal.init_state(theta_arr)

    init_carry = (
        theta_arr,
        dist0,
        lp0,
        prop_state0,
        jnp.array(0, dtype=jnp.int32),
        key,
    )

    # 3. Setup scan step function.
    step_fn = jax.tree_util.Partial(
        _abc_step,
        proposal,
        context,
        sim_distance_fn,
    )

    # 4. Run the scan loop.
    final_carry, (dist_trace, lp_trace, theta_trace) = jax.lax.scan(
        step_fn, init_carry, jnp.arange(1, context.M + 1, dtype=jnp.int32)
    )

    final_accepts = final_carry[4]

    # 5. Collect traces and prepend initial evaluation.
    dist_trace = jnp.concatenate((jnp.asarray([dist0]), dist_trace))
    lp_trace = jnp.concatenate((jnp.asarray([lp0]), lp_trace))
    theta_trace = jnp.concatenate((theta_arr[None, :], theta_trace), axis=0)

    return dist_trace, lp_trace, theta_trace, final_accepts


def _abc_step(
    proposal: Proposal,
    context: AbcContext,
    sim_distance_fn: Callable,
    carry: tuple[jax.Array, jax.Array, jax.Array, Any, jax.Array, jax.Array],
    n: int | jax.Array,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, Any, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]:
    """Run one step of the ABC-MCMC chain."""
    # 1. Unpack carry state.
    theta_cur, dist_cur, lp_cur, prop_state, accepts, key = carry
    key, prop_key, sim_key, accept_key = jax.random.split(key, 4)

    # 2. Draw a proposal theta_prop from the proposal distribution.
    theta_prop, new_prop_state = proposal.step(
        prop_state, theta_cur, prop_key, n, accepts
    )
    lp_prop = context.dprior(theta_prop, SHOULD_TRANS)

    # 3. Simulate and compute distance for the proposed theta.
    dist_prop = sim_distance_fn(theta_prop, sim_key)

    # 4. Perform Metropolis-Hastings acceptance check.
    # MH on the prior (symmetric proposal -> just the prior ratio).
    log_alpha_prior = lp_prop - lp_cur
    u = jax.random.uniform(accept_key)
    prior_pass = jnp.isfinite(lp_prop) & (jnp.log(u) < log_alpha_prior)
    eps2 = jnp.asarray(context.epsilon, dtype=theta_cur.dtype) ** 2
    dist_pass = dist_prop < eps2
    accept = prior_pass & dist_pass

    # 5. Update state variables based on acceptance.
    new_theta = jax.lax.select(accept, theta_prop, theta_cur)
    new_dist = jax.lax.select(accept, dist_prop, dist_cur)
    new_lp = jax.lax.select(accept, lp_prop, lp_cur)
    new_accepts = jnp.add(accepts, accept.astype(jnp.int32))

    # 6. Return new carry and outputs.
    new_carry = (new_theta, new_dist, new_lp, new_prop_state, new_accepts, key)
    return new_carry, (new_dist, new_lp, new_theta)


def _abc_sim_distance(
    theta: jax.Array,
    sim_key: jax.Array,
    context: AbcContext,
) -> jax.Array:
    """Simulate a dataset under ``theta`` and compute the probe distance.

    Returns the squared scaled Euclidean distance.
    """
    # 1. Simulate a single synthetic dataset under theta.
    _, Y = _simulate_internal(
        context.fns.rinitializer,
        context.fns.rprocess_interp,
        context.rmeasure,
        theta,
        context.series.t0,
        context.series.times,
        context.series.dt_array_extended,
        context.series.nstep_array,
        context.ydim,
        context.series.covars_extended,
        context.fns.accumvars,
        1,  # nsim
        sim_key,
        SHOULD_TRANS,
    )

    # Y shape: (n_obs, ydim, 1) -> (n_obs, ydim)
    y_arr = Y[..., 0]

    # 2. Compute probe values.
    sim_p = context.probe_fn(y_arr)

    # 3. Return the squared scaled Euclidean distance.
    return jnp.sum(((context.obs_probes - sim_p) / context.scale_arr) ** 2)


_vmapped_abc_internal = jax.vmap(
    _abc_internal,
    in_axes=(
        0,  # theta_arr per chain
        None,  # proposal
        None,  # context
        0,  # key per chain
    ),
)

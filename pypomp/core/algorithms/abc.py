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

The user provides a probe function ``probe_fn(y_arr) -> (n_probes,)``
operating on a ``(n_obs, ydim)`` JAX array, plus a matching
``(n_probes,)`` ``scale_arr`` for normalising distances.
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit

from .simulate import _simulate_internal


@partial(
    jit,
    static_argnames=(
        "Nabc",
        "rinitializer",
        "rprocess_interp",
        "rmeasure",
        "accumvars",
        "dprior",
        "probe_fn",
        "ydim",
    ),
)
def _abc_internal(
    theta_arr: jax.Array,
    proposal,
    dprior: Callable,
    probe_fn: Callable,
    obs_probes: jax.Array,
    scale_arr: jax.Array,
    epsilon: float,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    rinitializer: Callable,
    rprocess_interp: Callable,
    rmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    ydim: int,
    Nabc: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run one ABC-MCMC chain of length ``Nabc`` starting at ``theta_arr``.

    Returns ``(distance_trace, log_prior_trace, theta_trace, accepts)``:

    * ``distance_trace``: shape ``(Nabc + 1,)``.
    * ``log_prior_trace``: shape ``(Nabc + 1,)``.
    * ``theta_trace``: shape ``(Nabc + 1, d)``.
    * ``accepts``: scalar count of accepted proposals.
    """
    eps2 = jnp.asarray(epsilon, dtype=theta_arr.dtype) ** 2

    def _sim_distance(theta: jax.Array, sim_key: jax.Array) -> jax.Array:
        _, Y = _simulate_internal(
            rinitializer,
            rprocess_interp,
            rmeasure,
            theta,
            t0,
            times,
            dt_array_extended,
            nstep_array,
            ydim,
            covars_extended,
            accumvars,
            1,  # nsim
            sim_key,
        )
        # Y shape: (n_obs, ydim, 1) -> (n_obs, ydim)
        y_arr = Y[..., 0]
        sim_p = probe_fn(y_arr)
        return jnp.sum(((obs_probes - sim_p) / scale_arr) ** 2)

    # ---- Initial evaluation ----
    key, init_sim_key = jax.random.split(key)
    dist0 = _sim_distance(theta_arr, init_sim_key)
    lp0 = dprior(theta_arr)
    prop_state0 = proposal.init_state(theta_arr)

    init_carry = (
        theta_arr,
        dist0,
        lp0,
        prop_state0,
        jnp.array(0, dtype=jnp.int32),
        key,
    )

    def step(carry, n):
        theta_cur, dist_cur, lp_cur, prop_state, accepts, key = carry
        key, prop_key, sim_key, accept_key = jax.random.split(key, 4)

        theta_prop, new_prop_state = proposal.step(
            prop_state, theta_cur, prop_key, n, accepts
        )
        lp_prop = dprior(theta_prop)
        dist_prop = _sim_distance(theta_prop, sim_key)

        # MH on the prior (symmetric proposal -> just the prior ratio).
        log_alpha_prior = lp_prop - lp_cur
        u = jax.random.uniform(accept_key)
        prior_pass = jnp.isfinite(lp_prop) & (jnp.log(u) < log_alpha_prior)
        dist_pass = dist_prop < eps2
        accept = prior_pass & dist_pass

        new_theta = jnp.where(accept, theta_prop, theta_cur)
        new_dist = jnp.where(accept, dist_prop, dist_cur)
        new_lp = jnp.where(accept, lp_prop, lp_cur)
        new_accepts = accepts + accept.astype(jnp.int32)

        new_carry = (new_theta, new_dist, new_lp, new_prop_state, new_accepts, key)
        return new_carry, (new_dist, new_lp, new_theta)

    final_carry, (dist_trace, lp_trace, theta_trace) = jax.lax.scan(
        step, init_carry, jnp.arange(1, Nabc + 1, dtype=jnp.int32)
    )

    final_accepts = final_carry[4]

    dist_trace = jnp.concatenate((jnp.asarray([dist0]), dist_trace))
    lp_trace = jnp.concatenate((jnp.asarray([lp0]), lp_trace))
    theta_trace = jnp.concatenate((theta_arr[None, :], theta_trace), axis=0)

    return dist_trace, lp_trace, theta_trace, final_accepts


_vmapped_abc_internal = jax.vmap(
    _abc_internal,
    in_axes=(
        0,  # theta_arr per chain
        None,  # proposal
        None,  # dprior
        None,  # probe_fn
        None,  # obs_probes
        None,  # scale_arr
        None,  # epsilon
        None,  # dt_array_extended
        None,  # nstep_array
        None,  # t0
        None,  # times
        None,  # rinitializer
        None,  # rprocess_interp
        None,  # rmeasure
        None,  # accumvars
        None,  # covars_extended
        None,  # ydim
        None,  # Nabc
        0,  # key per chain
    ),
)

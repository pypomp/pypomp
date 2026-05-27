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
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit

from .pfilter import _pfilter_internal


@partial(
    jit,
    static_argnames=(
        "Nmcmc",
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "accumvars",
        "dprior",
        "should_trans",
    ),
)
def _pmcmc_internal(
    theta_arr: jax.Array,
    proposal,
    dprior: Callable,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,
    rinitializer: Callable,
    rprocess_interp: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    Nmcmc: int,
    key: jax.Array,
    should_trans: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run a single PMCMC chain of length ``Nmcmc`` starting at ``theta_arr``.

    Args:
        theta_arr: Starting parameter vector, shape ``(d,)``.
        proposal: Proposal object exposing ``init_state`` and ``step``.
            See :mod:`pypomp.proposals`.
        dprior: Log-prior density.  Pure JAX function with signature
            ``dprior(theta_arr) -> scalar``.
        dt_array_extended, nstep_array, t0, times, ys: Particle-filter
            inputs (see :func:`_pfilter_internal`).
        J: Number of particles.
        rinitializer, rprocess_interp, dmeasure: Particle-filter callbacks.
        accumvars: Accumulator state indices.
        covars_extended: Covariates, or ``None``.
        thresh: Adaptive resampling threshold.
        Nmcmc: Number of MCMC iterations (static).
        key: PRNG key.
        should_trans: Whether parameter transformations are applied inside
            the particle filter (static, default False).

    Returns:
        ``(loglik_trace, log_prior_trace, theta_trace, accepts)`` where:

        * ``loglik_trace`` has shape ``(Nmcmc + 1,)`` (iteration 0 is the
          initial evaluation).
        * ``log_prior_trace`` has shape ``(Nmcmc + 1,)``.
        * ``theta_trace`` has shape ``(Nmcmc + 1, d)``.
        * ``accepts`` is a scalar count of accepted proposals.
    """

    def _run_pfilter(theta: jax.Array, k: jax.Array) -> jax.Array:
        out = _pfilter_internal(
            theta,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            ys,
            J,
            rinitializer,
            rprocess_interp,
            dmeasure,
            accumvars,
            covars_extended,
            thresh,
            k,
            False,  # CLL
            False,  # ESS
            False,  # filter_mean
            False,  # prediction_mean
            should_trans,
        )
        return -out["neg_loglik"]

    # ---- Initial evaluation at starting theta ----
    key, init_pf_key = jax.random.split(key)
    loglik0 = _run_pfilter(theta_arr, init_pf_key)
    log_prior0 = dprior(theta_arr)
    prop_state0 = proposal.init_state(theta_arr)

    init_carry = (
        theta_arr,                    # current accepted theta
        loglik0,                      # current accepted loglik
        log_prior0,                   # current accepted log prior
        prop_state0,                  # adaptive proposal state
        jnp.array(0, dtype=jnp.int32),  # accepts counter
        key,                          # PRNG key
    )

    def step(carry, n):
        theta_cur, ll_cur, lp_cur, prop_state, accepts, key = carry
        key, prop_key, pf_key, accept_key = jax.random.split(key, 4)

        # Propose (and update proposal state)
        theta_prop, new_prop_state = proposal.step(
            prop_state, theta_cur, prop_key, n, accepts
        )

        # Log prior at proposal
        lp_prop = dprior(theta_prop)

        # Particle-filter likelihood at proposal
        ll_prop = _run_pfilter(theta_prop, pf_key)

        # Acceptance probability (Metropolis with symmetric proposal)
        log_alpha = ll_prop + lp_prop - ll_cur - lp_cur
        u = jax.random.uniform(accept_key)
        accept = (
            jnp.isfinite(log_alpha)
            & jnp.isfinite(lp_prop)
            & (jnp.log(u) < log_alpha)
        )

        new_theta = jnp.where(accept, theta_prop, theta_cur)
        new_ll = jnp.where(accept, ll_prop, ll_cur)
        new_lp = jnp.where(accept, lp_prop, lp_cur)
        new_accepts = accepts + accept.astype(jnp.int32)

        out = (new_ll, new_lp, new_theta)
        new_carry = (new_theta, new_ll, new_lp, new_prop_state, new_accepts, key)
        return new_carry, out

    final_carry, (ll_trace, lp_trace, theta_trace) = jax.lax.scan(
        step, init_carry, jnp.arange(1, Nmcmc + 1, dtype=jnp.int32)
    )

    final_accepts = final_carry[4]

    # Prepend iteration-0 (the initial evaluation) to each trace.
    ll_trace = jnp.concatenate((jnp.asarray([loglik0]), ll_trace))
    lp_trace = jnp.concatenate((jnp.asarray([log_prior0]), lp_trace))
    theta_trace = jnp.concatenate((theta_arr[None, :], theta_trace), axis=0)

    return ll_trace, lp_trace, theta_trace, final_accepts


# vmap over the chain (replicate) dimension: theta_arr (0), key (0), everything else None.
# proposal flows through unchanged (its leaves are not chain-dependent).
_vmapped_pmcmc_internal = jax.vmap(
    _pmcmc_internal,
    in_axes=(
        0,        # theta_arr per chain
        None,     # proposal
        None,     # dprior
        None,     # dt_array_extended
        None,     # nstep_array
        None,     # t0
        None,     # times
        None,     # ys
        None,     # J
        None,     # rinitializer
        None,     # rprocess_interp
        None,     # dmeasure
        None,     # accumvars
        None,     # covars_extended
        None,     # thresh
        None,     # Nmcmc
        0,        # key per chain
        None,     # should_trans
    ),
)

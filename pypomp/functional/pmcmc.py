"""
Pure-functional PMCMC entry point.

This is a thin wrapper around :func:`pypomp.core.algorithms.pmcmc._vmapped_pmcmc_internal`
that takes a compiled :class:`pypomp.functional.structs.PompStruct` and runs
``n_chains`` independent PMCMC chains in parallel via ``jax.vmap``.
"""

from typing import Callable

import jax

from .structs import PompStruct
from ..core.algorithms.pmcmc import _vmapped_pmcmc_internal


def pmcmc(
    struct: PompStruct,
    thetas_array: jax.Array,
    proposal,
    dprior: Callable,
    Nmcmc: int,
    J: int,
    thresh: float,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Functional PMCMC entry point.

    Runs ``n_chains`` independent particle-MCMC chains in parallel.  Each chain
    starts at the corresponding row of ``thetas_array`` using the corresponding
    PRNG key in ``keys``.  Intended for users who need to compose PMCMC inside
    larger JAX programs; see :meth:`pypomp.core.pomp.Pomp.pmcmc` for a
    higher-level interface.

    Parameters
    ----------
    struct : PompStruct
        Compiled POMP model (see :class:`PompStruct`).
    thetas_array : jax.Array
        Starting parameter vectors, shape ``(n_chains, d)``.
    proposal
        Proposal object (see :mod:`pypomp.proposals`).  Shared across
        chains; per-chain state is initialised internally.
    dprior : Callable
        Log-prior density.  Pure JAX function with signature
        ``dprior(theta_arr) -> scalar``.
    Nmcmc : int
        Number of MCMC iterations per chain.
    J : int
        Number of particles per filter evaluation.
    thresh : float
        Adaptive resampling threshold for the particle filter.
    keys : jax.Array
        PRNG keys, shape ``(n_chains, ...)``.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        Tuple ``(loglik_traces, log_prior_traces, theta_traces, accepts)``:

        * ``loglik_traces``: shape ``(n_chains, Nmcmc + 1)``.
        * ``log_prior_traces``: shape ``(n_chains, Nmcmc + 1)``.
        * ``theta_traces``: shape ``(n_chains, Nmcmc + 1, d)``.
        * ``accepts``: shape ``(n_chains,)`` -- count of accepted proposals per chain.
    """
    if struct.dmeas_pf is None:
        raise ValueError("PMCMC requires struct.dmeas_pf to be non-None.")
    return _vmapped_pmcmc_internal(
        thetas_array,
        proposal,
        dprior,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys,
        J,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.accumvars,
        struct.covars_extended,
        thresh,
        Nmcmc,
        keys,
        False,  # should_trans
    )

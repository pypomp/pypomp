"""
Pure-functional ABC-MCMC entry point.

Thin wrapper over :func:`pypomp.core.algorithms.abc._vmapped_abc_internal`
that takes a :class:`pypomp.functional.structs.PompStruct` plus pre-computed
``obs_probes`` and ``scale_arr`` and runs ``n_chains`` chains in parallel.
"""

from typing import Callable

import jax

from .structs import PompStruct
from ..core.algorithms.abc import _vmapped_abc_internal


def abc(
    struct: PompStruct,
    thetas_array: jax.Array,
    proposal,
    dprior: Callable,
    probe_fn: Callable,
    obs_probes: jax.Array,
    scale_arr: jax.Array,
    epsilon: float,
    ydim: int,
    Nabc: int,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Functional ABC-MCMC entry point.

    Args:
        struct: Compiled POMP model (see :class:`PompStruct`).  Requires
            ``struct.rmeas_pf`` to be non-``None``.
        thetas_array: Starting parameter vectors, shape ``(n_chains, d)``.
        proposal: Proposal object (see :mod:`pypomp.proposals`).
        dprior: Pure-JAX log-prior, ``dprior(theta_arr) -> scalar``.
        probe_fn: Pure-JAX probe function, ``probe_fn(y_arr) -> (n_probes,)``
            where ``y_arr`` has shape ``(n_obs, ydim)``.
        obs_probes: Observed probes, shape ``(n_probes,)``.
        scale_arr: Per-probe scale, shape ``(n_probes,)``.
        epsilon: ABC distance threshold (acceptance requires
            ``distance < epsilon**2``).
        ydim: Observation dimensionality (static).
        Nabc: Number of MCMC iterations per chain.
        keys: PRNG keys, shape ``(n_chains, ...)``.

    Returns:
        ``(distance_traces, log_prior_traces, theta_traces, accepts)`` with
        shapes ``(n_chains, Nabc + 1)``, ``(n_chains, Nabc + 1)``,
        ``(n_chains, Nabc + 1, d)``, ``(n_chains,)`` respectively.
    """
    if struct.rmeas_pf is None:
        raise ValueError("ABC requires struct.rmeas_pf to be non-None.")
    return _vmapped_abc_internal(
        thetas_array,
        proposal,
        dprior,
        probe_fn,
        obs_probes,
        scale_arr,
        epsilon,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.rmeas_pf,
        struct.accumvars,
        struct.covars_extended,
        ydim,
        Nabc,
        keys,
    )

"""
Pure-functional ABC-MCMC entry point.

Thin wrapper over :func:`pypomp.core.algorithms.abc._vmapped_abc_internal`
that takes a :class:`pypomp.functional.structs.PompStruct` plus pre-computed
``obs_probes`` and ``scale_arr`` and runs ``n_chains`` chains in parallel.
"""

from typing import Any, Callable

import jax

from .structs import PompStruct
from ..core.algorithms.abc import _vmapped_abc_internal
from ..core.algorithms.types import AbcConfig, AbcInputs


def abc(
    struct: PompStruct,
    thetas_array: jax.Array,
    proposal: Any,
    dprior: Callable,
    probe_fn: Callable,
    obs_probes: jax.Array,
    scale_arr: jax.Array,
    epsilon: float,
    ydim: int,
    M: int,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Functional ABC-MCMC entry point.

    Parameters
    ----------
    struct : PompStruct
        Compiled POMP model (see :class:`PompStruct`).  Requires
        ``struct.rmeas_pf`` to be non-``None``.
    thetas_array : jax.Array
        Starting parameter vectors, shape ``(n_chains, d)``.
    proposal
        Proposal object (see :mod:`pypomp.proposals`).
    dprior : Callable
        Pure-JAX log-prior, ``dprior(theta_arr) -> scalar``.
    probe_fn : Callable
        Pure-JAX probe function, ``probe_fn(y_arr) -> (n_probes,)``
        where ``y_arr`` has shape ``(n_obs, ydim)``.
    obs_probes : jax.Array
        Observed probes, shape ``(n_probes,)``.
    scale_arr : jax.Array
        Per-probe scale, shape ``(n_probes,)``.
    epsilon : float
        ABC distance threshold (acceptance requires ``distance < epsilon**2``).
    ydim : int
        Observation dimensionality (static).
    M : int
        Number of MCMC iterations per chain.
    keys : jax.Array
        PRNG keys, shape ``(n_chains, ...)``.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(distance_traces, log_prior_traces, theta_traces, accepts)`` with
        shapes ``(n_chains, M + 1)``, ``(n_chains, M + 1)``,
        ``(n_chains, M + 1, d)``, ``(n_chains,)`` respectively.
    """
    if struct.rmeas_pf is None:
        raise ValueError("ABC requires struct.rmeas_pf to be non-None.")

    config = AbcConfig.from_abc_struct(
        struct,
        M=M,
        dprior=dprior,
        probe_fn=probe_fn,
        ydim=ydim,
    )
    inputs = AbcInputs.from_abc_struct(
        struct,
        obs_probes=obs_probes,
        scale_arr=scale_arr,
        epsilon=epsilon,
    )

    return _vmapped_abc_internal(
        thetas_array,
        proposal,
        config,
        inputs,
        keys,
    )

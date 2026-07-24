"""
Pure-functional ABC-MCMC entry point.

Thin wrapper over :func:`pypomp.core.algorithms.abc._vmapped_abc_internal`
that takes a :class:`pypomp.functional.structs.PompStruct` plus pre-computed
``obs_probes`` and ``scale_arr`` and runs ``n_chains`` chains in parallel.
"""

from typing import Any, Callable

import jax

from .structs import PompStruct, resolve_dprior
from ..core.algorithms.abc import _vmapped_abc_internal
from ..core.algorithms.types import AbcConfig, AbcInputs


def abc(
    struct: PompStruct,
    thetas_array: jax.Array,
    proposal: Any,
    probe_fn: Callable,
    obs_probes: jax.Array,
    scale_arr: jax.Array,
    epsilon: float,
    M: int,
    keys: jax.Array,
    dprior: Callable | None = None,
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
    probe_fn : Callable
        Pure-JAX probe function, ``probe_fn(y_arr) -> (n_probes,)``
        where ``y_arr`` has shape ``(n_obs, ydim)``.
    obs_probes : jax.Array
        Observed probes, shape ``(n_probes,)``.
    scale_arr : jax.Array
        Per-probe scale, shape ``(n_probes,)``.
    epsilon : float
        ABC distance threshold (acceptance requires ``distance < epsilon**2``).
    M : int
        Number of MCMC iterations per chain.
    keys : jax.Array
        PRNG keys, shape ``(n_chains, ...)``.
    dprior : Callable or None, optional
        Pure-JAX log-prior function or ``None`` to use ``struct.dprior_pf``
        (or, if that is absent too, a flat prior on the natural parameter
        scale). Sampling is performed on the estimation scale, so the
        change-of-variables log-Jacobian is added to the est-scale acceptance
        ratio internally; this term is *not* included in the recorded
        ``log_prior_traces`` (see Returns). See :ref:`dprior-tutorial`.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        ``(distance_traces, log_prior_traces, theta_traces, accepts)`` with
        shapes ``(n_chains, M + 1)``, ``(n_chains, M + 1)``,
        ``(n_chains, M + 1, d)``, ``(n_chains,)`` respectively.
        ``log_prior_traces`` is evaluated on the **natural** parameter scale
        (no Jacobian), consistent with ``theta_traces``; a flat prior records
        0.0.
    """
    if struct.rmeas_pf is None:
        raise ValueError("ABC requires struct.rmeas_pf to be non-None.")

    thetas_est = struct.par_trans._transform_array(
        thetas_array,
        struct.param_names,
        direction="to_est",
    )

    proposal = proposal.canonicalize(struct.param_names)

    dprior_fn = resolve_dprior(dprior, struct)
    ydim = int(struct.ys.shape[1])
    config = AbcConfig.from_abc_struct(
        struct,
        M=M,
        dprior=dprior_fn,
        probe_fn=probe_fn,
        ydim=ydim,
    )

    inputs = AbcInputs.from_abc_struct(
        struct,
        obs_probes=obs_probes,
        scale_arr=scale_arr,
        epsilon=epsilon,
    )

    dist_traces, _lp_est_traces, theta_est_traces, accepts = _vmapped_abc_internal(
        thetas_est,
        proposal,
        config,
        inputs,
        keys,
    )
    theta_natural_traces = struct.par_trans._transform_array(
        theta_est_traces,
        struct.param_names,
        direction="from_est",
    )
    # The scan uses est-scale priors (natural prior + change-of-variables
    # Jacobian) for the acceptance ratio, but the recorded traces live on the
    # natural scale. Recompute the log-prior at the natural-scale parameters
    # (``should_trans=False`` -> no Jacobian) so the reported ``log_prior`` is
    # consistent with ``theta`` and can be recomputed independently by the user.
    lp_traces = jax.vmap(jax.vmap(lambda th: dprior_fn(th, False)))(
        theta_natural_traces
    )
    return dist_traces, lp_traces, theta_natural_traces, accepts

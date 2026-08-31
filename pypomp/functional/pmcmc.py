"""
Pure-functional PMCMC entry point.

This is a thin wrapper around :func:`pypomp.core.algorithms.pmcmc._vmapped_pmcmc_internal`
that takes a compiled :class:`pypomp.functional.structs.PompStruct` and runs
``n_chains`` independent PMCMC chains in parallel via ``jax.vmap``.
"""

from collections.abc import Callable

import jax

from ..core.algorithms.contexts import PmcmcContext
from ..core.algorithms.pmcmc import _vmapped_pmcmc_internal
from .structs import PompStruct, resolve_dprior


def pmcmc(
    struct: PompStruct,
    thetas_array: jax.Array,
    proposal,
    M: int,
    J: int,
    keys: jax.Array,
    thresh: float = 0.0,
    dprior: Callable | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Functional PMCMC entry point.

    Runs ``n_chains`` independent particle-MCMC chains in parallel.  Each chain
    starts at the corresponding row of ``thetas_array`` using the corresponding
    PRNG key in ``keys``.  Intended for users who need to compose PMCMC inside
    larger JAX programs; see :meth:`pypomp.core.pomp.Pomp._pmcmc` for a
    higher-level interface.

    Parameters
    ----------
    struct : PompStruct
        Compiled POMP model (see :class:`PompStruct`).
    thetas_array : jax.Array
        Starting parameter vectors, shape ``(n_chains, d)``.
    proposal
        Proposal object (see :mod:`pypomp.proposals`).
    M : int
        Number of MCMC iterations per chain.
    J : int
        Number of particles per filter evaluation.
    keys : jax.Array
        PRNG keys, shape ``(n_chains, ...)``.
    thresh : float, optional
        Adaptive resampling threshold for the particle filter.
        Default is 0.0.
    dprior : Callable or None, optional
        Pure-JAX log-prior function or ``None`` to use ``struct.dprior_pf``
        (or, if that is absent too, defaults to a flat prior on the natural parameter
        scale). Sampling is performed on the estimation scale, so the
        change-of-variables log-Jacobian is added to the est-scale acceptance
        ratio internally; this term is *not* included in the recorded
        ``log_prior_traces`` (see Returns). See :ref:`dprior-tutorial`.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        Tuple ``(loglik_traces, log_prior_traces, theta_traces, accepts)``:

        * ``loglik_traces``: shape ``(n_chains, M + 1)``.
        * ``log_prior_traces``: shape ``(n_chains, M + 1)``. Log-prior evaluated
          on the **natural** parameter scale (no Jacobian), consistent with
          ``theta_traces`` and ``loglik_traces``. A flat prior records 0.0.
        * ``theta_traces``: shape ``(n_chains, M + 1, d)``. Natural scale.
        * ``accepts``: shape ``(n_chains,)`` -- count of accepted proposals per chain.
    """
    if struct.dmeas_pf is None:
        raise ValueError("PMCMC requires struct.dmeas_pf to be non-None.")
    thetas_est = struct.par_trans._transform_array(
        thetas_array,
        struct.param_names,
        direction="to_est",
    )
    proposal = proposal.canonicalize(struct.param_names)
    dprior_fn = resolve_dprior(dprior, struct)
    context = PmcmcContext.from_struct(
        struct=struct,
        M=M,
        J=J,
        dprior=dprior_fn,
        thresh=thresh,
    )

    ll_traces, _lp_est_traces, theta_est_traces, accepts = _vmapped_pmcmc_internal(
        thetas_est,
        proposal,
        context,
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
    # consistent with ``theta`` and ``logLik`` and can be recomputed
    # independently by the user.
    lp_traces = jax.vmap(jax.vmap(lambda th: dprior_fn(th, False)))(
        theta_natural_traces
    )
    return ll_traces, lp_traces, theta_natural_traces, accepts

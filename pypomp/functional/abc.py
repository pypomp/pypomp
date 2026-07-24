"""
Pure-functional ABC-MCMC entry point.

Thin wrapper over :func:`pypomp.core.algorithms.abc._vmapped_abc_internal`
that takes a :class:`pypomp.functional.structs.PompStruct` plus a ``probes``
dict and runs ``n_chains`` chains in parallel.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from pypomp.proposals import Proposal

from .structs import PompStruct, resolve_dprior
from ..core.algorithms.abc import _vmapped_abc_internal
from ..core.algorithms.types import AbcConfig, AbcInputs


def abc(
    struct: PompStruct,
    thetas_array: jax.Array,
    proposal: Proposal,
    probes: dict[str, Callable],
    epsilon: float,
    M: int,
    keys: jax.Array,
    scale: dict[str, float] | None = None,
    dprior: Callable | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Functional ABC-MCMC entry point.

    Parameters
    ----------
    struct : PompStruct
        Compiled POMP model (see :class:`PompStruct`).  Requires
        ``struct.rmeas_pf`` to be non-``None``.
    thetas_array : jax.Array
        Starting parameter vectors, shape ``(n_chains, d)``.
    proposal
        Proposal object (see :mod:`pypomp.proposals`).
    probes : dict
        Mapping from probe name (``str``) to a pure-JAX summary-statistic
        callable ``probe_fn(y_arr) -> scalar``, where ``y_arr`` is a
        simulated observation array with shape ``(n_obs, ydim)``.
    epsilon : float
        ABC distance threshold (acceptance requires ``distance < epsilon**2``).
    M : int
        Number of MCMC iterations per chain.
    keys : jax.Array
        PRNG keys, shape ``(n_chains, ...)``.
    scale : dict, optional
        Mapping from probe name (``str``, matching the keys of ``probes``)
        to a positive scaling factor (``float``) used to normalize probe
        differences in the squared scaled Euclidean distance, e.g.,
        :math:`d = \sum_i \left( \frac{s_i(y^*) - s_i(y)}{w_i} \right)^2`
        where :math:`w_i` is ``scale[i]``. If ``None``, a scale of ``1.0``
        is used for every probe.
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
    if not probes:
        raise ValueError("probes must be a non-empty dict.")
    if scale is None:
        scale = {name: 1.0 for name in probes}
    if set(scale.keys()) != set(probes.keys()):
        raise ValueError("scale keys must match probes keys.")
    for name, value in scale.items():
        if value <= 0:
            raise ValueError(f"scale['{name}'] must be positive.")

    probe_names = sorted(probes.keys())
    scale_arr = jnp.asarray([float(scale[name]) for name in probe_names])

    def probe_fn(y_arr: jax.Array) -> jax.Array:
        return jnp.stack(
            [jnp.asarray(probes[name](y_arr)).reshape(()) for name in probe_names]
        )

    obs_probes = probe_fn(jnp.asarray(struct.ys))

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

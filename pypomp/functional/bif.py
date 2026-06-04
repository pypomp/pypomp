"""
Pure-functional BIF Stage 1 entry point.

The functional routine returns the Stage 1 IF cloud and traces. The high-level
``Pomp.bif`` method performs Stage 2 deconvolution and user-facing summaries.
"""

from typing import Callable

import jax

from .structs import PompStruct
from ..core.algorithms.bif import _jv_bif_internal


def bif(
    struct: PompStruct,
    thetas_array: jax.Array,
    rw_sigmas_array: jax.Array,
    perturb_sigmas_array: jax.Array,
    M: int,
    a: float,
    J: int,
    thresh: float,
    keys: jax.Array,
    dprior: Callable,
    n_monitors: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Run BIF Stage 1 in estimation space.

    Args:
        struct: Compiled POMP model.
        thetas_array: Initial parameter cloud, shape ``(J, n_reps, d)``.
        rw_sigmas_array: Within-trajectory random-walk standard deviations.
        perturb_sigmas_array: Fixed initial-perturbation standard deviations.
        M: Number of outer iterations.
        a: Geometric cooling fraction for the within-trajectory random walk.
        J: Number of particles.
        thresh: Adaptive resampling threshold.
        keys: One PRNG key per initial parameter replicate.
        dprior: Log-prior density on the estimation-scale parameter vector.
        n_monitors: Number of likelihood-monitoring particle filters per
            iteration. Use 0 to record the perturbed-filter estimate.

    Returns:
        ``(neg_logliks, theta_traces, final_cloud)``.
    """
    return _jv_bif_internal(
        thetas_array,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys,
        struct.rinit_per,
        struct.rproc_per,
        struct.dmeas_per,
        rw_sigmas_array,
        perturb_sigmas_array,
        struct.accumvars,
        struct.covars_extended,
        M,
        a,
        J,
        thresh,
        keys,
        dprior,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        n_monitors,
    )

import jax
from .structs import PompStruct
from ..core.algorithms.mop import _vmapped_mop_internal


def mop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
) -> jax.Array:
    """
    This is a pure functional implementation of the MOP differentiable particle
    filter, intended for users who need to compose it within custom JAX
    loops or higher-order functions.

    Unlike the standard particle filter (:func:`pypomp.functional.pfilter`), the MOP objective is specifically
    designed to be fully differentiable with respect to the model parameters. This allows
    for the computation of gradients and Hessians of the log-likelihood using
    JAX's automatic differentiation capabilities.

    This function evaluates the log-likelihood for the given parameter sets, but it is
    primarily intended to be used as an objective function within gradient-based
    optimization routines (e.g., :func:`pypomp.functional.train`).

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, n_params).
        J (int): Number of particles.
        alpha (float): Alpha parameter for MOP.
        keys (jax.Array): Random keys. Shape (n_reps, ...).

    Returns:
        jax.Array: Negative MOP log-likelihood estimates.
    """
    return _vmapped_mop_internal(
        thetas_array,
        struct.ys,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        J,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.accumvars,
        struct.covars_extended,
        alpha,
        keys,
    )

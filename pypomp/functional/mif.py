import jax
from typing import Callable
from .structs import PompStruct
from ..core.algorithms.mif import _jv_mif_internal


def mif(
    struct: PompStruct,
    thetas_array: jax.Array,
    sigmas_array: jax.Array,
    sigmas_init_array: jax.Array,
    M: int,
    cooling_fn: Callable | float,
    J: int,
    thresh: float,
    keys: jax.Array,
    n_monitors: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    This is a pure functional implementation of the Iterated Filtering algorithm,
    intended for users who need to compose it within custom JAX loops or
    higher-order functions. For a more user-friendly (but impurely-functional) interface, see
    :meth:`pypomp.core.pomp.Pomp.mif`.

    This implementation leverages JAX to efficiently vectorize the algorithm across
    multiple initial parameter sets simultaneously.

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (J, n_reps, n_params).
            Note that the batch dimension for `vmap` is the second axis (`n_reps`).
        sigmas_array (jax.Array): Array of random walk sigmas. Shape (n_params,).
        sigmas_init_array (jax.Array): Array of initial random walk sigmas. Shape (n_params,).
        M (int): Number of iterations.
        cooling_fn (Callable | float): Cooling function taking (nt, m, ntimes) or float cooling factor.
        J (int): Number of particles.
        thresh (float): Resampling threshold.
        keys (jax.Array): Random keys. Shape (n_reps, ...).
        n_monitors (int): Number of monitors for likelihood averaging.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]:
            Negative log-likelihood history: Shape (n_reps, M).
            Parameter trace history: Shape (n_reps, M+1, n_params).
            Final particle swarm: Shape (n_reps, J, n_params).
    """
    res = _jv_mif_internal(
        thetas_array,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys,
        struct.rinit_per,
        struct.rproc_per,
        struct.dmeas_per,
        sigmas_array,
        sigmas_init_array,
        struct.accumvars,
        struct.covars_extended,
        M,
        cooling_fn,
        0,
        J,
        thresh,
        keys,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        n_monitors,
        False,
    )
    return res[0], res[1], res[2]

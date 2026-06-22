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
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, J, n_params) on the natural scale.
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
            Parameter trace history: Shape (n_reps, M+1, n_params) on the natural scale.
            Final particle swarm: Shape (n_reps, J, n_params) on the natural scale.
    """

    thetas_est = struct.par_trans._transform_array_jax(
        thetas_array,
        struct.param_names,
        direction="to_est",
    )

    thetas_est_transposed = thetas_est.transpose(1, 0, 2)

    res = _jv_mif_internal(
        thetas_est_transposed,
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
    traces_natural = struct.par_trans._transform_array_jax(
        res[1],
        struct.param_names,
        direction="from_est",
    )
    final_thetas_natural = struct.par_trans._transform_array_jax(
        res[2],
        struct.param_names,
        direction="from_est",
    )
    return res[0], traces_natural, final_thetas_natural

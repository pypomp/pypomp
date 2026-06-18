import jax
from .structs import PompStruct
from ..core.algorithms.dpop import _vmapped_dpop_internal


def dpop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    process_weight_index: int,
    keys: jax.Array,
) -> jax.Array:
    """
    This is a pure functional implementation of the DPOP differentiable particle
    filter, intended for users who need to compose it within custom JAX
    loops or higher-order functions. For a more user-friendly (but impurely-functional) interface, see
    :meth:`pypomp.core.pomp.Pomp.dpop_train`.

    This function is analogous to :func:`pypomp.functional.mop` as a fully differentiable objective function
    for parameter estimation. However, it additionally
    incorporates a per-interval transition log-weight that is
    assumed to be stored in one of the state components.

    The process log-weight is expected to be accumulated over a
    single observation interval by the user-specified process
    model. At the beginning of each interval, the corresponding
    state component should be reset to zero (this is naturally
    handled by ``accumvars``).

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, n_params).
        J (int): Number of particles.
        alpha (float): Alpha parameter for DPOP.
        process_weight_index (int): Index of the process weight state.
        keys (jax.Array): Random keys. Shape (n_reps, ...).

    Returns:
        jax.Array: Negative DPOP log-likelihood estimates.
    """
    return _vmapped_dpop_internal(
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
        process_weight_index,
        len(struct.times),
        keys,
    )

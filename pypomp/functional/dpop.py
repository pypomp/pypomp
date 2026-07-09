import warnings
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
    """DPOP differentiable particle filter log-likelihood objective.

    A pure functional implementation of the DPOP differentiable particle
    filter, intended for composition within custom JAX loops or
    higher-order functions.

    .. warning::
       This function is experimental.  Its API and behavior are subject to change
       in future releases.

    This function is analogous to :func:`pypomp.functional.mop` as a fully
    differentiable objective function for parameter estimation.  However, it
    additionally incorporates a per-interval transition log-weight that is
    assumed to be stored in one of the state components.

    The process log-weight is expected to be accumulated over a single
    observation interval by the user-specified process model.  At the
    beginning of each interval, the corresponding state component should be
    reset to zero (this is naturally handled by ``accumvars``).

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.
    thetas_array : jax.Array
        Array of initial parameters of shape ``(n_reps, n_params)``, aligned
        with the canonical order of ``struct.param_names``.
    J : int
        Number of particles.
    alpha : float
        Alpha parameter for DPOP.
    process_weight_index : int
        Index of the process weight state component.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.

    Returns
    -------
    jax.Array
        Negative DPOP log-likelihood estimates.

    See Also
    --------
    pypomp.Pomp.dpop_train : High-level OOP training interface.
    pypomp.functional.align_params : Prepare parameter arrays.
    """
    warnings.warn(
        "dpop is experimental and its API and behavior are subject to change.",
        category=FutureWarning,
        stacklevel=2,
    )

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

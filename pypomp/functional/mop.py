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
    """MOP differentiable particle filter log-likelihood objective.

    A pure functional implementation of the Measurement Off-Parameter (MOP)
    differentiable particle filter (Tan et al. 2024 [1]_), intended for composition
    within custom JAX loops or higher-order functions.

    Unlike the standard particle filter (:func:`~pypomp.functional.pfilter`), the MOP
    objective is designed to be fully differentiable with respect to the model
    parameters using automatic differentiation.

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
        Alpha parameter for MOP.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.

    Returns
    -------
    jax.Array
        Negative MOP log-likelihood estimates.

    See Also
    --------
    pypomp.Pomp.train : High-level OOP training interface.
    pypomp.functional.align_params : Prepare parameter arrays.

    References
    ----------
    .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
       for Partially Observed Markov Processes using Automatic Differentiation."
       *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.
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

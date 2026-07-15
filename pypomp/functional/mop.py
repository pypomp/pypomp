import jax
from .structs import PompStruct
from ..core.algorithms.mop import _vmapped_mop_internal
from ..core.algorithms.types import MopConfig, MopInputs


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

    config = MopConfig.from_mop_struct(struct, J)
    inputs = MopInputs.from_mop_struct(struct, alpha)

    return _vmapped_mop_internal(
        thetas_array,
        keys,
        config,
        inputs,
    )

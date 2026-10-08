import warnings

import jax

from ..core.algorithms.contexts import MopContext
from ..core.algorithms.mop import _vmapped_mop_internal
from .structs import PompStruct


def _warn_dpop_experimental(stacklevel: int) -> None:
    """Warn that DPOP (``pop`` or a process-weight argument) is experimental."""
    warnings.warn(
        "DPOP (pop, or a process-weight argument to train) is experimental and "
        "its API and behavior are subject to change.",
        category=FutureWarning,
        stacklevel=stacklevel + 1,
    )


def _mop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
    process_weight_index: int | None,
) -> jax.Array:
    """Shared core of :func:`mop` and :func:`pop`."""
    context = MopContext.from_struct(
        struct, J=J, alpha=alpha, process_weight_index=process_weight_index
    )
    return _vmapped_mop_internal(thetas_array, keys, context)


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
        Array of parameters of shape ``(n_reps, n_params)`` on the
        estimation scale, aligned with the canonical order of
        ``struct.param_names``.
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

    Notes
    -----
    Because :func:`mop` internally transforms parameters from the estimation
    scale to the natural scale using ``struct.par_trans.from_est`` within its
    differentiable computation graph, gradients computed via automatic
    differentiation (e.g. :func:`jax.grad`) are with respect to the estimation
    parameters.

    Callers are responsible for transforming initial parameters to the
    estimation scale before the optimization loop (e.g. via
    :meth:`~pypomp.ParTrans._transform_array` with ``direction="to_est"``) and
    transforming optimized estimates back to the natural scale
    (``direction="from_est"``) when the loop concludes.

    See Also
    --------
    pypomp.functional.pop : MOP with the process score added to the weights.
    pypomp.Pomp.train : High-level OOP training interface.
    pypomp.functional.align_params : Prepare parameter arrays.

    References
    ----------
    .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
       for Partially Observed Markov Processes using Automatic Differentiation."
       *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.
    """
    return _mop(struct, thetas_array, J, alpha, keys, None)


def pop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
    process_weight_index: int,
) -> jax.Array:
    """POP differentiable particle filter log-likelihood objective (experimental).

    The Process Off-Parameter (POP) objective is :func:`mop` with the score of
    the process log-density added to the particle weights.  It gives useful
    gradients for process models whose sample paths are not differentiable in
    the parameters, such as discrete-state models.  Its value equals
    :func:`mop`'s for the same keys; only the gradient differs.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.
    thetas_array : jax.Array
        Array of parameters of shape ``(n_reps, n_params)`` on the
        estimation scale, aligned with the canonical order of
        ``struct.param_names``.
    J : int
        Number of particles.
    alpha : float
        Alpha parameter for MOP.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    process_weight_index : int
        Index of the state holding the process log-weight (see Notes).  Must
        be one of ``struct.accumvars``.

    Returns
    -------
    jax.Array
        Negative log-likelihood estimates.

    Notes
    -----
    For the gradient to be correct:

    - The state holds the log-density of the transitions sampled during the
      current observation interval, or a surrogate with the same gradient in
      the parameters, and is an accumulator variable (``accumvars``) so that
      it is reset at every observation time.
    - Sampled quantities whose log-density is accumulated there are wrapped
      in :func:`jax.lax.stop_gradient`.  Otherwise their pathwise gradient is
      added to the score and counted twice.
    - The model's gradients are finite, e.g. by clipping probabilities before
      taking ``log``.  NaN gradients are not masked.

    Parameters are on the estimation scale, as for :func:`mop`.

    See Also
    --------
    pypomp.functional.mop : The same objective without the process score.
    pypomp.Pomp.train : Pass ``process_weight_state`` to train with POP.
    """
    if process_weight_index is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise ValueError("pop requires a process_weight_index; use mop for MOP.")
    _warn_dpop_experimental(stacklevel=2)
    return _mop(struct, thetas_array, J, alpha, keys, process_weight_index)

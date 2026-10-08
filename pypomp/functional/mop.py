import warnings

import jax

from ..core.algorithms.contexts import MopContext
from ..core.algorithms.mop import _vmapped_mop_internal
from .structs import PompStruct


def _warn_dpop_experimental(stacklevel: int) -> None:
    """Warn that DPOP, enabled by a process-weight argument, is experimental."""
    warnings.warn(
        "DPOP (a process-weight argument) is experimental and its API and "
        "behavior are subject to change.",
        category=FutureWarning,
        stacklevel=stacklevel + 1,
    )


def mop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
    process_weight_index: int | None = None,
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
    process_weight_index : int or None, optional
        Index of the state holding the process log-weight, which enables
        DPOP (see Notes).  Must be one of ``struct.accumvars``.  Defaults to
        ``None`` (plain MOP).

    Returns
    -------
    jax.Array
        Negative MOP log-likelihood estimates.

    Notes
    -----
    DPOP (experimental) is enabled by giving the index of a state in which the
    process model accumulates the log-density of its own sampled transitions.
    The score of that log-density is added to the particle weights, which
    gives useful gradients for process models whose sample paths are not
    differentiable in the parameters, such as discrete-state models.  It
    changes only the gradient, not the log-likelihood estimate.  For the
    gradient to be correct:

    - The state holds the log-density of the transitions sampled during the
      current observation interval, or a surrogate with the same gradient in
      the parameters, and is an accumulator variable (``accumvars``) so that
      it is reset at every observation time.
    - Sampled quantities whose log-density is accumulated there are wrapped
      in :func:`jax.lax.stop_gradient`.  Otherwise their pathwise gradient is
      added to the score and counted twice.
    - The model's gradients are finite, e.g. by clipping probabilities before
      taking ``log``.  NaN gradients are not masked.

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
    pypomp.Pomp.train : High-level OOP training interface.
    pypomp.functional.align_params : Prepare parameter arrays.

    References
    ----------
    .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
       for Partially Observed Markov Processes using Automatic Differentiation."
       *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.
    """

    if process_weight_index is not None:
        _warn_dpop_experimental(stacklevel=2)
    context = MopContext.from_struct(
        struct, J=J, alpha=alpha, process_weight_index=process_weight_index
    )

    return _vmapped_mop_internal(
        thetas_array,
        keys,
        context,
    )

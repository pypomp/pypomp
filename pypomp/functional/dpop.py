"""Experimental DPOP objective and trainers.

DPOP extends MOP to process models whose sample paths are not differentiable
in the parameters, such as discrete-state models.  The process model records
the log-density of its own sampled transitions in a state variable, and the
score of that log-density is added to the particle weights.
"""

import warnings

import jax

from ..core.algorithms.contexts import MopContext
from ..core.algorithms.mop import _vmapped_mop_internal
from ..core.learning_rate import LearningRate
from ..core.optimizer import Optimizer
from .structs import PanelPompStruct, PompStruct
from .train import _panel_train, _train


def dpop(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    alpha: float,
    keys: jax.Array,
    process_weight_index: int,
) -> jax.Array:
    """DPOP differentiable particle filter log-likelihood objective.

    .. warning::
       This function is experimental.  Its API and behavior are subject to change
       in future releases.

    This is :func:`pypomp.functional.mop` with an additional term for the
    process model: the score of the transition log-density stored in state
    ``process_weight_index`` is added to the particle weights.  This gives
    useful gradients for process models whose sample paths do not depend
    differentiably on the parameters, such as discrete-state models.  The
    returned value is the same particle filter estimate that :func:`mop`
    returns for the same keys; only its gradient differs.

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
        Discount factor applied to the carried particle weights.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    process_weight_index : int
        Index of the state that accumulates the process log-weight.  It must
        be one of ``struct.accumvars``.

    Returns
    -------
    jax.Array
        Negative log-likelihood estimates of shape ``(n_reps,)``.

    Notes
    -----
    The process model must satisfy the following for the gradient to be
    correct:

    - The state at ``process_weight_index`` holds the log-density of the
      transitions sampled during the current observation interval, or a
      surrogate with the same gradient in the parameters.  It must be an
      accumulator variable (``accumvars``), so that it is reset at every
      observation time.
    - Sampled quantities whose log-density is accumulated there must be
      wrapped in :func:`jax.lax.stop_gradient`.  Otherwise their pathwise
      gradient is added to the score and counted twice.
    - The model's gradients must be finite, e.g. by clipping probabilities
      before taking ``log``.  NaN gradients are not masked.

    Because :func:`dpop` transforms parameters from the estimation scale to
    the natural scale inside its differentiable computation graph, gradients
    computed via automatic differentiation (e.g. :func:`jax.grad`) are with
    respect to the estimation parameters.

    See Also
    --------
    pypomp.functional.mop : The MOP objective that DPOP extends.
    """
    warnings.warn(
        "dpop is experimental and its API and behavior are subject to change.",
        category=FutureWarning,
        stacklevel=2,
    )
    context = MopContext.from_struct(
        struct, J=J, alpha=alpha, process_weight_index=process_weight_index
    )
    return _vmapped_mop_internal(thetas_array, keys, context)


def dpop_train(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    M: int,
    eta: LearningRate,
    keys: jax.Array,
    process_weight_index: int,
    optimizer: Optimizer | None = None,
    alpha: float | jax.Array = 0.97,
    alpha_cooling: float = 1.0,
    thresh: float = 0.0,
    n_monitors: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Optimize parameters via DPOP differentiable particle filter gradients.

    .. warning::
       This function is experimental.  Its API and behavior are subject to change
       in future releases.

    Identical to :func:`pypomp.functional.train` except that gradients come
    from the DPOP objective (see :func:`dpop`), which supports process models
    whose sample paths are not differentiable in the parameters, such as
    discrete-state models.  See :func:`dpop` for the requirements on the
    process model.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.
    thetas_array : jax.Array
        Initial parameter array of shape ``(n_reps, n_params)`` on the
        natural scale.  Must be aligned with ``struct.param_names``.
    J : int
        Number of particles.
    M : int
        Number of gradient steps.
    eta : LearningRate
        Per-parameter learning rates as a :class:`~pypomp.LearningRate` instance.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    process_weight_index : int
        Index of the state that accumulates the process log-weight.  It must
        be one of ``struct.accumvars``.
    optimizer : Optimizer or None, optional
        Optimizer configuration object.  Defaults to ``Adam()``.
    alpha : float or jax.Array, optional
        Discount factor applied to the carried particle weights.  Defaults to
        ``0.97``.
    alpha_cooling : float, optional
        Cosine cooling multiplier for ``alpha``.  Defaults to ``1.0``.
    thresh : float, optional
        ESS-based resampling threshold for the particle filters used by
        ``n_monitors > 1`` and by line search.  Defaults to ``0.0``.
    n_monitors : int, optional
        Number of unperturbed filter runs for log-likelihood monitoring.
        Defaults to ``1``.

    Returns
    -------
    tuple of (jax.Array, jax.Array)
        - Negative log-likelihood history of shape ``(n_reps, M + 1)``.  Row
          ``m`` is estimated at the parameters in row ``m`` of the trace.
        - Parameter trace history of shape ``(n_reps, M + 1, n_params)`` on the
          natural scale.

    See Also
    --------
    pypomp.functional.train : The MOP trainer that this extends.
    """
    warnings.warn(
        "dpop_train is experimental and its API and behavior are subject to change.",
        category=FutureWarning,
        stacklevel=2,
    )
    return _train(
        struct,
        thetas_array,
        J,
        M,
        eta,
        keys,
        optimizer,
        alpha,
        alpha_cooling,
        thresh,
        n_monitors,
        process_weight_index,
    )


def panel_dpop_train(
    struct: PanelPompStruct,
    shared_array: jax.Array,
    unit_array: jax.Array,
    J: int,
    M: int,
    eta: LearningRate,
    keys: jax.Array,
    process_weight_index: int,
    optimizer: Optimizer | None = None,
    alpha: float = 0.97,
    alpha_cooling: float = 1.0,
    chunk_size: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Optimize panel POMP parameters via DPOP differentiable particle filter gradients.

    .. warning::
       This function is experimental.  Its API and behavior are subject to change
       in future releases.

    Identical to :func:`pypomp.functional.panel_train` except that gradients
    come from the DPOP objective.  See :func:`dpop` for the requirements on
    the process model.

    Parameters
    ----------
    struct : PanelPompStruct
        Compiled structural representation of the Panel POMP model.
    shared_array : jax.Array
        Initial shared parameters of shape ``(n_reps, n_shared)`` on the
        natural scale.
    unit_array : jax.Array
        Initial unit-specific parameters of shape ``(n_reps, U, n_spec)`` on
        the natural scale.
    J : int
        Number of particles.
    M : int
        Number of iterations.
    eta : LearningRate
        Learning rates for shared and unit-specific parameters.
    keys : jax.Array
        Random keys of shape ``(n_reps, M + 1, U, ...)``.  Slab ``m < M``
        drives iteration ``m``; slab ``M`` evaluates the final parameters.
    process_weight_index : int
        Index of the state that accumulates the process log-weight.  It must
        be one of ``struct.accumvars``.
    optimizer : Optimizer or None, optional
        Optimizer configuration object (:class:`~pypomp.Adam`,
        :class:`~pypomp.SGD` or :class:`~pypomp.FullMatrixAdam`).  Defaults to
        ``Adam()``.
    alpha : float, optional
        Discount factor applied to the carried particle weights.  Defaults to
        ``0.97``.
    alpha_cooling : float, optional
        Cosine cooling multiplier for ``alpha``.  Defaults to ``1.0``.
    chunk_size : int, optional
        Number of units to process per gradient step; must divide ``U``.
        Defaults to ``1``.

    Returns
    -------
    tuple of (jax.Array, jax.Array, jax.Array)
        The negative log-likelihood history, shared parameter history, and
        unit-specific parameter history, as returned by
        :func:`pypomp.functional.panel_train`.

    See Also
    --------
    pypomp.functional.panel_train : The MOP trainer that this extends.
    """
    warnings.warn(
        "panel_dpop_train is experimental and its API and behavior are subject "
        "to change.",
        category=FutureWarning,
        stacklevel=2,
    )
    return _panel_train(
        struct,
        shared_array,
        unit_array,
        J,
        M,
        eta,
        keys,
        optimizer,
        alpha,
        alpha_cooling,
        chunk_size,
        process_weight_index,
    )

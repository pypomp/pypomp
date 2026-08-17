import warnings

import jax

from ..core.algorithms.contexts import DpopTrainContext
from ..core.algorithms.dpop import _vmapped_dpop_internal
from ..core.algorithms.train_dpop import _vmapped_dpop_train_internal
from ..core.learning_rate import LearningRate
from ..core.optimizer import Optimizer
from .structs import PompStruct


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
    pypomp.Pomp._dpop_train : High-level OOP training interface.
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


def dpop_train(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    optimizer: Optimizer,
    M: int,
    eta: LearningRate,
    alpha: float | jax.Array,
    process_weight_index: int,
    keys: jax.Array,
    alpha_cooling: float = 1.0,
    thresh: float = 0.0,
    n_monitors: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Optimize parameters via DPOP differentiable particle filter gradient training.

    Pure-functional implementation intended for users who need to compose
    the algorithm within custom JAX loops or higher-order functions.

    This function is analogous to :func:`pypomp.functional.train` as an
    optimization algorithm for for parameter estimation, but it can handle
    continuous states. It additionally incorporates a per-interval transition
    log-weight that is assumed to be stored in one of the state components.

    The process log-weight is expected to be accumulated over a single
    observation interval by the user-specified process model.  At the
    beginning of each interval, the corresponding state component should be
    reset to zero (this is naturally handled by ``accumvars``).

    .. warning::
       This function is experimental.  Its API and behavior are subject to change
       in future releases.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.
    thetas_array : jax.Array
        Initial parameter array of shape ``(n_reps, n_params)`` on the
        natural scale.
    J : int
        Number of particles.
    optimizer : Optimizer
        Optimizer configuration object (e.g. ``Adam()``, ``SGD()``).
    M : int
        Number of gradient steps.
    eta : LearningRate
        Per-parameter learning rates as a :class:`~pypomp.LearningRate` instance.
    alpha : float or jax.Array
        DPOP discount / cooling factor.
    process_weight_index : int
        Index of the process weight state component.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    alpha_cooling : float, optional
        Cosine cooling multiplier for ``alpha``. Defaults to ``1.0``.
    thresh : float, optional
        Resampling threshold. Defaults to ``0.0``.
    n_monitors : int, optional
        Number of monitors. Defaults to ``1``.

    Returns
    -------
    tuple of (jax.Array, jax.Array)
        - Negative log-likelihood history of shape ``(n_reps, M + 1)``.
        - Parameter trace history of shape ``(n_reps, M + 1, n_params)``.
    """
    warnings.warn(
        "dpop_train is experimental and its API and behavior are subject to change.",
        category=FutureWarning,
        stacklevel=2,
    )

    eta_array = eta.to_array(struct.param_names, M)

    context = DpopTrainContext.from_dpop_train_struct(
        struct,
        J=J,
        M=M,
        alpha_cooling=alpha_cooling,
        thresh=thresh,
        n_monitors=n_monitors,
        eta=eta_array,
        alpha=alpha,
        process_weight_index=process_weight_index,
    )

    return _vmapped_dpop_train_internal(
        thetas_array,
        keys,
        context,
        optimizer,
    )

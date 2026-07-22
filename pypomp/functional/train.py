import jax
import jax.numpy as jnp
from .structs import PompStruct, PanelPompStruct
from ..core.algorithms.train import (
    _vmapped_train_internal,
    _vmapped_panel_train_internal,
)
from ..core.algorithms.types import (
    PanelTrainConfig,
    PanelTrainInputs,
    TrainConfig,
    TrainInputs,
)
from ..core.learning_rate import LearningRate
from ..core.optimizer import Optimizer


def train(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    optimizer: Optimizer,
    M: int,
    eta: LearningRate,
    alpha: float | jax.Array,
    keys: jax.Array,
    alpha_cooling: float = 1.0,
    thresh: float = 0.0,
    n_monitors: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Optimize parameters via a differentiable particle filter (MOP).

    Performs Maximum Likelihood Estimation using the Measurement Off-Parameter (MOP) particle filter (Tan et al. 2024 [1]_), treating the particle filter
    as a differentiable computation graph and applies gradient-based
    optimizers (e.g. Adam, SGD, Newton) via JAX reverse-mode
    automatic differentiation.

    Pure-functional implementation intended for users who need to compose
    the algorithm within custom JAX loops or higher-order functions.
    For the standard interface, see :meth:`pypomp.Pomp.train`.

    JAX vectorizes the computation across all starting parameter sets
    simultaneously.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.  Obtain via
        :meth:`~pypomp.Pomp.to_struct`.
    thetas_array : jax.Array
        Initial parameter array of shape ``(n_reps, n_params)`` on the
        natural scale.  Must be aligned with ``struct.param_names``.
    J : int
        Number of particles.
    optimizer : Optimizer
        Optimizer configuration object (e.g. :class:`~pypomp.Adam`,
        :class:`~pypomp.SGD`).
    M : int
        Maximum number of gradient steps.
    eta : LearningRate
        Per-parameter learning rates as a :class:`~pypomp.LearningRate` instance.
    alpha : float or jax.Array
        MOP discount factor.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    alpha_cooling : float, optional
        Cosine cooling multiplier for ``alpha``.  Defaults to ``1.0``.
    thresh : float, optional
        ESS-based resampling threshold.  Defaults to ``0.0``.
    n_monitors : int, optional
        Number of unperturbed filter runs for log-likelihood monitoring.
        Defaults to ``1``.

    Returns
    -------
    tuple of (jax.Array, jax.Array)
        - Negative log-likelihood history of shape ``(n_reps, M)``.
        - Parameter trace history of shape ``(n_reps, M+1, n_params)``.

    Notes
    -----
    To align and stack input parameter dictionaries into the correct
    canonical ordering, use :func:`pypomp.functional.align_params`.

    See Also
    --------
    pypomp.Pomp.train : Object-oriented interface.
    align_params : Parameter alignment utility.

    References
    ----------
    .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
       for Partially Observed Markov Processes using Automatic Differentiation."
       *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.
    """
    eta_array = eta.to_array(struct.param_names, M)

    config = TrainConfig.from_train_struct(
        struct, J, M, alpha_cooling, thresh, n_monitors
    )
    inputs = TrainInputs.from_train_struct(struct, eta_array, alpha)

    return _vmapped_train_internal(
        thetas_array,
        keys,
        config,
        inputs,
        optimizer,
    )


def panel_train(
    struct: PanelPompStruct,
    shared_array: jax.Array,  # (n_reps, n_shared) on natural scale
    unit_array: jax.Array,  # (n_reps, U, n_spec) on natural scale
    J: int,
    optimizer: Optimizer,
    M: int,
    eta: LearningRate,
    alpha: float,
    keys: jax.Array,
    alpha_cooling: float,
    chunk_size: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Optimize panel POMP parameters via a differentiable particle filter (MOP).

    This function performs Maximum Likelihood Estimation (MLE) for Panel POMP
    models by treating the particle filter as a differentiable computational
    graph (Tan et al. 2024 [1]_).  It computes gradients of the log-likelihood
    with respect to parameters across units, and updates them using an optimizer (e.g. Adam, SGD).

    A pure functional implementation of the optimization (gradient-descent)
    algorithm, intended for composition within custom JAX code.

    Parameters
    ----------
    struct : PanelPompStruct
        Compiled structural representation of the Panel POMP model.
    shared_array : jax.Array
        Array of initial shared parameters of shape ``(n_reps, n_shared)`` on
        the natural scale.
    unit_array : jax.Array
        Array of initial unit-specific parameters of shape ``(n_reps, U, n_spec)``
        on the natural scale.
    J : int
        Number of particles.
    optimizer : Optimizer
        Optimizer configuration object (e.g. :class:`~pypomp.Adam`,
        :class:`~pypomp.SGD`, :class:`~pypomp.Newton`).
    M : int
        Number of iterations.
    eta : LearningRate
        Learning rates object containing rates for shared and unit-specific parameters.
    alpha : float
        Discount factor for MOP updates.
    keys : jax.Array
        Random keys of shape ``(n_reps, M, U, ...)``.
    alpha_cooling : float
        Cooling factor for discount factor alpha.
    chunk_size : int, optional
        Number of units to process per gradient step.  Defaults to ``1``.

    Returns
    -------
    logliks_history : jax.Array
        Average negative log-likelihood trace across iterations of shape
        ``(n_reps, M + 1)``.
    shared_history_natural : jax.Array
        Shared parameter history trace of shape ``(n_reps, M + 1, n_shared)`` on the
        natural scale.
    unit_history_natural : jax.Array
        Unit-specific parameter history trace of shape
        ``(n_reps, M + 1, U, n_spec)`` on the natural scale.

    Notes
    -----
    To align and stack input parameter arrays into the correct canonical
    ordering, use :func:`pypomp.functional.align_params`.

    See Also
    --------
    pypomp.PanelPomp.train : Object-oriented interface.
    align_params : Parameter alignment utility.

    References
    ----------
    .. [1] Tan, Kevin, Giles Hooker, and Edward L. Ionides. "Accelerated Inference
       for Partially Observed Markov Processes using Automatic Differentiation."
       *arXiv preprint arXiv:2407.03085* (2024). https://arxiv.org/abs/2407.03085.
    """
    eta_shared = (
        eta.to_array(struct.shared_param_names, M)
        if struct.shared_param_names
        else jnp.zeros((M, 0))
    )
    eta_spec = (
        eta.to_array(struct.unit_param_names, M)
        if struct.unit_param_names
        else jnp.zeros((M, 0))
    )

    shared_est, unit_est = struct.par_trans._transform_panel_array(
        shared_array,
        unit_array,
        struct.shared_param_names,
        struct.unit_param_names,
        direction="to_est",
    )

    config = PanelTrainConfig.from_panel_train_struct(
        struct, J, chunk_size, M, alpha_cooling
    )
    inputs = PanelTrainInputs.from_panel_train_struct(
        struct, keys, eta_shared, eta_spec, alpha
    )

    (
        logliks_history,
        shared_history,
        unit_history,
    ) = _vmapped_panel_train_internal(
        shared_est,
        unit_est,
        config,
        inputs,
        optimizer,
    )

    shared_history_natural, unit_history_natural = (
        struct.par_trans._transform_panel_array(
            shared_history,
            unit_history,
            struct.shared_param_names,
            struct.unit_param_names,
            direction="from_est",
        )
    )

    return logliks_history, shared_history_natural, unit_history_natural

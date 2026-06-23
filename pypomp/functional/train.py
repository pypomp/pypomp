import jax
from .structs import PompStruct, PanelPompStruct
from ..core.algorithms.train import (
    _vmapped_train_internal,
    _vmapped_panel_train_internal,
)
from ..core.optimizer import Optimizer


def train(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    optimizer: Optimizer,
    M: int,
    eta: jax.Array,
    thresh: float,
    alpha: float | jax.Array,
    keys: jax.Array,
    alpha_cooling: float,
    n_monitors: int,
) -> tuple[jax.Array, jax.Array]:
    """
    This is a pure functional implementation of the optimization algorithm, intended
    for users who need to compose it within custom JAX loops or higher-order
    functions. For a more user-friendly (but impurely-functional) interface, see
    :meth:`pypomp.core.pomp.Pomp.train`.

    This function performs Maximum Likelihood Estimation (MLE) by treating the particle filter
    as a differentiable computational graph. It computes gradients of the log-likelihood
    with respect to the parameters via reverse-mode automatic differentiation (using JAX),
    and updates the parameters using optimizers (e.g., Adam, SGD).

    This implementation leverages JAX to efficiently vectorize the algorithm across
    multiple initial parameter sets simultaneously.

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, n_params).
            Must be aligned with the canonical order of `struct.param_names` (e.g. prepared via `align_params`).
        J (int): Number of particles.
        optimizer (Optimizer): Optimizer configuration object.
        M (int): Number of iterations.
        eta (jax.Array): Learning rates array. Shape (M, n_params).
            Must be aligned with the canonical order of `struct.param_names` along the last axis.
        thresh (float): Resampling threshold.
        alpha (float | jax.Array): Alpha parameter.
        keys (jax.Array): Random keys. Shape (n_reps, ...).
        alpha_cooling (float): Alpha cooling factor.
        n_monitors (int): Number of monitors.

    Returns:
        tuple[jax.Array, jax.Array]:
            Negative logLik history: Shape (n_reps, M)
            Theta history: Shape (n_reps, M+1, n_params)

    Note:
        To align and stack input parameter dictionaries/scalars into the correct canonical ordering required by
        these arrays, use :func:`pypomp.functional.align_params`.
    """

    opt_name = optimizer.__class__.__name__
    clip_norm = optimizer.clip_norm
    beta1 = getattr(optimizer, "beta1", 0.9)
    beta2 = getattr(optimizer, "beta2", 0.999)
    epsilon = getattr(optimizer, "epsilon", 1e-8 if opt_name == "Adam" else 1e-4)
    c = optimizer.c
    max_ls_itn = optimizer.max_ls_itn
    scale = optimizer.scale
    ls = optimizer.ls

    return _vmapped_train_internal(
        thetas_array,
        struct.ys,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.accumvars,
        struct.covars_extended,
        J,
        opt_name,
        M,
        eta,
        c,
        max_ls_itn,
        thresh,
        scale,
        ls,
        alpha,
        keys,
        alpha_cooling,
        n_monitors,
        clip_norm,
        beta1,
        beta2,
        epsilon,
    )


def panel_train(
    struct: PanelPompStruct,
    shared_array: jax.Array,  # (n_reps, n_shared) on natural scale
    unit_array: jax.Array,  # (n_reps, U, n_spec) on natural scale
    J: int,
    optimizer: Optimizer,
    M: int,
    eta_shared: jax.Array,  # (M, n_shared)
    eta_spec: jax.Array,  # (M, n_spec)
    alpha: float,
    keys: jax.Array,
    alpha_cooling: float,
    chunk_size: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Pure functional implementation of the optimization (gradient-descent) algorithm,
    intended for users who need to compose it within custom JAX code.

    This function performs Maximum Likelihood Estimation (MLE) for Panel POMP models
    by treating the particle filter as a differentiable computational graph. It computes gradients
    of the log-likelihood with respect to parameters across units, and updates them using
    an optimizer (e.g. Adam, SGD).

    Args:
        struct (PanelPompStruct): The compiled structural representation of the Panel POMP model.
        shared_array (jax.Array): Array of initial shared parameters on natural scale.
            Shape (n_reps, n_shared). Must be aligned with the canonical order of `struct.shared_param_names` (e.g. prepared via `align_params`).
        unit_array (jax.Array): Array of initial unit-specific parameters on natural scale.
            Shape (n_reps, U, n_spec). Must be aligned with the canonical order of `struct.unit_param_names` (e.g. prepared via `align_params`).
        J (int): Number of particles.
        optimizer (Optimizer): Optimizer configuration object (e.g. Adam, SGD, FullMatrixAdam).
        M (int): Number of iterations.
        eta_shared (jax.Array): Learning rates array for shared parameters. Shape (M, n_shared).
            Must be aligned with the canonical order of `struct.shared_param_names` along the last axis.
        eta_spec (jax.Array): Learning rates array for unit-specific parameters. Shape (M, n_spec).
            Must be aligned with the canonical order of `struct.unit_param_names` along the last axis.
        alpha (float): Discount factor for MOP updates.
        keys (jax.Array): Random keys. Shape (n_reps, M, U, ...).
        alpha_cooling (float): Cooling factor for discount factor alpha.
        chunk_size (int, optional): Number of units to process per gradient step. Defaults to 1.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]:
            logliks_history: Average negative log-likelihood trace across iterations. Shape (n_reps, M + 1).
            shared_history_natural: Shared parameter history trace on natural scale. Shape (n_reps, M + 1, n_shared).
            unit_history_natural: Unit-specific parameter history trace on natural scale. Shape (n_reps, M + 1, U, n_spec).

    Note:
        To align and stack input parameter dictionaries/scalars into the correct canonical ordering required by
        these arrays, you can use :func:`pypomp.functional.align_params`.
    """

    U = len(struct.unit_names)

    shared_est, unit_est = struct.par_trans._transform_panel_array_jax(
        shared_array,
        unit_array,
        struct.shared_param_names,
        struct.unit_param_names,
        direction="to_est",
    )

    opt_name = optimizer.__class__.__name__
    clip_norm = optimizer.clip_norm
    beta1 = getattr(optimizer, "beta1", 0.9)
    beta2 = getattr(optimizer, "beta2", 0.999)
    epsilon = getattr(optimizer, "epsilon", 1e-8 if opt_name == "Adam" else 1e-4)
    scale = optimizer.scale

    (
        logliks_history,
        shared_history,
        unit_history,
    ) = _vmapped_panel_train_internal(
        shared_est,
        unit_est,
        struct.unit_param_permutations,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys_per_unit,
        struct.covars_per_unit,
        keys,
        J,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.accumvars,
        chunk_size,
        opt_name,
        M,
        eta_shared,
        eta_spec,
        alpha,
        alpha_cooling,
        struct.ys_per_unit.shape[1],
        U,
        clip_norm,
        beta1,
        beta2,
        epsilon,
        scale,
    )

    shared_history_natural, unit_history_natural = (
        struct.par_trans._transform_panel_array_jax(
            shared_history,
            unit_history,
            struct.shared_param_names,
            struct.unit_param_names,
            direction="from_est",
        )
    )

    return logliks_history, shared_history_natural, unit_history_natural

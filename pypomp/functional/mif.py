import jax
import jax.numpy as jnp
from typing import Callable
from .structs import PompStruct, PanelPompStruct
from ..core.algorithms.mif import (
    _jv_mif_internal,
    _jv_panel_mif_internal,
)


def mif(
    struct: PompStruct,
    thetas_array: jax.Array,
    sigmas_array: jax.Array,
    sigmas_init_array: jax.Array,
    M: int,
    cooling_fn: Callable | float,
    J: int,
    thresh: float,
    keys: jax.Array,
    n_monitors: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    This is a pure functional implementation of the Iterated Filtering algorithm,
    intended for users who need to compose it within custom JAX loops or
    higher-order functions. For a more user-friendly (but impurely-functional) interface, see
    :meth:`pypomp.core.pomp.Pomp.mif`.

    This implementation leverages JAX to efficiently vectorize the algorithm across
    multiple initial parameter sets simultaneously.

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, J, n_params) on the natural scale.
            Must be aligned with the canonical order of `struct.param_names` (e.g. prepared via `align_params`).
        sigmas_array (jax.Array): Array of random walk sigmas. Shape (n_params,).
            Must be aligned with the canonical order of `struct.param_names`.
        sigmas_init_array (jax.Array): Array of initial random walk sigmas. Shape (n_params,).
            Must be aligned with the canonical order of `struct.param_names`.
        M (int): Number of iterations.
        cooling_fn (Callable | float): Cooling function taking (nt, m, ntimes) or float cooling factor.
        J (int): Number of particles.
        thresh (float): Resampling threshold.
        keys (jax.Array): Random keys. Shape (n_reps, ...).
        n_monitors (int): Number of monitors for likelihood averaging.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]:
            Negative log-likelihood history: Shape (n_reps, M).
            Parameter trace history: Shape (n_reps, M+1, n_params) on the natural scale.
            Final particle swarm: Shape (n_reps, J, n_params) on the natural scale.

    Note:
        To align and stack input parameter dictionaries/scalars into the correct canonical ordering required by
        these arrays, use :func:`pypomp.functional.align_params`.
    """

    thetas_est = struct.par_trans._transform_array_jax(
        thetas_array,
        struct.param_names,
        direction="to_est",
    )

    res = _jv_mif_internal(
        thetas_est,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys,
        struct.rinit_per,
        struct.rproc_per,
        struct.dmeas_per,
        sigmas_array,
        sigmas_init_array,
        struct.accumvars,
        struct.covars_extended,
        M,
        cooling_fn,
        0,
        J,
        thresh,
        keys,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        n_monitors,
        False,
    )
    traces_natural = struct.par_trans._transform_array_jax(
        res[1],
        struct.param_names,
        direction="from_est",
    )
    final_thetas_natural = struct.par_trans._transform_array_jax(
        res[2],
        struct.param_names,
        direction="from_est",
    )
    return res[0], traces_natural, final_thetas_natural


def panel_mif(
    struct: PanelPompStruct,
    shared_array: jax.Array,  # (n_reps, J, n_shared) on natural scale
    unit_array: jax.Array,  # (n_reps, J, U, n_spec) on natural scale
    sigmas_array: jax.Array,  # (n_params,)
    sigmas_init_array: jax.Array,  # (n_params,)
    M: int,
    cooling_fn: Callable | float,
    J: int,
    thresh: float,
    keys: jax.Array,
    n_monitors: int = 0,
    block: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Pure functional implementation of the (Marginal) Panel Iterated Filtering (PIF/MPIF) algorithm,
    intended for users who need to compose it within custom JAX loops.

    This function estimates parameters for a Panel POMP model by introducing random perturbations
    to the parameters and sequentially filtering them across all units. The perturbation variance
    is decayed according to a cooling schedule.

    Args:
        struct (PanelPompStruct): The compiled structural representation of the Panel POMP model.
        shared_array (jax.Array): Swarm of initial shared parameters on natural scale.
            Shape (n_reps, J, n_shared). Must be aligned with the canonical order of `struct.shared_param_names` (e.g. prepared via `align_params`).
        unit_array (jax.Array): Swarm of initial unit-specific parameters on natural scale.
            Shape (n_reps, J, U, n_spec). Must be aligned with the canonical order of `struct.unit_param_names` (e.g. prepared via `align_params`).
        sigmas_array (jax.Array): Random walk standard deviations. Shape (n_params,).
            Must be aligned with the canonical order of `struct.param_names` (e.g. prepared via `align_params`).
        sigmas_init_array (jax.Array): Initial random walk standard deviations. Shape (n_params,).
            Must be aligned with the canonical order of `struct.param_names` (e.g. prepared via `align_params`).
        M (int): Number of iterated filtering iterations.
        cooling_fn (Callable | float): Cooling schedule function or constant decay factor.
        J (int): Number of particles.
        thresh (float): Resampling threshold.
        keys (jax.Array): Random keys. Shape (n_reps, ...).
        n_monitors (int, optional): Number of monitor runs to perform at each iteration. Defaults to 0.
        block (bool, optional): Whether to use MPIF. Defaults to True.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            shared_traces: Shared parameter history trace. Shape (n_reps, M + 1, n_shared + 1).
            unit_traces: Unit-specific parameter history trace. Shape (n_reps, M + 1, U, n_spec + 1).
            final_shared_swarm: Final swarm of shared parameters. Shape (n_reps, J, n_shared).
            final_unit_swarm: Final swarm of unit-specific parameters. Shape (n_reps, J, U, n_spec).

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

    shared_array_f, unit_array_f, shared_traces, unit_traces = _jv_panel_mif_internal(
        shared_est,
        unit_est,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys_per_unit,
        struct.rinit_per,
        struct.rproc_per,
        struct.dmeas_per,
        sigmas_array,
        sigmas_init_array,
        struct.accumvars,
        struct.covars_per_unit,
        struct.unit_param_permutations,
        M,
        cooling_fn,
        J,
        U,
        thresh,
        keys,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        n_monitors,
        block,
    )

    n_shared = len(struct.shared_param_names)
    n_spec = len(struct.unit_param_names)

    shared_traces_natural = shared_traces
    unit_traces_natural = unit_traces

    if n_shared > 0 or n_spec > 0:
        # Extract shared parameter traces (slice off log-likelihood)
        shared_params = (
            shared_traces[:, :, 1:]
            if n_shared > 0
            else jnp.zeros((shared_traces.shape[0], shared_traces.shape[1], 0))
        )

        unit_params = (
            unit_traces[:, :, :, 1:]
            if n_spec > 0
            else jnp.zeros((unit_traces.shape[0], unit_traces.shape[1], U, 0))
        )

        shared_transformed, unit_transformed = (
            struct.par_trans._transform_panel_array_jax(
                shared_params,
                unit_params,
                struct.shared_param_names,
                struct.unit_param_names,
                direction="from_est",
            )
        )

        if n_shared > 0:
            shared_traces_natural = jnp.concatenate(
                [shared_traces[:, :, :1], shared_transformed], axis=-1
            )
        if n_spec > 0:
            unit_traces_natural = jnp.concatenate(
                [unit_traces[:, :, :, :1], unit_transformed], axis=-1
            )

    final_shared_swarm_natural, final_unit_swarm_natural = (
        struct.par_trans._transform_panel_array_jax(
            shared_array_f,
            unit_array_f,
            struct.shared_param_names,
            struct.unit_param_names,
            direction="from_est",
        )
    )

    return (
        shared_traces_natural,
        unit_traces_natural,
        final_shared_swarm_natural,
        final_unit_swarm_natural,
    )

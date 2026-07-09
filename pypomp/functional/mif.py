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
    keys: jax.Array,
    thresh: float = 0.0,
    n_monitors: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the Iterated Filtering 2 (IF2) algorithm on a POMP model struct.

    Pure-functional implementation intended for users who need to compose
    the algorithm within custom JAX loops or higher-order functions.
    For the standard interface, see :meth:`pypomp.Pomp.mif`.

    JAX vectorises the computation across all starting parameter sets
    simultaneously.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.  Obtain via
        :meth:`~pypomp.Pomp.to_struct`.
    thetas_array : jax.Array
        Initial parameter array of shape ``(n_reps, J, n_params)`` on the
        natural scale.  Must be aligned with ``struct.param_names``.
    sigmas_array : jax.Array
        Per-parameter random walk standard deviations.  Shape
        ``(n_params,)``.
    sigmas_init_array : jax.Array
        Initial random walk standard deviations.  Shape ``(n_params,)``.
    M : int
        Number of IF2 iterations.
    cooling_fn : callable or float
        Cooling schedule.  Pass a callable ``(nt, m, ntimes) -> float``
        for custom schedules, or a single float for geometric cooling.
    J : int
        Number of particles.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    thresh : float, optional
        ESS-based resampling threshold.  Defaults to ``0.0``.
    n_monitors : int, optional
        Number of unperturbed filter runs for log-likelihood monitoring.
        Defaults to ``0``.

    Returns
    -------
    tuple of (jax.Array, jax.Array, jax.Array)
        - Negative log-likelihood history of shape ``(n_reps, M)``.
        - Parameter trace history of shape ``(n_reps, M+1, n_params)``
          on the natural scale.
        - Final particle swarm of shape ``(n_reps, J, n_params)`` on the
          natural scale.

    Notes
    -----
    To align and stack input parameter dictionaries into the correct
    canonical ordering, use :func:`pypomp.functional.align_params`.

    See Also
    --------
    pypomp.Pomp.mif : Object-oriented interface.
    align_params : Parameter alignment utility.
    """

    thresh = float(max(0.0, thresh))
    thetas_est = struct.par_trans._transform_array(
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
        J,
        thresh,
        keys,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        n_monitors,
        False,
    )
    traces_natural = struct.par_trans._transform_array(
        res[1],
        struct.param_names,
        direction="from_est",
    )
    final_thetas_natural = struct.par_trans._transform_array(
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
    keys: jax.Array,
    thresh: float = 0.0,
    n_monitors: int = 0,
    block: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Estimate panel POMP parameters using Panel Iterated Filtering.

    A pure functional implementation of the (Marginal) Panel Iterated
    Filtering (PIF/MPIF) algorithm, intended for composition within custom JAX
    loops.

    This function estimates parameters for a Panel POMP model by introducing
    random perturbations to the parameters and sequentially filtering them
    across all units.  The perturbation variance is decayed according to a
    given cooling schedule.

    Parameters
    ----------
    struct : PanelPompStruct
        Compiled structural representation of the Panel POMP model.
    shared_array : jax.Array
        Swarm of initial shared parameters of shape ``(n_reps, J, n_shared)``
        on the natural scale.
    unit_array : jax.Array
        Swarm of initial unit-specific parameters of shape
        ``(n_reps, J, U, n_spec)`` on the natural scale.
    sigmas_array : jax.Array
        Random walk standard deviations of shape ``(n_params,)``.
    sigmas_init_array : jax.Array
        Initial random walk standard deviations of shape ``(n_params,)``.
    M : int
        Number of iterated filtering iterations.
    cooling_fn : callable or float
        Cooling schedule function or constant decay factor.
    J : int
        Number of particles.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    thresh : float, optional
        Resampling threshold.  Defaults to ``0.0``.
    n_monitors : int, optional
        Number of monitor runs to perform at each iteration.  Defaults to
        ``0``.
    block : bool, optional
        Whether to use block updates (MPIF).  Defaults to ``True``.

    Returns
    -------
    shared_traces : jax.Array
        Shared parameter history trace of shape ``(n_reps, M + 1, n_shared + 1)``.
    unit_traces : jax.Array
        Unit-specific parameter history trace of shape
        ``(n_reps, M + 1, U, n_spec + 1)``.
    final_shared_swarm : jax.Array
        Final swarm of shared parameters of shape ``(n_reps, J, n_shared)``.
    final_unit_swarm : jax.Array
        Final swarm of unit-specific parameters of shape ``(n_reps, J, U, n_spec)``.

    Notes
    -----
    To align and stack input parameter arrays into the correct canonical
    ordering, use :func:`pypomp.functional.align_params`.
    """

    thresh = float(max(0.0, thresh))
    U = len(struct.unit_names)

    shared_est, unit_est = struct.par_trans._transform_panel_array(
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

        shared_transformed, unit_transformed = struct.par_trans._transform_panel_array(
            shared_params,
            unit_params,
            struct.shared_param_names,
            struct.unit_param_names,
            direction="from_est",
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
        struct.par_trans._transform_panel_array(
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

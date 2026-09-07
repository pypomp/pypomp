import jax
import jax.numpy as jnp

from ..core.algorithms.contexts import MifContext, PanelMifContext
from ..core.algorithms.mif import (
    _jv_mif_internal,
)
from ..core.algorithms.panel_mif import (
    _jv_panel_mif_internal,
)
from ..core.rw_sigma import RWSigma
from .structs import PanelPompStruct, PompStruct


def mif(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    M: int,
    rw_sd: RWSigma,
    keys: jax.Array,
    thresh: float = 0.0,
    n_monitors: int = 0,
    return_swarm: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the Iterated Filtering 2 (IF2) algorithm on a POMP model struct.

    Pure-functional implementation of the Iterated Filtering 2 (IF2) algorithm
    (Ionides et al. 2015 [1]_), intended for users who need to compose the algorithm
    within custom JAX loops or higher-order functions.
    For the standard interface, see :meth:`pypomp.Pomp.mif`.

    JAX vectorises the computation across all starting parameter sets
    simultaneously.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.  Obtain via
        :meth:`~pypomp.Pomp.to_struct`.
    thetas_array : jax.Array
        Initial parameter array of shape ``(n_reps, n_params)`` or
        ``(n_reps, J, n_params)`` on the natural scale.  Must be aligned with
        ``struct.param_names``.
    J : int
        Number of particles.
    M : int
        Number of IF2 iterations.
    rw_sd : RWSigma
        Random-walk standard deviations and cooling schedule.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    thresh : float, optional
        ESS-based resampling threshold.  Defaults to ``0.0``.
    n_monitors : int, optional
        Number of unperturbed filter runs for log-likelihood monitoring.
        Defaults to ``0``.
    return_swarm : bool, optional
        Whether to transform and return the final particle swarm.  Defaults to
        ``True``.

    Returns
    -------
    tuple of (jax.Array, jax.Array, jax.Array)
        - Log-likelihood history of shape ``(n_reps, M)``.
        - Parameter trace history of shape ``(n_reps, M+1, n_params)``
        - on the natural scale.
        - Final particle swarm of shape ``(n_reps, J, n_params)`` on the
          natural scale (or zero-sized if ``return_swarm=False``).

    Notes
    -----
    To align and stack input parameter dictionaries into the correct
    canonical ordering, use :func:`pypomp.functional.align_params`.

    See Also
    --------
    pypomp.Pomp.mif : Object-oriented interface.
    align_params : Parameter alignment utility.

    References
    ----------
    .. [1] Ionides, Edward L., Dao Nguyen, Yves Atchadé, Stilian Stoev, and Aaron A. King.
       "Inference for dynamic and latent variable models via iterated, perturbed Bayes maps."
       *Proceedings of the National Academy of Sciences* 112, no. 3 (2015): 719–724.
       https://doi.org/10.1073/pnas.1410597112.
    """

    thresh = float(max(0.0, thresh))
    if thetas_array.ndim == 2 or (
        thetas_array.ndim == 3 and thetas_array.shape[1] == 1
    ):
        thetas_untiled = (
            thetas_array if thetas_array.ndim == 2 else thetas_array[:, 0, :]
        )
        thetas_est_untiled = struct.par_trans._transform_array(
            thetas_untiled,
            struct.param_names,
            direction="to_est",
        )
        thetas_est = jnp.repeat(thetas_est_untiled[:, jnp.newaxis, :], J, axis=1)
    else:
        thetas_est = struct.par_trans._transform_array(
            thetas_array,
            struct.param_names,
            direction="to_est",
        )

    if struct.dmeas_per is None:
        raise ValueError("dmeasure is required for MIF")
    if struct.dmeas_pf is None:
        raise ValueError("dmeasure_pf is required for MIF")

    rw_sd = rw_sd._canonicalize(struct.param_names)

    context = MifContext.from_struct(
        struct=struct,
        rw_sigma=rw_sd,
        J=J,
        M=M,
        thresh=thresh,
        n_monitors=n_monitors,
        return_ancestry=False,
    )
    res = _jv_mif_internal(
        thetas_est,
        keys,
        context,
    )
    traces_natural = struct.par_trans._transform_array(
        res[1],
        struct.param_names,
        direction="from_est",
    )
    if return_swarm:
        final_thetas_natural = struct.par_trans._transform_array(
            res[2],
            struct.param_names,
            direction="from_est",
        )
    else:
        final_thetas_natural = jnp.zeros(
            (thetas_array.shape[0], 0, len(struct.param_names))
        )
    return -res[0], traces_natural, final_thetas_natural


def panel_mif(
    struct: PanelPompStruct,
    shared_array: jax.Array,  # (n_reps, J, n_shared) or (n_reps, n_shared) on natural scale
    unit_array: jax.Array,  # (n_reps, J, U, n_spec) or (n_reps, U, n_spec) on natural scale
    J: int,
    M: int,
    rw_sd: RWSigma,
    keys: jax.Array,
    thresh: float = 0.0,
    block: bool = True,
    n_monitors: int = 0,
    return_swarm: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Estimate panel POMP parameters using Panel Iterated Filtering.

    A pure functional implementation of the (Marginal) Panel Iterated
    Filtering (PIF/MPIF) algorithm (Bretó et al. 2020 [1]_; Wheeler et al. 2025 [2]_), intended for composition within custom JAX
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
        Initial shared parameters of shape ``(n_reps, n_shared)`` or swarm of
        shape ``(n_reps, J, n_shared)`` on the natural scale.
    unit_array : jax.Array
        Initial unit-specific parameters of shape ``(n_reps, U, n_spec)`` or
        swarm of shape ``(n_reps, J, U, n_spec)`` on the natural scale.
    J : int
        Number of particles.
    M : int
        Number of iterated filtering iterations.
    rw_sd : RWSigma
        Random-walk standard deviations and cooling schedule.  Reordered
        internally to ``struct.param_names`` order, so any order may be passed.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    thresh : float, optional
        Resampling threshold.  Defaults to ``0.0``.
    block : bool, optional
        Whether to use block updates (MPIF).  Defaults to ``True``.
    n_monitors : int, optional
        Number of monitor runs to perform at each iteration.  Defaults to
        ``0``.
    return_swarm : bool, optional
        Whether to transform and return the final swarms.  Defaults to ``True``.

    Returns
    -------
    shared_traces : jax.Array
        Shared parameter history trace of shape ``(n_reps, M + 1, n_shared + 1)``.
    unit_traces : jax.Array
        Unit-specific parameter history trace of shape
        ``(n_reps, M + 1, U, n_spec + 1)``.
    final_shared_swarm : jax.Array
        Final swarm of shared parameters of shape ``(n_reps, J, n_shared)``
        (or zero-sized if ``return_swarm=False``).
    final_unit_swarm : jax.Array
        Final swarm of unit-specific parameters of shape ``(n_reps, J, U, n_spec)``
        (or zero-sized if ``return_swarm=False``).

    Notes
    -----
    To align and stack input parameter arrays into the correct canonical
    ordering, use :func:`pypomp.functional.align_params`.

    See Also
    --------
    pypomp.PanelPomp.mif : Object-oriented interface.
    align_params : Parameter alignment utility.

    References
    ----------
    .. [1] Bretó, Carles, Edward L. Ionides, and Aaron A. King. "Panel Data Analysis
       via Mechanistic Models." *Journal of the American Statistical Association*
       115, no. 531 (2020): 1178–1188. https://doi.org/10.1080/01621459.2019.1604367.
    .. [2] Wheeler, Jesse, Aaron J. Abkemeier, and Edward L. Ionides. "Iterating
       marginalized Bayes maps for likelihood maximization with application to nonlinear
       panel models." *arXiv preprint arXiv:2511.17438* (2025). https://arxiv.org/abs/2511.17438.
    """

    thresh = float(max(0.0, thresh))
    U = len(struct.unit_names)
    n_shared = len(struct.shared_param_names)
    n_spec = len(struct.unit_param_names)

    is_untiled = (
        shared_array.ndim == 2
        or (shared_array.ndim == 3 and shared_array.shape[1] == 1)
        or unit_array.ndim == 3
        or (unit_array.ndim == 4 and unit_array.shape[1] == 1)
    )

    if is_untiled:
        shared_untiled = (
            shared_array if shared_array.ndim == 2 else shared_array[:, 0, :]
        )
        unit_untiled = unit_array if unit_array.ndim == 3 else unit_array[:, 0, :, :]
        shared_est_untiled, unit_est_untiled = struct.par_trans._transform_panel_array(
            shared_untiled,
            unit_untiled,
            struct.shared_param_names,
            struct.unit_param_names,
            direction="to_est",
        )
        shared_est = (
            jnp.repeat(shared_est_untiled[:, jnp.newaxis, :], J, axis=1)
            if (shared_est_untiled is not None and n_shared > 0)
            else jnp.zeros((keys.shape[0], J, 0))
        )
        unit_est = (
            jnp.repeat(unit_est_untiled[:, jnp.newaxis, :, :], J, axis=1)
            if (unit_est_untiled is not None and n_spec > 0)
            else jnp.zeros((keys.shape[0], J, U, 0))
        )
    else:
        shared_est, unit_est = struct.par_trans._transform_panel_array(
            shared_array,
            unit_array,
            struct.shared_param_names,
            struct.unit_param_names,
            direction="to_est",
        )

    if struct.dmeas_per is None:
        raise ValueError("dmeasure is required for Panel MIF")
    if struct.dmeas_pf is None:
        raise ValueError("dmeasure_pf is required for Panel MIF")

    rw_sd = rw_sd._canonicalize(struct.param_names)

    context = PanelMifContext.from_struct(
        struct=struct,
        rw_sigma=rw_sd,
        J=J,
        M=M,
        U=U,
        thresh=thresh,
        n_monitors=n_monitors,
        block=block,
    )
    shared_array_f, unit_array_f, shared_traces, unit_traces = _jv_panel_mif_internal(
        shared_est,
        unit_est,
        keys,
        context,
    )

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

    if return_swarm:
        final_shared_swarm_natural, final_unit_swarm_natural = (
            struct.par_trans._transform_panel_array(
                shared_array_f,
                unit_array_f,
                struct.shared_param_names,
                struct.unit_param_names,
                direction="from_est",
            )
        )
    else:
        final_shared_swarm_natural = jnp.zeros((shared_traces.shape[0], 0, n_shared))
        final_unit_swarm_natural = jnp.zeros((unit_traces.shape[0], 0, U, n_spec))

    return (
        shared_traces_natural,
        unit_traces_natural,
        final_shared_swarm_natural,
        final_unit_swarm_natural,
    )

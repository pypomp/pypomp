import jax
from .structs import PompStruct, PanelPompStruct
from ..core.algorithms.pfilter import (
    _vmapped_pfilter_internal2,
    _chunked_panel_pfilter_internal,
)


def pfilter(
    struct: PompStruct,
    thetas_array: jax.Array,
    J: int,
    keys: jax.Array,
    thresh: float = 0.0,
    CLL: bool = False,
    ESS: bool = False,
    filter_mean: bool = False,
    prediction_mean: bool = False,
) -> dict[str, jax.Array]:
    """Run the bootstrap particle filter on a POMP model struct.

    Pure-functional implementation intended for users who need to compose
    the particle filter within custom JAX loops or higher-order functions.
    For the standard interface, see :meth:`pypomp.Pomp.pfilter`.

    JAX vectorises the computation across all parameter sets in
    ``thetas_array`` simultaneously.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.  Obtain via
        :meth:`~pypomp.Pomp.to_struct`.
    thetas_array : jax.Array
        Parameter array of shape ``(n_reps, n_params)`` on the natural
        scale.  Must be aligned with ``struct.param_names`` (e.g. via
        :func:`align_params`).
    J : int
        Number of particles.
    keys : jax.Array
        Random keys of shape ``(n_reps, reps, ...)``.
    thresh : float, optional
        ESS-based resampling threshold.  Defaults to ``0.0``.
    CLL : bool, optional
        Compute conditional log-likelihoods.  Defaults to ``False``.
    ESS : bool, optional
        Compute effective sample size.  Defaults to ``False``.
    filter_mean : bool, optional
        Compute filtered state means.  Defaults to ``False``.
    prediction_mean : bool, optional
        Compute predicted state means.  Defaults to ``False``.

    Returns
    -------
    dict of str to jax.Array
        Always contains ``'logLik'``.  Optionally contains ``'CLL'``,
        ``'ESS'``, ``'filter_mean'``, and ``'prediction_mean'`` if their
        corresponding flags are ``True``.

    Notes
    -----
    To align and stack input parameter arrays into the correct
    canonical ordering, use :func:`pypomp.functional.align_params`.

    See Also
    --------
    pypomp.Pomp.pfilter : Object-oriented interface.
    align_params : Parameter alignment utility.
    """

    thresh = float(max(0.0, thresh))
    results = _vmapped_pfilter_internal2(
        thetas_array,
        struct.dt_array_extended,
        struct.nstep_array,
        struct.t0,
        struct.times,
        struct.ys,
        J,
        struct.rinit_pf,
        struct.rproc_pf,
        struct.dmeas_pf,
        struct.accumvars,
        struct.covars_extended,
        thresh,
        keys,
        CLL,
        ESS,
        filter_mean,
        prediction_mean,
        False,
    )
    results["logLik"] = -results.pop("neg_loglik")
    return results


def panel_pfilter(
    struct: PanelPompStruct,
    thetas_array: jax.Array,
    J: int,
    keys: jax.Array,
    thresh: float = 0.0,
    chunk_size: int = 1,
    CLL: bool = False,
    ESS: bool = False,
    filter_mean: bool = False,
    prediction_mean: bool = False,
) -> dict[str, jax.Array]:
    """Evaluate panel POMP log-likelihood via particle filtering.

    A pure functional implementation of the panel particle filter, intended
    for composition within custom JAX loops.

    Parameters
    ----------
    struct : PanelPompStruct
        Compiled structural representation of the Panel POMP model.
    thetas_array : jax.Array
        Swarm of parameters of shape ``(n_reps, U, n_params)`` on the natural
        scale, aligned with the canonical order of ``struct.shared_param_names``
        and ``struct.unit_param_names`` per unit.
    J : int
        Number of particles.
    keys : jax.Array
        Random keys of shape ``(n_reps, U_padded, ...)``.
    thresh : float, optional
        Resampling threshold.  Defaults to ``0.0``.
    chunk_size : int, optional
        Number of units to process per chunk.  Defaults to ``1``.
    CLL : bool, optional
        Whether to compute conditional log-likelihoods.  Defaults to ``False``.
    ESS : bool, optional
        Whether to compute effective sample sizes.  Defaults to ``False``.
    filter_mean : bool, optional
        Whether to compute filtered state means.  Defaults to ``False``.
    prediction_mean : bool, optional
        Whether to compute prediction state means.  Defaults to ``False``.

    Returns
    -------
    dict of str to jax.Array
        A dictionary containing the results of the panel particle filter.
        Always contains ``'logLik'``.  Optionally contains ``'CLL'``,
        ``'ESS'``, ``'filter_mean'``, and ``'prediction_mean'`` if their
        corresponding flags are ``True``.

    Notes
    -----
    To align and stack input parameter arrays into the correct
    canonical ordering, use :func:`pypomp.functional.align_params`.
    """
    thresh = float(max(0.0, thresh))
    results = _chunked_panel_pfilter_internal(
        thetas_array,
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
        thresh,
        chunk_size,
        CLL,
        ESS,
        filter_mean,
        prediction_mean,
        False,
    )
    results["logLik"] = -results.pop("neg_loglik")
    return results

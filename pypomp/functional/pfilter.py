from dataclasses import replace

import jax

from ..core.algorithms.contexts import PfilterContext
from ..core.algorithms.pfilter import (
    _chunked_panel_pfilter_internal,
    _vmapped_pfilter_internal2,
)
from .structs import PanelPompStruct, PompStruct


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
    reps_batch_size: int | None = None,
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
        Resampling threshold. Particles are resampled at an observation time
        when the largest particle weight exceeds ``thresh`` times the smallest.
        Values below 1, including the default ``0.0``, resample at every
        observation time.
    CLL : bool, optional
        Compute conditional log-likelihoods.  Defaults to ``False``.
    ESS : bool, optional
        Compute effective sample size.  Defaults to ``False``.
    filter_mean : bool, optional
        Compute filtered state means.  Defaults to ``False``.
    prediction_mean : bool, optional
        Compute predicted state means.  Defaults to ``False``.
    reps_batch_size : int or None, optional
        How many replicate filters per parameter set to run at once; smaller
        batches run one after another and use less memory.  ``None`` (default)
        chooses: on a GPU, all replicates at once if XLA's memory estimate for
        that fits in 80% of the free device memory, otherwise the largest batch
        that fits; on CPU, one at a time, which is faster there.  Inside a
        caller's ``jax.jit`` the memory cannot be measured, so on a GPU all
        replicates run at once.

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
    context = PfilterContext.from_struct(
        struct,
        J=J,
        thresh=thresh,
        CLL=CLL,
        ESS=ESS,
        filter_mean=filter_mean,
        prediction_mean=prediction_mean,
        should_trans=False,
        reps_batch_size=reps_batch_size,
    )
    results = _run_reps(thetas_array, keys, context)
    results["logLik"] = -results.pop("neg_loglik")
    return results


# Share of free device memory the all-at-once filter may use before batching.
MEMORY_HEADROOM = 0.8


def reps_batch_for(peak_bytes: int, free_bytes: int, n_reps: int) -> int | None:
    """``None`` (all replicates at once) if they fit in the memory budget,
    otherwise the largest batch that does, assuming memory grows linearly with
    the batch size (which overstates a batch's need slightly)."""
    budget = MEMORY_HEADROOM * free_bytes
    if peak_bytes <= budget:
        return None
    return max(1, int(budget // (peak_bytes / n_reps)))


def _free_device_bytes(arrays) -> int | None:
    """Free allocator memory on the fullest device holding `arrays`, if known."""
    free = []
    for a in arrays:
        for d in a.devices():
            stats = d.memory_stats()
            if not stats or "bytes_limit" not in stats:
                return None
            free.append(stats["bytes_limit"] - stats.get("bytes_in_use", 0))
    return min(free) if free else None


def _peak_bytes(compiled) -> int | None:
    m = compiled.memory_analysis()
    if m is None:
        return None
    return (m.argument_size_in_bytes + m.output_size_in_bytes + m.temp_size_in_bytes
            - m.alias_size_in_bytes)


def _run_reps(thetas_array, keys, context):
    """Run the filters, choosing how many replicates to batch when not given."""
    if context.reps_batch_size is not None or keys.shape[1] == 1:
        return _vmapped_pfilter_internal2(thetas_array, keys, context)
    if jax.default_backend() == "cpu":
        return _vmapped_pfilter_internal2(thetas_array, keys, replace(context, reps_batch_size=1))
    if any(isinstance(x, jax.core.Tracer) for x in (thetas_array, keys)):
        return _vmapped_pfilter_internal2(thetas_array, keys, context)
    # Compile the all-at-once filter (needed anyway) and ask XLA how much memory it needs.
    compiled = jax.jit(_vmapped_pfilter_internal2).lower(thetas_array, keys, context).compile()
    peak, free = _peak_bytes(compiled), _free_device_bytes((thetas_array, keys))
    batch = None if peak is None or free is None else reps_batch_for(peak, free, keys.shape[1])
    if batch is None:
        return compiled(thetas_array, keys, context)
    return _vmapped_pfilter_internal2(thetas_array, keys, replace(context, reps_batch_size=batch))


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
        Resampling threshold. Particles are resampled at an observation time
        when the largest particle weight exceeds ``thresh`` times the smallest.
        Values below 1, including the default ``0.0``, resample at every
        observation time.
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

    See Also
    --------
    pypomp.PanelPomp.pfilter : Object-oriented interface.
    align_params : Parameter alignment utility.
    """
    thresh = float(max(0.0, thresh))
    context = PfilterContext.from_panel_struct(
        struct,
        J=J,
        thresh=thresh,
        CLL=CLL,
        ESS=ESS,
        filter_mean=filter_mean,
        prediction_mean=prediction_mean,
        should_trans=False,
    )
    results = _chunked_panel_pfilter_internal(
        thetas_array,
        keys,
        context,
        chunk_size,
    )
    results["logLik"] = -results.pop("neg_loglik")
    return results

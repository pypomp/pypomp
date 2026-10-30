from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jax.stages import Compiled

from ..core.algorithms.contexts import PfilterContext
from ..core.algorithms.helpers import pad_array
from ..core.algorithms.pfilter import (
    _chunked_panel_pfilter_internal,
    _pfilter_internal,
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
        chooses: on a GPU, each device runs batches of about 3 million particles
        across all parameter sets and replicates, fewer if XLA's memory estimate
        exceeds 80% of the free device memory; on CPU, one replicate at a time,
        which is faster there.  Inside a caller's ``jax.jit``, a GPU runs every
        filter at once.

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

    if reps_batch_size is not None and reps_batch_size < 1:
        raise ValueError("reps_batch_size should be None or greater than 0.")
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


# Particles each device runs at once.  Far fewer and launching kernels dominates;
# far more gains no speed, costs memory and slows some GPUs.
_PARTICLES_PER_BATCH = 3 * 2**20

# Share of free device memory that the filters may use.
_MEMORY_HEADROOM = 0.8


def _batch_for(
    fixed_bytes: int, temp_bytes: int, free_bytes: int, batch: int
) -> int | None:
    """Return the largest filter batch that fits in memory, or None if `batch` does."""
    budget = _MEMORY_HEADROOM * free_bytes
    if fixed_bytes + temp_bytes <= budget:
        return None
    # Inputs and outputs are full size at any batch size; the workspace scales.
    return max(1, int((budget - fixed_bytes) // (temp_bytes / batch)))


def _free_device_bytes(arrays: tuple[jax.Array, ...]) -> int | None:
    """Return the free memory on the fullest device holding `arrays`, if known."""
    free = []
    for a in arrays:
        for d in a.devices():
            stats = d.memory_stats()
            if not stats or "bytes_limit" not in stats:
                return None
            free.append(stats["bytes_limit"] - stats.get("bytes_in_use", 0))
    return min(free) if free else None


def _memory_bytes(compiled: Compiled) -> tuple[int, int] | None:
    """Return XLA's estimate of a compiled function's input and output memory and
    its workspace, if known."""
    m = compiled.memory_analysis()
    if m is None:
        return None
    fixed = m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes
    return fixed, m.temp_size_in_bytes


def _flat_reps(
    thetas_array: jax.Array, keys: jax.Array, context: PfilterContext
) -> dict[str, jax.Array]:
    """Run every replicate at once, with parameter sets and replicates in one batch."""
    # Nested inside the parameter sets' vmap, the replicates make XLA copy data on
    # every Euler step.
    n, reps = keys.shape[:2]
    flat_keys = keys.reshape(n * reps, 1, *keys.shape[2:])
    out = _vmapped_pfilter_internal2(
        jnp.repeat(thetas_array, reps, axis=0), flat_keys, context
    )
    return jax.tree.map(lambda x: x.reshape(n, reps, *x.shape[2:]), out)


def _even_batch(size: int, batch: int) -> int:
    """Return the batch of at most `batch`, and at least half of it, that needs the
    least padding to cover `size` filters, preferring the larger of any ties."""
    return min(range((batch + 1) // 2, batch + 1), key=lambda b: (-size % b, -b))


@partial(jax.jit, static_argnames=("batch", "n_devices"))
def _filter_batches(
    thetas_array: jax.Array,
    keys: jax.Array,
    context: PfilterContext,
    *,
    batch: int,
    n_devices: int,
) -> dict[str, jax.Array]:
    """Run all filters, `batch` at a time on each of `n_devices` devices."""
    n, reps = keys.shape[:2]
    per_device = n * reps // n_devices
    if batch >= per_device:
        return _flat_reps(thetas_array, keys, context)

    # Pad to even batches: a remainder would compile a second copy of the filter.
    batch = _even_batch(per_device, batch)
    padded = -(-per_device // batch) * batch

    def by_device(x: jax.Array) -> jax.Array:
        x = x.reshape(n_devices, per_device, *x.shape[1:])
        return pad_array(x, 1, padded, per_device)

    def run_device(thetas: jax.Array, device_keys: jax.Array) -> dict[str, jax.Array]:
        return jax.lax.map(
            lambda args: _pfilter_internal(args[0], args[1], context),
            (thetas, device_keys),
            batch_size=batch,
        )

    out = jax.vmap(run_device)(
        by_device(jnp.repeat(thetas_array, reps, axis=0)),
        by_device(keys.reshape(n * reps, *keys.shape[2:])),
    )
    return jax.tree.map(lambda x: x[:, :per_device].reshape(n, reps, *x.shape[2:]), out)


def _run_reps(
    thetas_array: jax.Array, keys: jax.Array, context: PfilterContext
) -> dict[str, jax.Array]:
    """Run the filters, choosing how to batch them when no batch size is given."""
    on_cpu = jax.default_backend() == "cpu"
    if keys.shape[1] == 1 and (context.reps_batch_size is not None or on_cpu):
        return _vmapped_pfilter_internal2(thetas_array, keys, context)
    if context.reps_batch_size is not None:
        if context.reps_batch_size >= keys.shape[1]:
            return _flat_reps(thetas_array, keys, context)
        return _vmapped_pfilter_internal2(thetas_array, keys, context)
    if on_cpu:
        return _vmapped_pfilter_internal2(
            thetas_array, keys, replace(context, reps_batch_size=1)
        )
    leaves = jax.tree_util.tree_leaves((thetas_array, keys, context))
    if any(isinstance(x, Tracer) for x in leaves):
        return _flat_reps(thetas_array, keys, context)

    # Parameter sets may be split over devices; replicates never are.
    shard_shape = thetas_array.sharding.shard_shape(thetas_array.shape)
    n_devices = thetas_array.shape[0] // shard_shape[0]
    per_device = keys.shape[0] * keys.shape[1] // n_devices
    batch = min(per_device, -(-_PARTICLES_PER_BATCH // context.J))

    compiled = _filter_batches.lower(
        thetas_array, keys, context, batch=batch, n_devices=n_devices
    ).compile()
    mem, free = _memory_bytes(compiled), _free_device_bytes((thetas_array, keys))
    smaller = None if mem is None or free is None else _batch_for(*mem, free, batch)
    if smaller is None:
        return compiled(thetas_array, keys, context)
    return _filter_batches(
        thetas_array, keys, context, batch=smaller, n_devices=n_devices
    )


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

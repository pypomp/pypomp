from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from .carries import PfilterState
from .contexts import PfilterContext, SeriesData
from .helpers import _no_resampler, _normalize_weights, _resampler

SHOULD_TRANS = False  # Should transformations be applied to the parameters?


@jit
def _pfilter_internal(
    theta: jax.Array,
    key: jax.Array,
    context: PfilterContext,
) -> dict[str, jax.Array]:
    """Main internal function for particle the filtering algorithm."""
    # 1. Setup and initialize keys.
    split_keys = jax.random.split(key, num=context.J + 1)
    key = split_keys[0]
    keys = split_keys[1:]

    # 2. Initialize particle states at t0.
    covars0 = (
        None
        if context.series.covars_extended is None
        else context.series.covars_extended[0]
    )
    particlesF = context.fns.rinitializer(
        theta, keys, covars0, context.series.t0, context.should_trans
    )
    norm_weights = jnp.log(jnp.ones(context.J) / context.J)
    counts = jnp.ones(context.J).astype(int)
    loglik = 0.0

    # 3. Prepare arrays to store diagnostics/metrics if requested.
    n_obs = len(context.series.ys)
    CLL_arr = jnp.zeros(n_obs) if context.CLL else jnp.zeros(0)
    ESS_arr = jnp.zeros(n_obs) if context.ESS else jnp.zeros(0)
    filter_mean_arr = (
        jnp.zeros((n_obs, particlesF.shape[-1]))
        if context.filter_mean
        else jnp.zeros((0, particlesF.shape[-1]))
    )
    prediction_mean_arr = (
        jnp.zeros((n_obs, particlesF.shape[-1]))
        if context.prediction_mean
        else jnp.zeros((0, particlesF.shape[-1]))
    )

    # 4. Prepare input for particle filter loop and run it.
    initial_state = PfilterState(
        t=context.series.t0,
        particlesF=particlesF,
        loglik=loglik,
        norm_weights=norm_weights,
        counts=counts,
        key=key,
        t_idx=0,
        CLL_arr=CLL_arr,
        ESS_arr=ESS_arr,
        filter_mean_arr=filter_mean_arr,
        prediction_mean_arr=prediction_mean_arr,
    )

    pfilter_step_checkpointed = jax.checkpoint(partial(_pfilter_step, context, theta))

    def body_fun(i, state):
        return pfilter_step_checkpointed(i, state)

    final_state = jax.lax.fori_loop(
        lower=0,
        upper=n_obs,
        body_fun=body_fun,
        init_val=initial_state,
    )

    # 5. Package and return the results.
    output = {"neg_loglik": -final_state.loglik}

    if context.CLL:
        output["CLL"] = final_state.CLL_arr
    if context.ESS:
        output["ESS"] = final_state.ESS_arr
    if context.filter_mean:
        output["filter_mean"] = final_state.filter_mean_arr
    if context.prediction_mean:
        output["prediction_mean"] = final_state.prediction_mean_arr

    return output


def _pfilter_step(
    context: PfilterContext,
    theta: jax.Array,
    i: int,
    state: PfilterState,
) -> PfilterState:
    """Run the particle filter for one observation interval."""
    # 1. Setup and initialize keys.
    key, subkey = jax.random.split(state.key)
    nstep = context.series.nstep_array[i].astype(int)

    # 2. Propagate particles for one observation interval.
    particlesP, t_idx = context.fns.rprocess_interp(
        state.particlesF,
        theta,
        subkey,
        context.series.covars_extended,
        context.series.dt_array_extended,
        state.t,
        state.t_idx,
        nstep,
        context.fns.accumvars,
        context.should_trans,
    )
    t = context.series.times[i]

    # 3. Update covariates to current observation time.
    covars_t = (
        None
        if context.series.covars_extended is None
        else context.series.covars_extended[t_idx]
    )

    # 4. Compute log-likelihood contribution of current observation.
    measurements = context.fns.dmeasure(
        context.series.ys[i], particlesP, theta, covars_t, t, context.should_trans
    )

    # 5. Update running log-likelihood and normalize particle weights.
    weights = state.norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik = state.loglik + loglik_t

    # 6. Compute and store diagnostics/metrics if requested.
    CLL_arr = state.CLL_arr
    ESS_arr = state.ESS_arr
    filter_mean_arr = state.filter_mean_arr
    prediction_mean_arr = state.prediction_mean_arr

    if context.CLL:
        CLL_arr = CLL_arr.at[i].set(loglik_t)
    if context.ESS:
        ess_t = 1.0 / jnp.sum(jnp.exp(2.0 * norm_weights))
        ESS_arr = ESS_arr.at[i].set(ess_t)
    if context.filter_mean:
        filter_mean_t = (particlesP * jnp.exp(norm_weights[:, None])).sum(axis=0)
        filter_mean_arr = filter_mean_arr.at[i].set(filter_mean_t)
    if context.prediction_mean:
        prediction_mean_t = particlesP.mean(axis=0)
        prediction_mean_arr = prediction_mean_arr.at[i].set(prediction_mean_t)

    # 7. Resample particles if criteria met.
    resample = jnp.max(norm_weights) - jnp.min(norm_weights) > jnp.log(context.thresh)
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights = jax.lax.cond(
        resample,
        _resampler,
        _no_resampler,
        *(state.counts, particlesP, norm_weights, subkey),
    )

    # 8. Return the updated filter state.

    return PfilterState(
        t=t,
        particlesF=particlesF,
        loglik=loglik,
        norm_weights=norm_weights,
        counts=counts,
        key=key,
        t_idx=t_idx,
        CLL_arr=CLL_arr,
        ESS_arr=ESS_arr,
        filter_mean_arr=filter_mean_arr,
        prediction_mean_arr=prediction_mean_arr,
    )


@jit
def _mapped_pfilter_internal_reps(
    theta: jax.Array,
    keys: jax.Array,
    context: PfilterContext,
) -> dict[str, jax.Array]:
    def body(key):
        return _pfilter_internal(theta, key, context)

    return jax.lax.map(body, keys)


# Map over key
_vmapped_pfilter_internal = jax.vmap(
    _pfilter_internal,
    in_axes=(None, 0, None),
)

# Map over theta and lax.map over key
_vmapped_pfilter_internal2 = jax.vmap(
    _mapped_pfilter_internal_reps,
    in_axes=(0, 0, None),
)


def _panel_pfilter_vmap(
    thetas: jax.Array,
    keys: jax.Array,
    context: PfilterContext,
) -> dict[str, jax.Array]:
    """vmap ``_pfilter_internal`` over units, mapping ys/covars per unit.

    The ``in_axes`` prototype is derived from ``context`` so that its static
    fields match the value's treedef exactly (a module-level constant with
    placeholder statics would not).
    """
    axes = replace(context, series=SeriesData.axes(ys=0, covars_extended=0))
    return jax.vmap(
        _pfilter_internal,
        in_axes=(
            0,  # theta
            0,  # key
            axes,  # context (only ys/covars are mapped)
        ),
    )(thetas, keys, context)


@jit
def _pfilter_internal_mean(
    theta: jax.Array,
    key: jax.Array,
    context: PfilterContext,
) -> jax.Array:
    """
    Returns particle filter estimate of the negative log likelihood divided by the
    length of the observations. Used in internal pypomp.train functions.
    """
    return (
        _pfilter_internal(theta=theta, key=key, context=context)["neg_loglik"]
        / context.series.ys.shape[0]
    )


@partial(jit, static_argnames=("chunk_size",))
def _chunked_panel_pfilter_internal(
    thetas: jax.Array,
    keys: jax.Array,
    context: PfilterContext,
    chunk_size: int,
) -> dict[str, jax.Array]:
    """Run pfilter in vmapped chunks over multiple panel units."""
    # 1. Reshape inputs for chunked processing.
    n_reps, U, n_params = thetas.shape
    n_chunks = U // chunk_size

    thetas_c = thetas.reshape((n_reps, n_chunks, chunk_size, n_params))
    ys_c = context.series.ys.reshape(
        (n_chunks, chunk_size) + context.series.ys.shape[1:]
    )
    covars_c = (
        None
        if context.series.covars_extended is None
        else context.series.covars_extended.reshape(
            (n_chunks, chunk_size) + context.series.covars_extended.shape[1:]
        )
    )
    keys_c = keys.reshape((n_reps, n_chunks, chunk_size) + keys.shape[2:])

    # 2. Define unit/chunk processing loop.
    def process_rep(theta_r, key_r):
        def scan_fn(carry, chunk_idx):
            theta_chunk = theta_r[chunk_idx]
            ys_chunk = ys_c[chunk_idx]
            covars_chunk = None if covars_c is None else covars_c[chunk_idx]
            key_chunk = key_r[chunk_idx]

            context_chunk = replace(
                context,
                series=replace(
                    context.series, ys=ys_chunk, covars_extended=covars_chunk
                ),
            )

            res = _panel_pfilter_vmap(
                theta_chunk,
                key_chunk,
                context_chunk,
            )
            return carry, res

        # 3. Perform scan and run the chunked particle filter.
        _, res_chunks = jax.lax.scan(scan_fn, None, jnp.arange(n_chunks))

        # 4. Reshape outputs back to the original panel format.
        def reshape_back(arr):
            return arr.reshape((U,) + arr.shape[2:])

        return jax.tree_util.tree_map(reshape_back, res_chunks)

    return jax.vmap(process_rep)(thetas_c, keys_c)

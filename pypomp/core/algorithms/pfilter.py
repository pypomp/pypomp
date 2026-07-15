from functools import partial
from typing import cast
import jax
import jax.numpy as jnp
from jax import jit
from .helpers import _resampler
from .helpers import _no_resampler
from .helpers import _normalize_weights
from .types import PfilterConfig, PfilterInputs, PfilterState

SHOULD_TRANS = False  # Should transformations be applied to the parameters?


@partial(jit, static_argnames=("config",))
def _pfilter_internal(
    theta: jax.Array,
    key: jax.Array,
    config: PfilterConfig,
    inputs: PfilterInputs,
) -> dict[str, jax.Array]:
    """Main internal function for particle the filtering algorithm."""
    # 1. Setup and initialize keys.
    split_keys = jax.random.split(key, num=config.J + 1)
    key = split_keys[0]
    keys = split_keys[1:]

    # 2. Initialize particle states at t0.
    covars0 = None if inputs.covars_extended is None else inputs.covars_extended[0]
    particlesF = config.rinitializer(
        theta, keys, covars0, inputs.t0, config.should_trans
    )
    norm_weights = jnp.log(jnp.ones(config.J) / config.J)
    counts = jnp.ones(config.J).astype(int)
    loglik = 0.0

    # 3. Prepare arrays to store diagnostics/metrics if requested.
    n_obs = len(inputs.ys)
    CLL_arr = jnp.zeros(n_obs) if config.CLL else jnp.zeros(0)
    ESS_arr = jnp.zeros(n_obs) if config.ESS else jnp.zeros(0)
    filter_mean_arr = (
        jnp.zeros((n_obs, particlesF.shape[-1]))
        if config.filter_mean
        else jnp.zeros((0, particlesF.shape[-1]))
    )
    prediction_mean_arr = (
        jnp.zeros((n_obs, particlesF.shape[-1]))
        if config.prediction_mean
        else jnp.zeros((0, particlesF.shape[-1]))
    )

    # 4. Prepare input for particle filter loop and run it.
    initial_state = PfilterState(
        t=inputs.t0,
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

    pfilter_step_checkpointed = jax.checkpoint(
        partial(
            _pfilter_step,
            config,
            inputs,
            theta,
        )
    )

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

    if config.CLL:
        output["CLL"] = final_state.CLL_arr
    if config.ESS:
        output["ESS"] = final_state.ESS_arr
    if config.filter_mean:
        output["filter_mean"] = final_state.filter_mean_arr
    if config.prediction_mean:
        output["prediction_mean"] = final_state.prediction_mean_arr

    return output


def _pfilter_step(
    config: PfilterConfig,
    inputs: PfilterInputs,
    theta: jax.Array,
    i: int,
    state: PfilterState,
) -> PfilterState:
    """Run the particle filter for one observation interval."""
    # 1. Setup and initialize keys.
    split_keys = jax.random.split(state.key, num=config.J + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    nstep = inputs.nstep_array[i].astype(int)

    # 2. Propagate particles for one observation interval.
    particlesP, t_idx = config.rprocess_interp(
        state.particlesF,
        theta,
        keys,
        inputs.covars_extended,
        inputs.dt_array_extended,
        state.t,
        state.t_idx,
        nstep,
        config.accumvars,
        config.should_trans,
    )
    t = inputs.times[i]

    # 3. Update covariates to current observation time.
    covars_t = None if inputs.covars_extended is None else inputs.covars_extended[t_idx]

    # 4. Compute log-likelihood contribution of current observation.
    measurements = config.dmeasure(
        inputs.ys[i], particlesP, theta, covars_t, t, config.should_trans
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

    if config.CLL:
        CLL_arr = CLL_arr.at[i].set(loglik_t)
    if config.ESS:
        ess_t = 1.0 / jnp.sum(jnp.exp(2.0 * norm_weights))
        ESS_arr = ESS_arr.at[i].set(ess_t)
    if config.filter_mean:
        filter_mean_t = (particlesP * jnp.exp(norm_weights[:, None])).sum(axis=0)
        filter_mean_arr = filter_mean_arr.at[i].set(filter_mean_t)
    if config.prediction_mean:
        prediction_mean_t = particlesP.mean(axis=0)
        prediction_mean_arr = prediction_mean_arr.at[i].set(prediction_mean_t)

    # 7. Resample particles if criteria met.
    resample = jnp.max(norm_weights) - jnp.min(norm_weights) > jnp.log(config.thresh)
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


@partial(jit, static_argnames=("config",))
def _mapped_pfilter_internal_reps(
    theta: jax.Array,
    keys: jax.Array,
    config: PfilterConfig,
    inputs: PfilterInputs,
) -> dict[str, jax.Array]:
    def body(key):
        return _pfilter_internal(
            theta,
            key,
            config,
            inputs,
        )

    return jax.lax.map(body, keys)


# Map over key
_vmapped_pfilter_internal = jax.vmap(
    _pfilter_internal,
    in_axes=(None, 0, None, None),
)

# Map over theta and lax.map over key
_vmapped_pfilter_internal2 = jax.vmap(
    _mapped_pfilter_internal_reps,
    in_axes=(0, 0, None, None),
)

inputs_in_axes = PfilterInputs(
    ys=cast(jax.Array, 0),
    dt_array_extended=cast(jax.Array, None),
    nstep_array=cast(jax.Array, None),
    t0=cast(float, None),
    times=cast(jax.Array, None),
    covars_extended=cast(jax.Array, 0),
)

_panel_pfilter_vmap = jax.vmap(
    _pfilter_internal,
    in_axes=(
        0,  # theta
        0,  # key
        None,  # config
        inputs_in_axes,  # inputs
    ),
)


@partial(jit, static_argnames=("config",))
def _pfilter_internal_mean(
    theta: jax.Array,
    key: jax.Array,
    config: PfilterConfig,
    inputs: PfilterInputs,
) -> jax.Array:
    """
    Returns particle filter estimate of the negative log likelihood divided by the
    length of the observations. Used in internal pypomp.train functions.
    """
    return (
        _pfilter_internal(
            theta=theta,
            key=key,
            config=config,
            inputs=inputs,
        )["neg_loglik"]
        / inputs.ys.shape[0]
    )


@partial(
    jit,
    static_argnames=(
        "config",
        "chunk_size",
    ),
)
def _chunked_panel_pfilter_internal(
    thetas: jax.Array,
    keys: jax.Array,
    config: PfilterConfig,
    inputs: PfilterInputs,
    chunk_size: int,
) -> dict[str, jax.Array]:
    """Run pfilter in vmapped chunks over multiple panel units."""
    # 1. Reshape inputs for chunked processing.
    n_reps, U, n_params = thetas.shape
    n_chunks = U // chunk_size

    thetas_c = thetas.reshape((n_reps, n_chunks, chunk_size, n_params))
    ys_c = inputs.ys.reshape((n_chunks, chunk_size) + inputs.ys.shape[1:])
    covars_c = (
        None
        if inputs.covars_extended is None
        else inputs.covars_extended.reshape(
            (n_chunks, chunk_size) + inputs.covars_extended.shape[1:]
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

            inputs_chunk = PfilterInputs(
                ys=ys_chunk,
                dt_array_extended=inputs.dt_array_extended,
                nstep_array=inputs.nstep_array,
                t0=inputs.t0,
                times=inputs.times,
                covars_extended=covars_chunk,
            )

            res = _panel_pfilter_vmap(
                theta_chunk,
                key_chunk,
                config,
                inputs_chunk,
            )
            return carry, res

        # 3. Perform scan and run the chunked particle filter.
        _, res_chunks = jax.lax.scan(scan_fn, None, jnp.arange(n_chunks))

        # 4. Reshape outputs back to the original panel format.
        def reshape_back(arr):
            return arr.reshape((U,) + arr.shape[2:])

        return jax.tree_util.tree_map(reshape_back, res_chunks)

    return jax.vmap(process_rep)(thetas_c, keys_c)

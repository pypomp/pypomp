"""
This module implements the MOP algorithm for POMP models.
"""

from functools import partial
from typing import cast
import jax
import jax.numpy as jnp
from jax import jit
from .helpers import _normalize_weights
from .helpers import _resampler
from .types import MopConfig, MopInputs, MopState

SHOULD_TRANS = True  # Should transformations be applied to the parameters?


@partial(jit, static_argnames=("config",))
def _mop_internal(
    theta: jax.Array,
    key: jax.Array,
    config: MopConfig,
    inputs: MopInputs,
) -> jax.Array | float:
    """
    Internal function for the MOP algorithm, which calls function '_mop_step'
    iteratively.
    """
    split_keys = jax.random.split(key, num=config.J + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    covars0 = None if inputs.covars_extended is None else inputs.covars_extended[0]
    particlesF = config.rinitializer(theta, keys, covars0, inputs.t0, SHOULD_TRANS)
    weightsF = jnp.log(jnp.ones(config.J) / config.J)
    counts = jnp.ones(config.J).astype(int)
    loglik = 0.0

    initial_state = MopState(
        t=inputs.t0,
        particlesF=particlesF,
        loglik=loglik,
        weightsF=weightsF,
        counts=counts,
        key=key,
        t_idx=0,
    )

    mop_step_checkpointed = jax.checkpoint(
        partial(
            _mop_step,
            config,
            inputs,
            theta,
        )
    )

    def body_fun(i, state):
        return mop_step_checkpointed(i, state)

    final_state = jax.lax.fori_loop(
        lower=0,
        upper=len(inputs.ys),
        body_fun=body_fun,
        init_val=initial_state,
    )

    return -final_state.loglik


def _mop_step(
    config: MopConfig,
    inputs: MopInputs,
    theta: jax.Array,
    i: int,
    state: MopState,
) -> MopState:
    """
    Helper function for the MOP algorithm, which conducts a single iteration of
    filtering and is called in the function '_mop_internal'.
    """
    t = state.t
    particlesF = state.particlesF
    loglik = state.loglik
    weightsF = state.weightsF
    counts = state.counts
    key = state.key
    t_idx = state.t_idx

    J = config.J
    weightsP = inputs.alpha * weightsF

    split_keys = jax.random.split(key, num=J + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    nstep = inputs.nstep_array[i].astype(int)
    particlesP, t_idx = config.rprocess_interp(
        particlesF,
        theta,
        keys,
        inputs.covars_extended,
        inputs.dt_array_extended,
        t,
        t_idx,
        nstep,
        config.accumvars,
        SHOULD_TRANS,
    )
    t = inputs.times[i]

    covars_t = None if inputs.covars_extended is None else inputs.covars_extended[t_idx]
    measurements = config.dmeasure(
        inputs.ys[i], particlesP, theta, covars_t, t, SHOULD_TRANS
    )

    loglik = (
        loglik
        + jax.scipy.special.logsumexp(weightsP + measurements)
        - jax.scipy.special.logsumexp(weightsP)
    )

    norm_weights, _ = _normalize_weights(jax.lax.stop_gradient(measurements))

    key, subkey = jax.random.split(key)
    counts, particlesF, _ = _resampler(counts, particlesP, norm_weights, subkey=subkey)

    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    return MopState(
        t=t,
        particlesF=particlesF,
        loglik=loglik,
        weightsF=weightsF,
        counts=counts,
        key=key,
        t_idx=t_idx,
    )


_vmapped_mop_internal = jax.vmap(
    _mop_internal,
    in_axes=(
        0,  # theta
        0,  # key
        None,  # config
        None,  # inputs
    ),
)


@partial(jit, static_argnames=("config",))
def _mop_internal_mean(
    theta: jax.Array,
    key: jax.Array,
    config: MopConfig,
    inputs: MopInputs,
) -> jax.Array | float:
    """
    Internal function for calculating the MOP estimate of the log likelihood divided by
    the length of the observations. This is used in internal pypomp.train functions.
    """
    return _mop_internal(
        theta=theta,
        key=key,
        config=config,
        inputs=inputs,
    ) / len(inputs.ys)


inputs_in_axes = MopInputs(
    ys=cast(jax.Array, 0),
    dt_array_extended=cast(jax.Array, None),
    nstep_array=cast(jax.Array, None),
    t0=cast(float, None),
    times=cast(jax.Array, None),
    covars_extended=cast(jax.Array, 0),
    alpha=cast(float, None),
)


_panel_mop_internal_vmap = jax.vmap(
    _mop_internal,
    in_axes=(
        0,  # theta
        0,  # key
        None,  # config
        inputs_in_axes,  # inputs
    ),
)


@partial(
    jit,
    static_argnames=(
        "config",
        "chunk_size",
    ),
)
def _chunked_panel_mop_internal(
    shared_array: jax.Array,  # (n_shared,)
    unit_array: jax.Array,  # (U, n_spec)
    unit_param_permutations: jax.Array,  # (U, n_params)
    config: MopConfig,
    inputs: MopInputs,
    keys: jax.Array,
    chunk_size: int,
) -> jax.Array | float:
    U = unit_array.shape[0]
    n_params = unit_param_permutations.shape[1]
    n_chunks = U // chunk_size

    ys_c = inputs.ys.reshape((n_chunks, chunk_size) + inputs.ys.shape[1:])
    covars_c = (
        None
        if inputs.covars_extended is None
        else inputs.covars_extended.reshape(
            (n_chunks, chunk_size) + inputs.covars_extended.shape[1:]
        )
    )
    keys_c = keys.reshape((n_chunks, chunk_size) + keys.shape[1:])

    # unit_array: (U, n_spec) -> (n_chunks, chunk_size, n_spec)
    unit_array_c = unit_array.reshape((n_chunks, chunk_size, -1))

    # unit_param_permutations: (U, n_params) -> (n_chunks, chunk_size, n_params)
    unit_param_permutations_c = unit_param_permutations.reshape(
        (n_chunks, chunk_size, n_params)
    )

    shared_tiled = jnp.tile(shared_array, (chunk_size, 1))

    scan_fn = jax.tree_util.Partial(
        _panel_mop_scan_step,
        config,
        inputs,
        ys_c,
        covars_c,
        keys_c,
        unit_array_c,
        unit_param_permutations_c,
        shared_tiled,
    )

    total_neg_loglik, _ = jax.lax.scan(scan_fn, 0.0, jnp.arange(n_chunks))

    return total_neg_loglik / (U * inputs.ys.shape[1])


def _panel_mop_scan_step(
    config: MopConfig,
    inputs: MopInputs,
    ys_c: jax.Array,
    covars_c: jax.Array | None,
    keys_c: jax.Array,
    unit_array_c: jax.Array,
    unit_param_permutations_c: jax.Array,
    shared_tiled: jax.Array,
    carry: jax.Array,
    chunk_idx: int,
) -> tuple[jax.Array, None]:
    unit_array_chunk = unit_array_c[chunk_idx]  # (chunk_size, n_spec)
    unit_param_perm_chunk = unit_param_permutations_c[
        chunk_idx
    ]  # (chunk_size, n_params)

    theta_chunk_unordered = jnp.concatenate([shared_tiled, unit_array_chunk], axis=1)

    def apply_perm(theta, perm):
        return theta[perm]

    theta_chunk = jax.vmap(apply_perm)(theta_chunk_unordered, unit_param_perm_chunk)

    ys_chunk = ys_c[chunk_idx]
    covars_chunk = None if covars_c is None else covars_c[chunk_idx]
    key_chunk = keys_c[chunk_idx]

    inputs_chunk = MopInputs(
        ys=ys_chunk,
        dt_array_extended=inputs.dt_array_extended,
        nstep_array=inputs.nstep_array,
        t0=inputs.t0,
        times=inputs.times,
        covars_extended=covars_chunk,
        alpha=inputs.alpha,
    )

    res = _panel_mop_internal_vmap(
        theta_chunk,
        key_chunk,
        config,
        inputs_chunk,
    )
    return carry + jnp.sum(res), None


_vg_chunked_panel_mop_internal = jax.value_and_grad(
    _chunked_panel_mop_internal, argnums=(0, 1)
)

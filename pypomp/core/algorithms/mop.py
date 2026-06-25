"""
This module implements the MOP algorithm for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable
from .helpers import _normalize_weights
from .helpers import _resampler

SHOULD_TRANS = True  # Should transformations be applied to the parameters?


@partial(jit, static_argnames=("J", "rinitializer", "rprocess_interp", "dmeasure"))
def _mop_internal(
    theta: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function for the MOP algorithm, which calls function 'mop_helper'
    iteratively.
    """
    times = times.astype(float)
    split_keys = jax.random.split(key, num=J + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF = rinitializer(theta, keys, covars0, t0, SHOULD_TRANS)
    weightsF = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0.0

    # My linter thinks jax.checkpoint isn't exported from jax, but it is.
    mop_helper_2 = jax.checkpoint(  # type: ignore[reportAttributeAccessIssue]
        partial(
            _mop_helper,
            ys=ys,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            times=times,
            theta=theta,
            rprocess_interp=rprocess_interp,
            dmeasure=dmeasure,
            covars_extended=covars_extended,
            alpha=alpha,
            accumvars=accumvars,
        )
    )

    t, particlesF, loglik, weightsF, counts, key, obs_idx = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=mop_helper_2,
        init_val=(t0, particlesF, loglik, weightsF, counts, key, 0),
    )

    return -loglik


@partial(jit, static_argnames=("J", "rinitializer", "rprocess_interp", "dmeasure"))
def _mop_internal_mean(
    theta: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function for calculating the MOP estimate of the log likelihood divided by
    the length of the observations. This is used in internal pypomp.train functions.
    """
    return _mop_internal(
        theta=theta,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess_interp,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        accumvars=accumvars,
        alpha=alpha,
        key=key,
    ) / len(ys)


def _mop_helper(
    i: int,
    inputs: tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int
    ],
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    times: jax.Array,
    theta: jax.Array,
    rprocess_interp: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """
    Helper function for the MOP algorithm, which conducts a single iteration of
    filtering and is called in the function 'mop_internal'.
    """
    t, particlesF, loglik, weightsF, counts, key, t_idx = inputs
    J = len(particlesF)

    weightsP = alpha * weightsF

    split_keys = jax.random.split(key, num=J + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    nstep = nstep_array[i].astype(int)
    particlesP, t_idx = rprocess_interp(
        particlesF,
        theta,
        keys,
        covars_extended,
        dt_array_extended,
        t,
        t_idx,
        nstep,
        accumvars,
        SHOULD_TRANS,
    )
    t = times[i]

    covars_t = None if covars_extended is None else covars_extended[t_idx]
    measurements = dmeasure(ys[i], particlesP, theta, covars_t, t, SHOULD_TRANS)

    loglik = (
        loglik
        + jax.scipy.special.logsumexp(weightsP + measurements)
        - jax.scipy.special.logsumexp(weightsP)
    )
    # test different, logsumexp - source code (floating point arithmetic issue)
    # make a little note in the code, discuss it in the quant test about the small difference
    # logsumexp source code

    norm_weights, loglik_phi_t = _normalize_weights(jax.lax.stop_gradient(measurements))

    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weightsF = _resampler(
        counts, particlesP, norm_weights, subkey=subkey
    )

    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    return (t, particlesF, loglik, weightsF, counts, key, t_idx)


_vmapped_mop_internal = jax.vmap(
    _mop_internal,
    in_axes=(
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
    ),
)


_panel_mop_internal_vmap = jax.vmap(
    _mop_internal,
    in_axes=(
        0,  # theta
        0,  # ys
        None,  # dt_array_extended
        None,  # nstep_array
        None,  # t0
        None,  # times
        None,  # J
        None,  # rinitializer
        None,  # rprocess_interp
        None,  # dmeasure
        None,  # accumvars
        0,  # covars_extended
        None,  # alpha
        0,  # key
    ),
)


@partial(
    jit,
    static_argnames=(
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "accumvars",
        "chunk_size",
    ),
)
def _chunked_panel_mop_internal(
    shared_array: jax.Array,  # (n_shared,)
    unit_array: jax.Array,  # (U, n_spec)
    unit_param_permutations: jax.Array,  # (U, n_params)
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    covars_extended: jax.Array | None,
    keys: jax.Array,
    J: int,
    rinitializer: Callable,
    rprocess_interp: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    chunk_size: int,
    alpha: float,
):
    U = unit_array.shape[0]
    n_params = unit_param_permutations.shape[1]
    n_chunks = U // chunk_size

    ys_c = ys.reshape((n_chunks, chunk_size) + ys.shape[1:])
    covars_c = (
        None
        if covars_extended is None
        else covars_extended.reshape((n_chunks, chunk_size) + covars_extended.shape[1:])
    )
    keys_c = keys.reshape((n_chunks, chunk_size) + keys.shape[1:])

    # unit_array: (U, n_spec) -> (n_chunks, chunk_size, n_spec)
    unit_array_c = unit_array.reshape((n_chunks, chunk_size, -1))

    # unit_param_permutations: (U, n_params) -> (n_chunks, chunk_size, n_params)
    unit_param_permutations_c = unit_param_permutations.reshape(
        (n_chunks, chunk_size, n_params)
    )

    shared_tiled = jnp.tile(shared_array, (chunk_size, 1))

    def scan_fn(carry, chunk_idx):
        unit_array_chunk = unit_array_c[chunk_idx]  # (chunk_size, n_spec)
        unit_param_perm_chunk = unit_param_permutations_c[
            chunk_idx
        ]  # (chunk_size, n_params)

        theta_chunk_unordered = jnp.concatenate(
            [shared_tiled, unit_array_chunk], axis=1
        )

        def apply_perm(theta, perm):
            return theta[perm]

        theta_chunk = jax.vmap(apply_perm)(theta_chunk_unordered, unit_param_perm_chunk)

        ys_chunk = ys_c[chunk_idx]
        covars_chunk = None if covars_c is None else covars_c[chunk_idx]
        key_chunk = keys_c[chunk_idx]

        res = _panel_mop_internal_vmap(
            theta_chunk,
            ys_chunk,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            J,
            rinitializer,
            rprocess_interp,
            dmeasure,
            accumvars,
            covars_chunk,
            alpha,
            key_chunk,
        )
        return carry + jnp.sum(res), None

    total_neg_loglik, _ = jax.lax.scan(scan_fn, 0.0, jnp.arange(n_chunks))

    return total_neg_loglik / (U * ys.shape[1])


_vg_chunked_panel_mop_internal = jax.value_and_grad(
    _chunked_panel_mop_internal, argnums=(0, 1)
)

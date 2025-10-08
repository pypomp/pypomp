"""
This module implements the MOP algorithm for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights
from .internal_functions import _resampler


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
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF = rinitializer(theta, keys, covars0, t0)
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

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
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
    )
    t = times[i]

    covars_t = None if covars_extended is None else covars_extended[t_idx]
    measurements = dmeasure(ys[i], particlesP, theta, covars_t, t)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

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

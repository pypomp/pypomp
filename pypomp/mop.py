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


@partial(jit, static_argnums=(5, 6, 7, 8))
def _mop_internal(
    theta: jax.Array,
    dt_array_extended: jax.Array,
    t0: float,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
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
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF = rinitializer(theta, keys, covars0, t0)
    weightsF = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0.0

    mop_helper_2 = partial(
        _mop_helper,
        dt_array_extended=dt_array_extended,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        theta=theta,
        rprocess=rprocess,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        alpha=alpha,
        accumvars=accumvars,
    )

    t, particlesF, loglik, weightsF, counts, key = jax.lax.fori_loop(
        lower=0,
        upper=len(ys_extended),
        body_fun=mop_helper_2,
        init_val=(t0, particlesF, loglik, weightsF, counts, key),
    )

    return -loglik


@partial(jit, static_argnums=(5, 6, 7, 8))
def _mop_internal_mean(
    theta: jax.Array,
    dt_array_extended: jax.Array,
    t0: float,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
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
        t0=t0,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        accumvars=accumvars,
        alpha=alpha,
        key=key,
    ) / jnp.sum(ys_observed)


def _mop_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    dt_array_extended: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    theta: jax.Array,
    rprocess: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Helper function for the MOP algorithm, which conducts a single iteration of
    filtering and is called in the function 'mop_internal'.
    """
    t, particlesF, loglik, weightsF, counts, key = inputs
    J = len(particlesF)

    time_interval_begins = jnp.logical_or(i == 0, ys_observed[i - 1])
    weightsP = jax.lax.cond(
        time_interval_begins,
        lambda weightsF: weightsF,
        lambda weightsF: alpha * weightsF,
        weightsF,
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars_t = None if covars_extended is None else covars_extended[i]
    particlesP = rprocess(particlesF, theta, keys, covars_t, t, dt_array_extended[i])
    t = t + dt_array_extended[i]

    def _with_observation(loglik, norm_weights, counts, key, dmeasure):
        covars_t = None if covars_extended is None else covars_extended[i + 1]
        measurements = dmeasure(ys_extended[i], particlesP, theta, covars_t, t)
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

        norm_weights, loglik_phi_t = _normalize_weights(
            jax.lax.stop_gradient(measurements)
        )

        key, subkey = jax.random.split(key)
        counts, particlesF, norm_weightsF = _resampler(
            counts, particlesP, norm_weights, subkey=subkey
        )

        weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[
            counts
        ]
        particlesF = jnp.where(
            accumvars is not None, particlesF.at[:, accumvars].set(0.0), particlesF
        )
        return particlesF, loglik, weightsF, counts, key

    def _without_observation(loglik, weightsF, counts, key):
        return particlesP, loglik, weightsF, counts, key

    _with_observation_partial = partial(_with_observation, dmeasure=dmeasure)

    particles, loglik, weightsF, counts, key = jax.lax.cond(
        ys_observed[i],
        _with_observation_partial,
        _without_observation,
        *(loglik, weightsF, counts, key),
    )

    return (t, particles, loglik, weightsF, counts, key)

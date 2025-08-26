from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from typing import Callable
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights


@partial(jit, static_argnums=(5, 6, 7, 8))
def _pfilter_internal(
    theta: jax.Array,  # should be first for _line_search in train.py
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
    thresh: float,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function for particle the filtering algorithm, which calls the function
    'pfilter_helper' iteratively. Returns the negative log-likelihood.
    """
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF = rinitializer(theta, keys, covars0, t0)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0.0

    pfilter_helper_2 = partial(
        _pfilter_helper,
        dt_array_extended=dt_array_extended,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        theta=theta,
        rprocess=rprocess,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        thresh=thresh,
        accumvars=accumvars,
    )
    t, particlesF, loglik, norm_weights, counts, key = jax.lax.fori_loop(
        lower=0,
        upper=len(ys_extended),
        body_fun=pfilter_helper_2,
        init_val=(t0, particlesF, loglik, norm_weights, counts, key),
    )

    return -loglik


# Map over key
_vmapped_pfilter_internal = jax.vmap(
    _pfilter_internal,
    in_axes=(None,) * 12 + (0,),
)

# Map over theta and key
_vmapped_pfilter_internal2 = jax.vmap(
    _pfilter_internal,
    in_axes=(0,) + (None,) * 11 + (0,),
)


@partial(jit, static_argnums=(6, 7, 8, 9))
def _pfilter_internal_mean(
    theta: jax.Array,
    dt_array: jax.Array,
    t0: float,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function for calculating the particle filter estimate of the neagative log
    likelihood divided by the length of the observations. This is used in internal
    pypomp.train functions.
    """
    return _pfilter_internal(
        theta=theta,
        dt_array=dt_array,
        t0=t0,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        accumvars=accumvars,
        thresh=thresh,
        key=key,
    ) / len(ys_extended)


def _pfilter_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    dt_array_extended: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    theta: jax.Array,
    rprocess: Callable,
    dmeasure: Callable,
    covars_extended: jax.Array | None,
    thresh: float,
    accumvars: tuple[int, ...] | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Helper function for the particle filtering algorithm in POMP, which conducts
    filtering for one time-iteration.
    """
    (t, particlesF, loglik, norm_weights, counts, key) = inputs
    J = len(particlesF)

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars_t = None if covars_extended is None else covars_extended[i]
    particlesP = rprocess(particlesF, theta, keys, covars_t, t, dt_array_extended[i])
    t = t + dt_array_extended[i]

    def _with_observation(loglik, norm_weights, counts, key, dmeasure):
        covars_t = None if covars_extended is None else covars_extended[i + 1]
        measurements = dmeasure(ys_extended[i], particlesP, theta, covars_t, t)

        if len(measurements.shape) > 1:
            measurements = measurements.sum(axis=-1)

        weights = norm_weights + measurements
        norm_weights, loglik_t = _normalize_weights(weights)
        loglik = loglik + loglik_t

        oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
        key, subkey = jax.random.split(key)
        counts, particlesF, norm_weights = jax.lax.cond(
            oddr > thresh,
            _resampler,
            _no_resampler,
            *(counts, particlesP, norm_weights, subkey),
        )
        particlesF = jnp.where(
            accumvars is not None, particlesF.at[:, accumvars].set(0.0), particlesF
        )
        return (particlesF, loglik, norm_weights, counts, key)

    def _without_observation(loglik, norm_weights, counts, key):
        return (particlesP, loglik, norm_weights, counts, key)

    _with_observation_partial = partial(_with_observation, dmeasure=dmeasure)

    particles, loglik, norm_weights, counts, key = jax.lax.cond(
        ys_observed[i],
        _with_observation_partial,
        _without_observation,
        *(loglik, norm_weights, counts, key),
    )
    return (t, particles, loglik, norm_weights, counts, key)

from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from typing import Callable
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights

@partial(jit, static_argnums=(5, 6, 7, 8, 13, 14, 15, 16, 17))
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
    n_obs: int, # static, number of observed values in ys_observed
    CLL: bool = False, # static
    ESS: bool = False, # static
    filter_mean: bool = False, #static
    prediction_mean: bool = False # static
) -> dict[str, jax.Array]:
    """
    Internal function for particle the filtering algorithm, which calls the function
    'pfilter_helper' iteratively. 
    If no diagnostics are requested, return the negative log likelihood.
    If diagnostics are requested, return a tuple with the negative log likelihood and the requested diagnostics.
    """
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF = rinitializer(theta, keys, covars0, t0)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0.0
    # prepare arrays to store diagnostics if requested
    CLL_arr = jnp.zeros(n_obs) if CLL else jnp.zeros(0)
    ESS_arr = jnp.zeros(n_obs) if ESS else jnp.zeros(0)
    filter_mean_arr = (jnp.zeros((n_obs, particlesF.shape[-1]))
                       if filter_mean else jnp.zeros((0, particlesF.shape[-1])))
    prediction_mean_arr = (jnp.zeros((n_obs, particlesF.shape[-1]))
                       if prediction_mean else jnp.zeros((0, particlesF.shape[-1])))

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
        CLL=CLL,
        ESS=ESS,
        filter_mean=filter_mean,
        prediction_mean=prediction_mean,
    )
    t, particlesF, loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr = jax.lax.fori_loop(
        lower=0,
        upper=len(ys_extended),
        body_fun=pfilter_helper_2,
        init_val=(t0, particlesF, loglik, norm_weights, counts, key, 0, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr),
    )
    
    output = {"neg_loglik": -loglik}

    if CLL: 
        output["CLL"] = CLL_arr
    if ESS: 
        output["ESS"] = ESS_arr
    if filter_mean: 
        output["filter_mean"] = filter_mean_arr
    if prediction_mean: 
        output["prediction_mean"] = prediction_mean_arr

    return output

# Map over key
_vmapped_pfilter_internal = jax.vmap(
    _pfilter_internal,
    in_axes=(None,) * 12 + (0,) + (None,) * 5,
)

# Map over theta and key
_vmapped_pfilter_internal2 = jax.vmap(
    _pfilter_internal,
    in_axes=(0,) + (None,) * 11 + (0,) + (None,) * 5,
)


@partial(jit, static_argnums=(5, 6, 7, 8))
def _pfilter_internal_mean(
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
    thresh: float,
    key: jax.Array,
    n_obs: int
) -> jax.Array:
    """
    Internal function for calculating the particle filter estimate of the negative log
    likelihood divided by the length of the observations. This is used in internal
    pypomp.train functions.
    """
    return _pfilter_internal(
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
        thresh=thresh,
        key=key,
        n_obs=n_obs,
        CLL=False,
        ESS=False,
        filter_mean=False,
        prediction_mean=False
    )["neg_loglik"] / jnp.sum(ys_observed)


def _pfilter_helper(
    i: int,
    inputs: tuple[jax.Array, 
                  jax.Array, 
                  jax.Array, 
                  jax.Array, 
                  jax.Array, 
                  jax.Array,
                  int,
                  jax.Array, 
                  jax.Array, 
                  jax.Array, 
                  jax.Array],
    dt_array_extended: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    theta: jax.Array,
    rprocess: Callable,
    dmeasure: Callable,
    covars_extended: jax.Array | None,
    thresh: float,
    CLL: bool,
    ESS: bool,
    filter_mean: bool,
    prediction_mean: bool,
    accumvars: tuple[int, ...] | None,
) -> tuple[jax.Array, 
           jax.Array, 
           jax.Array, 
           jax.Array, 
           jax.Array, 
           jax.Array,
           int,
           jax.Array, 
           jax.Array, 
           jax.Array, 
           jax.Array]:
    """
    Helper function for the particle filtering algorithm in POMP, which conducts
    filtering for one time-iteration.
    Only update the diagnostics elements when their corresponding boolean elements are set to be TRUE
    """
    (t, particlesF, loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr) = inputs
    J = len(particlesF)

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars_t = None if covars_extended is None else covars_extended[i]
    particlesP = rprocess(particlesF, theta, keys, covars_t, t, dt_array_extended[i])
    t = t + dt_array_extended[i]

    def _with_observation(loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr, dmeasure):
        covars_t = None if covars_extended is None else covars_extended[i + 1]
        measurements = dmeasure(ys_extended[i], particlesP, theta, covars_t, t)

        if len(measurements.shape) > 1:
            measurements = measurements.sum(axis=-1)

        weights = norm_weights + measurements
        norm_weights, loglik_t = _normalize_weights(weights)
        loglik = loglik + loglik_t

        #mask = jnp.arange(ys_observed.shape[0]) < i
        #obs_idx = jnp.sum(ys_observed * mask) # number of observed values up to time i (excluding i) 
        #mask_diagnostics = jnp.arange(ys_observed.shape[0]) == obs_idx
        #mask_diag_vec = mask_diagnostics[:, None] # broadcasting for 2D arrays

        if CLL:
            #CLL_arr = CLL_arr * (~mask_diagnostics) + loglik_t * mask_diagnostics
            CLL_arr = CLL_arr.at[obs_idx].set(loglik_t)
        if ESS:
            ess_t = 1.0 / jnp.sum(jnp.exp(2.0 * norm_weights))
            #ESS_arr = ESS_arr * (~mask_diagnostics) + ess_t * mask_diagnostics
            ESS_arr = ESS_arr.at[obs_idx].set(ess_t)
        if filter_mean:
            filter_mean_t = (particlesP * jnp.exp(norm_weights[:, None])).sum(axis=0)
            #filter_mean_arr = filter_mean_arr * (~mask_diag_vec) + filter_mean_t * mask_diag_vec
            filter_mean_arr = filter_mean_arr.at[obs_idx].set(filter_mean_t)
        if prediction_mean:
            prediction_mean_t = particlesP.mean(axis=0)
            #prediction_mean_arr = prediction_mean_arr * (~mask_diag_vec) + prediction_mean_t * mask_diag_vec
            prediction_mean_arr = prediction_mean_arr.at[obs_idx].set(prediction_mean_t)

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
        obs_idx = obs_idx + 1
        return (particlesF, loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr)

    def _without_observation(loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr):
        return (particlesP, loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr)

    _with_observation_partial = partial(_with_observation, dmeasure=dmeasure)

    particles, loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr = jax.lax.cond(
        ys_observed[i],
        _with_observation_partial,
        _without_observation,
        *(loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr),
    )
    return (t, particles, loglik, norm_weights, counts, key, obs_idx, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr)

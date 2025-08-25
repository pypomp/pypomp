from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from typing import Callable
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights
from .internal_functions import _interp_covars


@partial(jit, static_argnums=(4, 5, 6, 7, 12, 13, 14, 15))
def _pfilter_internal(
    theta: jax.Array,  # should be first for _line_search
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
    CLL: bool = False, # static
    ESS: bool = False, # static
    filter_mean: bool = False, #static
    prediction_mean: bool = False # static
) -> jax.Array | dict[str, jax.Array]:
    """
    Internal function for particle the filtering algorithm, which calls the function
    'pfilter_helper' iteratively. 
    If no diagnostics are requested, return the negative log likelihood.
    If diagnostics are requested, return a tuple with the negative log likelihood and the requested diagnostics.
    """
    key, keys = _keys_helper(key=key, J=J, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    particlesF = rinitializer(theta, keys, covars_t, t0)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0

    CLL_arr = jnp.zeros(len(ys)) if CLL else jnp.zeros(0)
    ESS_arr = jnp.zeros(len(ys)) if ESS else jnp.zeros(0)
    filter_mean_arr = (jnp.zeros((len(ys), particlesF.shape[-1]))
                       if filter_mean else jnp.zeros((0, particlesF.shape[-1])))
    prediction_mean_arr = (jnp.zeros((len(ys), particlesF.shape[-1]))
                           if prediction_mean else jnp.zeros((0, particlesF.shape[-1])))

    pfilter_helper_2 = partial(
        _pfilter_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        ys=ys,
        theta=theta,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        CLL=CLL,
        ESS=ESS,
        filter_mean=filter_mean,
        prediction_mean=prediction_mean
    )
    particlesF, loglik, norm_weights, counts, key, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=pfilter_helper_2,
        init_val=(particlesF, loglik, norm_weights, counts, key, CLL_arr, ESS_arr, filter_mean_arr, prediction_mean_arr),
    )
    
    any_diagnostics = CLL or ESS or filter_mean or prediction_mean
    if not any_diagnostics:
        return -loglik

    #outputs = [-loglik]
    #if CLL: outputs.append(CLL_arr)
    #if ESS: outputs.append(ESS_arr)
    #if filter_mean: outputs.append(filter_mean_arr)
    #if prediction_mean: outputs.append(prediction_mean_arr)
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
    in_axes=(None,) * 11 + (0,) + (None,) * 4,
)

# Map over theta and key
_vmapped_pfilter_internal2 = jax.vmap(
    _pfilter_internal,
    in_axes=(0,) + (None,) * 10 + (0,) + (None,) * 4,
)


@partial(jit, static_argnums=(4, 5, 6, 7))
def _pfilter_internal_mean(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function for calculating the particle filter estimate of the negative log
    likelihood divided by the length of the observations. This is used in internal
    pypomp.train functions.
    """
    return _pfilter_internal(
        theta=theta,
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        key=key,
    ) / len(ys)


def _pfilter_helper(
    i: int,
    inputs: tuple[jax.Array, float, jax.Array, jax.Array, jax.Array,
                  jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    ys: jax.Array,
    theta: jax.Array,
    rprocess: Callable,
    dmeasure: Callable,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    CLL: bool,
    ESS: bool,
    filter_mean: bool,
    prediction_mean: bool
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, 
           jax.Array, jax.Array, jax.Array]:
    """
    Helper function for the particle filtering algorithm in POMP, which conducts
    filtering for one time-iteration.
    Only update the diagnostics elements when their corresponding boolean elements are set to be TRUE
    """
    (particlesF, loglik, norm_weights, counts, key, CLL_arr, ESS_arr, 
     filter_mean_arr, prediction_mean_arr) = inputs
    J = len(particlesF)
    t1 = times0[i]
    t2 = times0[i + 1]

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesP = rprocess(particlesF, theta, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    measurements = dmeasure(ys[i], particlesP, theta, covars_t, t2)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik += loglik_t

    if CLL:
        CLL_arr = CLL_arr.at[i].set(loglik_t)
    if ESS:
        ess_t = 1.0 / jnp.sum(jnp.exp(2.0 * norm_weights))
        ESS_arr = ESS_arr.at[i].set(ess_t)
    if filter_mean:
        filter_mean_t = (particlesP * jnp.exp(norm_weights[:, None])).sum(axis=0)
        filter_mean_arr = filter_mean_arr.at[i].set(filter_mean_t)
    if prediction_mean:
        prediction_mean_t = particlesP.mean(axis=0)
        prediction_mean_arr = prediction_mean_arr.at[i].set(prediction_mean_t)

    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights = jax.lax.cond(
        oddr > thresh,
        _resampler,
        _no_resampler,
        *(counts, particlesP, norm_weights, subkey),
    )

    return (particlesF, 
            loglik, 
            norm_weights, 
            counts, 
            key, 
            CLL_arr, 
            ESS_arr, 
            filter_mean_arr, 
            prediction_mean_arr,
            )

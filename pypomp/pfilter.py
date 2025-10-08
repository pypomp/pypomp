from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from typing import Callable
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights


@partial(
    jit,
    static_argnames=(
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "CLL",
        "ESS",
        "filter_mean",
        "prediction_mean",
    ),
)
def _pfilter_internal(
    theta: jax.Array,  # should be first for _line_search in train.py
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
    CLL: bool = False,  # static
    ESS: bool = False,  # static
    filter_mean: bool = False,  # static
    prediction_mean: bool = False,  # static
) -> dict[str, jax.Array]:
    """
    Internal function for particle the filtering algorithm, which calls the function
    'pfilter_helper' iteratively.
    If no diagnostics are requested, return the negative log likelihood.
    If diagnostics are requested, return a tuple with the negative log likelihood and the requested diagnostics.
    """
    times = times.astype(float)
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF = rinitializer(theta, keys, covars0, t0)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0.0
    # prepare arrays to store diagnostics if requested
    n_obs = len(ys)
    CLL_arr = jnp.zeros(n_obs) if CLL else jnp.zeros(0)
    ESS_arr = jnp.zeros(n_obs) if ESS else jnp.zeros(0)
    filter_mean_arr = (
        jnp.zeros((n_obs, particlesF.shape[-1]))
        if filter_mean
        else jnp.zeros((0, particlesF.shape[-1]))
    )
    prediction_mean_arr = (
        jnp.zeros((n_obs, particlesF.shape[-1]))
        if prediction_mean
        else jnp.zeros((0, particlesF.shape[-1]))
    )

    pfilter_helper_2 = partial(
        _pfilter_helper,
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        times=times,
        theta=theta,
        rprocess_interp=rprocess_interp,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        thresh=thresh,
        accumvars=accumvars,
        CLL=CLL,
        ESS=ESS,
        filter_mean=filter_mean,
        prediction_mean=prediction_mean,
    )
    (
        t,
        particlesF,
        loglik,
        norm_weights,
        counts,
        key,
        t_idx,
        CLL_arr,
        ESS_arr,
        filter_mean_arr,
        prediction_mean_arr,
    ) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=pfilter_helper_2,
        init_val=(
            t0,
            particlesF,
            loglik,
            norm_weights,
            counts,
            key,
            0,
            CLL_arr,
            ESS_arr,
            filter_mean_arr,
            prediction_mean_arr,
        ),
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
    in_axes=(None,) * 13 + (0,) + (None,) * 4,
)

# Map over theta and key
_vmapped_pfilter_internal2 = jax.vmap(
    _pfilter_internal,
    in_axes=(0,) + (None,) * 12 + (0,) + (None,) * 4,
)


@partial(jit, static_argnums=(5, 6, 7, 8, 9))
def _pfilter_internal_mean(
    theta: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
) -> jax.Array:
    """
    Internal function for calculating the particle filter estimate of the negative log
    likelihood divided by the length of the observations. This is used in internal
    pypomp.train functions.
    """
    return (
        _pfilter_internal(
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
            thresh=thresh,
            key=key,
            CLL=False,
            ESS=False,
            filter_mean=False,
            prediction_mean=False,
        )["neg_loglik"]
        / ys.shape[0]
    )


def _pfilter_helper(
    i: int,
    inputs: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        int,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    times: jax.Array,
    theta: jax.Array,
    rprocess_interp: Callable,
    dmeasure: Callable,
    covars_extended: jax.Array | None,
    thresh: float,
    CLL: bool,
    ESS: bool,
    filter_mean: bool,
    prediction_mean: bool,
    accumvars: tuple[int, ...] | None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    int,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """
    Helper function for the particle filtering algorithm in POMP, which conducts
    filtering for one time-iteration.
    Only update the diagnostics elements when their corresponding boolean elements are set to be TRUE
    """
    (
        t,
        particlesF,
        loglik,
        norm_weights,
        counts,
        key,
        t_idx,
        CLL_arr,
        ESS_arr,
        filter_mean_arr,
        prediction_mean_arr,
    ) = inputs
    J = len(particlesF)

    # Propagate from times[i] to times[i+1] using the interpolator
    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    nstep = nstep_array[i]
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

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik = loglik + loglik_t

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

    return (
        t,
        particlesF,
        loglik,
        norm_weights,
        counts,
        key,
        t_idx,
        CLL_arr,
        ESS_arr,
        filter_mean_arr,
        prediction_mean_arr,
    )

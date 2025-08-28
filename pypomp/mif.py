from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable
from .internal_functions import _normalize_weights
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas
from .internal_functions import _geometric_cooling
from .parameter_trans import ParTrans, _pt_forward, _pt_inverse


def _mif_internal(
    theta: jax.Array,
    dt_array_extended: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    partrans: ParTrans,
    M: int,
    a: float,
    J: int,
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logliks = jnp.zeros(M)
    params = jnp.zeros((M, J, theta.shape[-1]))
    params = jnp.concatenate([theta.reshape((1, J, theta.shape[-1])), params], axis=0)

    _perfilter_internal_2 = partial(
        _perfilter_internal,
        dt_array_extended=dt_array_extended,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        rinitializers=rinitializers,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        accumvars=accumvars,
        covars_extended=covars_extended,
        partrans=partrans,
        thresh=thresh,
        a=a,
    )

    (params, logliks, key) = jax.lax.fori_loop(
        lower=0,
        upper=M,
        body_fun=_perfilter_internal_2,
        init_val=(params, logliks, key),
    )
    return logliks, params


_jit_mif_internal = jit(_mif_internal, static_argnums=(6, 7, 8, 13, 14, 16))

_vmapped_mif_internal = jax.vmap(
    _mif_internal,
    in_axes=(1,) + (None,) * 17 + (0,),
)

_jv_mif_internal = jit(_vmapped_mif_internal, static_argnums=(6, 7, 8, 13, 14, 16))


def _perfilter_internal(
    m: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    dt_array_extended: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    partrans: ParTrans,
    thresh: float,
    a: float,
):
    (params, logliks, key) = inputs
    thetas = _pt_forward(params[m], partrans)
    loglik = 0.0
    key, subkey = jax.random.split(key)
    sigmas_init_cooled = (
        _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init
    )
    thetas = thetas + sigmas_init_cooled * jax.random.normal(
        shape=thetas.shape, key=subkey
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    thetas_nat0 = _pt_inverse(thetas, partrans)
    particlesF = rinitializers(thetas_nat0, keys, covars0, t0)

    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)

    perfilter_helper_2 = partial(
        _perfilter_helper,
        dt_array_extended=dt_array_extended,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        sigmas=sigmas,
        accumvars=accumvars,
        covars_extended=covars_extended,
        partrans=partrans,
        thresh=thresh,
        m=m,
        a=a,
    )
    (t, particlesF, thetas, loglik, norm_weights, counts, key) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys_extended),
        body_fun=perfilter_helper_2,
        init_val=(t0, particlesF, thetas, loglik, norm_weights, counts, key),
    )

    logliks = logliks.at[m + 1].set(-loglik)
    thetas_nat_end = _pt_inverse(thetas, partrans)
    params = params.at[m + 1].set(thetas_nat_end)
    return params, logliks, key


def _perfilter_helper(
    i: int,
    inputs: tuple[
        jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
    ],
    dt_array_extended: jax.Array,
    times0: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    partrans: ParTrans,
    thresh: float,
    m: int,
    a: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    (t, particlesF, thetas, loglik, norm_weights, counts, key) = inputs
    J = len(particlesF)

    sigmas_cooled = (
        _geometric_cooling(nt=i + 1, m=m, ntimes=len(times0) - 1, a=a) * sigmas
    )
    key, subkey = jax.random.split(key)
    thetas = thetas + sigmas_cooled * jnp.array(
        jax.random.normal(shape=thetas.shape, key=subkey)
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars_t = None if covars_extended is None else covars_extended[i]
    thetas_nat = _pt_inverse(thetas, partrans)
    particlesP = rprocesses(particlesF, thetas_nat, keys, covars_t, t, dt_array_extended[i])
    t = t + dt_array_extended[i]

    def _with_observation(loglik, norm_weights, counts, thetas, key, dmeasures):
        covars_t = None if covars_extended is None else covars_extended[i + 1]
        measurements = jnp.nan_to_num(
            dmeasures(ys_extended[i], particlesP, thetas_nat, covars_t, t).squeeze(),
            nan=jnp.log(1e-18),
        )

        if len(measurements.shape) > 1:
            measurements = measurements.sum(axis=-1)

        weights = norm_weights + measurements
        norm_weights, loglik_t = _normalize_weights(weights)

        loglik = loglik + loglik_t
        oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
        key, subkey = jax.random.split(key)
        counts, particlesF2, norm_weights, thetas2 = jax.lax.cond(
            oddr > thresh,
            _resampler_thetas,
            _no_resampler_thetas,
            *(counts, particlesP, norm_weights, thetas, subkey),
        )
        particlesF2 = jnp.where(
            accumvars is not None, particlesF2.at[:, accumvars].set(0.0), particlesF2
        )
        return (particlesF2, loglik, norm_weights, counts, thetas2, key)

    def _without_observation(loglik, norm_weights, counts, thetas, key):
        return (particlesP, loglik, norm_weights, counts, thetas, key)

    _with_observation_partial = partial(_with_observation, dmeasures=dmeasures)

    particles, loglik, norm_weights, counts, thetas, key = jax.lax.cond(
        ys_observed[i],
        _with_observation_partial,
        _without_observation,
        *(loglik, norm_weights, counts, thetas, key),
    )

    return (t, particles, thetas, loglik, norm_weights, counts, key)


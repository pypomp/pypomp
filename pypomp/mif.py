from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable
from .internal_functions import _normalize_weights
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas
from .internal_functions import _interp_covars
from .internal_functions import _geometric_cooling


def _mif_internal(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializers: Callable,  # static
    rprocesses: Callable,  # static
    dmeasures: Callable,  # static
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    M: int,  # static
    a: float,
    J: int,  # static
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logliks = jnp.zeros(M)
    params = jnp.zeros((M, J, theta.shape[-1]))

    params = jnp.concatenate([theta.reshape((1, J, theta.shape[-1])), params], axis=0)

    _perfilter_internal_2 = partial(
        _perfilter_internal,
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        rinitializers=rinitializers,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        ctimes=ctimes,
        covars=covars,
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


_jit_mif_internal = jit(_mif_internal, static_argnums=(4, 5, 6, 11, 13))

_vmapped_mif_internal = jax.vmap(
    _mif_internal,
    in_axes=(1,) + (None,) * 14 + (0,),
)

_jv_mif_internal = jit(_vmapped_mif_internal, static_argnums=(4, 5, 6, 11, 13))


def _perfilter_internal(
    m: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    a: float,
):
    """
    Internal function for the perturbed particle filtering algorithm, which calls
    function 'perfilter_helper' iteratively.
    """
    (params, logliks, key) = inputs
    thetas = params[m]
    loglik = 0
    key, subkey = jax.random.split(key)
    sigmas_init_cooled = (
        _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init
    )
    thetas = thetas + sigmas_init_cooled * jax.random.normal(
        shape=thetas.shape, key=subkey
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    particlesF = rinitializers(thetas, keys, covars_t, t0)

    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)

    perfilter_helper_2 = partial(
        _perfilter_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        ys=ys,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        sigmas=sigmas,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        m=m,
        a=a,
    )
    (particlesF, thetas, loglik, norm_weights, counts, key) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=perfilter_helper_2,
        init_val=(particlesF, thetas, loglik, norm_weights, counts, key),
    )

    logliks = logliks.at[m + 1].set(-loglik)
    params = params.at[m + 1].set(thetas)
    return params, logliks, key


def _perfilter_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    ys: jax.Array,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    m: int,
    a: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Helper functions for perturbed particle filtering algorithm, which conducts
    a single iteration of filtering and is called in function
    'perfilter_internal'.
    """
    (particlesF, thetas, loglik, norm_weights, counts, key) = inputs
    J = len(particlesF)
    t1 = times0[i]
    t2 = times0[i + 1]

    sigmas_cooled = _geometric_cooling(nt=i + 1, m=m, ntimes=len(ys), a=a) * sigmas
    key, subkey = jax.random.split(key)
    thetas = thetas + sigmas_cooled * jnp.array(
        jax.random.normal(shape=thetas.shape, key=subkey)
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesP = rprocesses(particlesF, thetas, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    measurements = jnp.nan_to_num(
        dmeasures(ys[i], particlesP, thetas, covars_t, t2).squeeze(), nan=jnp.log(1e-18)
    )  # shape (Np,)

    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)

    loglik += loglik_t
    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights, thetas = jax.lax.cond(
        oddr > thresh,
        _resampler_thetas,
        _no_resampler_thetas,
        *(counts, particlesP, norm_weights, thetas, subkey),
    )

    return (particlesF, thetas, loglik, norm_weights, counts, key)

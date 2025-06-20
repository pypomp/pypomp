from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
from typing import Callable
from .pfilter import _pfilter_internal
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
    rinitializer: Callable,
    rprocess: Callable,
    dmeasure: Callable,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    M: int,
    a: float,
    J: int,
    thresh: float,
    verbose: bool,
    key: jax.Array,
    particle_thetas: bool,  # set true when theta already contains a row for each particle
) -> tuple[jax.Array, jax.Array]:
    logliks = []
    params = []

    if particle_thetas:
        ndim = theta.ndim - 1
        thetas = theta
    else:
        ndim = theta.ndim
        thetas = jnp.tile(theta, (J,) + (1,) * ndim)
    params.append(thetas)

    for m in tqdm(range(M)):
        key, subkey = jax.random.split(key=key)
        loglik_ext, thetas = _perfilter_internal(
            thetas=thetas,
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
            key=subkey,
            m=m,
            a=a,
        )
        params.append(thetas)
        logliks.append(loglik_ext)

        if verbose:
            print(loglik_ext)
            print(thetas.mean(0))

    return jnp.array(logliks), jnp.array(params)


@partial(jit, static_argnums=(4, 7, 8, 9))
def _perfilter_internal(
    thetas: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    sigmas: jax.Array,
    sigmas_init: jax.Array,
    rinitializers: Callable,  # static
    rprocesses: Callable,  # static
    dmeasures: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
    m: int,
    a: float,
):
    """
    Internal function for the perturbed particle filtering algorithm, which calls
    function 'perfilter_helper' iteratively.
    """
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

    return -loglik, thetas


def _perfilter_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    ys: jax.Array,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: jax.Array,
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

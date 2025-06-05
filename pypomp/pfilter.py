from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights


def pfilter(
    J,
    rinit,
    rproc,
    dmeas,
    theta,
    ys,
    key,
    covars=None,
    thresh=0,
):
    """
    An outside function for the particle filtering algorithm.

    Args:
        J (int): The number of particles.
        rinit (RInit): Simulator for the initial-state distribution.
        rproc (RProc): Simulator for the process model.
        dmeas (DMeas): Density evaluation for the measurement model.
        theta (dict): Parameters involved in the POMP model. Each value must be a float.
        ys (array-like): The measurement array.
        key (jax.random.PRNGKey): The random key.
        covars (array-like, optional): Covariates or None if not applicable.
        thresh (float, optional): Threshold value to determine whether to
            resample particles.

    Raises:
        ValueError: Missing the pomp class object and required arguments for
            calling 'pfilter_internal' directly

    Returns:
        float: The log-likelihood estimate
    """
    if J < 1:
        raise ValueError("J should be greater than 0.")
    if rinit is None or rproc is None or dmeas is None or ys is None:
        raise ValueError("Missing rinit, rproc, dmeas, theta, or ys.")

    if not isinstance(theta, dict):
        raise TypeError("theta must be a dictionary")
    if not all(isinstance(val, float) for val in theta.values()):
        raise TypeError("Each value of theta must be a float")

    return -_pfilter_internal(
        theta=jnp.array(list(theta.values())),
        ys=jnp.array(ys),
        J=J,
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        dmeasure=dmeas.struct_pf,
        covars=jnp.array(covars) if covars is not None else None,
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_internal(
    theta: jax.Array,
    ys: jax.Array,
    J: int,
    rinitializer: callable,
    rprocess: callable,
    dmeasure: callable,
    covars: jax.Array,
    thresh: float,
    key: jax.Array,
):
    """
    Internal functions for particle filtering algorithm, which calls function
    'pfilter_helper' iteratively.
    """
    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesF = rinitializer(theta, keys, covars)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0

    pfilter_helper_2 = partial(_pfilter_helper, rprocess=rprocess, dmeasure=dmeasure)
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = (
        jax.lax.fori_loop(
            lower=0,
            upper=len(ys),
            body_fun=pfilter_helper_2,
            init_val=[
                particlesF,
                theta,
                covars,
                loglik,
                norm_weights,
                counts,
                ys,
                thresh,
                key,
            ],
        )
    )

    return -loglik


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_internal_mean(
    theta, ys, J, rinitializer, rprocess, dmeasure, covars, thresh, key
):
    return _pfilter_internal(
        theta, ys, J, rinitializer, rprocess, dmeasure, covars, thresh, key
    ) / len(ys)


def _pfilter_helper(
    t: int,
    inputs: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        float,
        jax.Array,
        jax.Array,
        jax.Array,
        float,
        jax.Array,
    ],
    rprocess: callable,
    dmeasure: callable,
) -> tuple:
    """
    Helper functions for particle filtering algorithm in POMP, which conducts a
    single iteration of filtering.
    """
    (particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key) = inputs
    J = len(particlesF)

    key, keys = _keys_helper(key=key, J=J, covars=covars)

    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)  # if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys, None)

    measurements = dmeasure(ys[t], particlesP, theta, covars)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements

    norm_weights, loglik_t = _normalize_weights(weights)
    loglik += loglik_t

    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights = jax.lax.cond(
        oddr > thresh,
        _resampler,
        _no_resampler,
        counts,
        particlesP,
        norm_weights,
        subkey,
    )

    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]

from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights


def pfilter(
    pomp_object=None,
    J=50,
    rinit=None,
    rproc=None,
    dmeas=None,
    theta=None,
    ys=None,
    covars=None,
    thresh=100,
    key=None,
):
    """
    An outside function for particle filtering algorithm. It receives two kinds
    of input - pomp class object or required arguments of 'pfilter_internal'
    function, and executes on the object or calls 'pfilter_internal' directly.

    Args:
        pomp_object (Pomp, optional): An instance of the POMP class. If
            provided, the function will execute on this object to conduct the
            particle filtering algorithm. Defaults to None.
        J (int, optional): The number of particles. Defaults to 50.
        rinit (RInit, optional): Simulator for the initial-state
            distribution. Defaults to None.
        rprocess (RProc, optional): Simulator for the process model. Defaults
            to None.
        dmeasure (DMeas, optional): Density evaluation for the measurement
            model. Defaults to None.
        theta (array-like, optional): Parameters involved in the POMP model.
            Defaults to None.
        ys (array-like, optional): The measurement array. Defaults to None.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        thresh (float, optional):Threshold value to determine whether to
            resample particles. Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Raises:
        ValueError: Missing the pomp class object and required arguments for
            calling 'pfilter_internal' directly

    Returns:
        float: Negative log-likelihood value
    """
    if J < 1:
        raise ValueError("J should be greater than 0")
    if pomp_object is not None:
        return pomp_object.pfilter(J, thresh, key)
    elif (
        rinit is not None
        and rproc is not None
        and dmeas is not None
        and theta is not None
        and ys is not None
    ):
        return _pfilter_internal(
            theta=theta,
            ys=ys,
            J=J,
            rinitializer=rinit.struct_pf,
            rprocess=rproc.struct_pf,
            dmeasure=dmeas.struct_pf,
            covars=covars,
            thresh=thresh,
            key=key,
        )
    else:
        raise ValueError("Invalid Arguments Input")


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_internal(
    theta, ys, J, rinitializer, rprocess, dmeasure, covars, thresh, key
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


def _pfilter_helper(t, inputs, rprocess, dmeasure):
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

    measurements = dmeasure(ys[t], particlesP, theta)
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

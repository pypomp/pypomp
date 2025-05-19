"""
This module implements the MOP algorithm for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights
from .internal_functions import _resampler


def mop(
    pomp_object=None,
    J=50,
    rinit=None,
    rproc=None,
    dmeas=None,
    theta=None,
    ys=None,
    covars=None,
    alpha=0.97,
    key=None,
):
    """
    An outside function for MOP algorithm. It receives two kinds of input
    - pomp class object or the required arguments of 'mop_internal'
    function, and executes on the object or calls 'mop_internal' directly.

    Args:
        pomp_object (Pomp, optional): An instance of the POMP class. If
            provided, the function will execute on this object to conduct
            the MOP algorithm. Defaults to None.
        J (int, optional): The number of particles. Defaults to 50.
        rinit (RInit, optional): Simulator for the initial-state
            distribution. Defaults to None.
        rproc (RProc, optional): Simulator for the process model.
            Defaults to None.
        dmeas (DMeas, optional): Density evaluation for the measurement
            model. Defaults to None.
        theta (array-like, optional): Parameters involved in the POMP
            model. Defaults to None.
        ys (array-like, optional): The measurement array. Defaults to
            None.
        covars (array-like, optional): Covariates or None if not
            applicable. Defaults to None.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to
            None.

    Raises:
        ValueError: Missing the pomp class object and required arguments
            for calling 'mop_internal' directly

    Returns:
        float: Negative log-likelihood value
    """
    if pomp_object is not None:
        return pomp_object.mop(J, alpha, key=key)
    elif (
        rinit is not None
        and rproc is not None
        and dmeas is not None
        and theta is not None
        and ys is not None
    ):
        return _mop_internal(
            theta=theta,
            ys=ys,
            J=J,
            rinit=rinit.struct,
            rprocess=rproc.struct_pf,
            dmeasure=dmeas.struct_pf,
            covars=covars,
            alpha=alpha,
            key=key,
        )
    else:
        raise ValueError("Invalid Arguments Input")


@partial(jit, static_argnums=(2, 3, 4, 5))
def _mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key):
    """
    Internal function for MOP algorithm, which calls function 'mop_helper'
    iteratively.
    """
    # if key is None:
    # key = jax.random.PRNGKey(np.random.choice(int(1e18)))

    particlesF = rinit(theta, J, covars=covars)
    weightsF = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0

    mop_helper_2 = partial(_mop_helper, rprocess=rprocess, dmeasure=dmeasure)

    (particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key) = (
        jax.lax.fori_loop(
            lower=0,
            upper=len(ys),
            body_fun=mop_helper_2,
            init_val=[
                particlesF,
                theta,
                covars,
                loglik,
                weightsF,
                counts,
                ys,
                alpha,
                key,
            ],
        )
    )

    return -loglik


@partial(jit, static_argnums=(2, 3, 4, 5))
def _mop_internal_mean(
    theta, ys, J, rinit, rprocess, dmeasure, covars=None, alpha=0.97, key=None
):
    """
    Internal function for calculating the mean result using MOP algorithm
    across the measurements.
    """
    return _mop_internal(
        theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key
    ) / len(ys)


def _mop_helper(t, inputs, rprocess, dmeasure):
    """
    Helper functions for MOP algorithm, which conducts a single iteration of
    filtering and is called in function 'mop_internal'.
    """
    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = inputs
    J = len(particlesF)

    key, keys = _keys_helper(key=key, J=J, covars=covars)

    weightsP = alpha * weightsF

    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)
    else:
        particlesP = rprocess(particlesF, theta, keys, None)

    measurements = dmeasure(ys[t], particlesP, theta)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    loglik += jax.scipy.special.logsumexp(
        weightsP + measurements
    ) - jax.scipy.special.logsumexp(weightsP)
    # test different, logsumexp - source code (floating point arithmetic issue)
    # make a little note in the code, discuss it in the quant test about the small difference
    # logsumexp source code

    (norm_weights, loglik_phi_t) = _normalize_weights(
        jax.lax.stop_gradient(measurements)
    )

    key, subkey = jax.random.split(key)
    (counts, particlesF, norm_weightsF) = _resampler(
        counts, particlesP, norm_weights, subkey=subkey
    )

    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    return [particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key]

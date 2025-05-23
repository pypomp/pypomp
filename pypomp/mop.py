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
    J,
    rinit,
    rproc,
    dmeas,
    theta,
    ys,
    key,
    covars=None,
    alpha=0.97,
):
    """
    An outside function for the MOP algorithm.

    Args:
        J (int, optional): The number of particles.
        rinit (RInit, optional): Simulator for the initial-state
            distribution.
        rproc (RProc, optional): Simulator for the process model.
        dmeas (DMeas, optional): Density evaluation for the measurement
            model.
        theta (array-like, optional): Parameters involved in the POMP
            model.
        ys (array-like, optional): The measurement array.
        covars (array-like, optional): Covariates or None if not
            applicable.
        alpha (float, optional): Discount factor.
        key (jax.random.PRNGKey, optional): The random key.

    Returns:
        float: The log-likelihood estimate
    """
    if J < 1:
        raise ValueError("J should be greater than 0")

    if not isinstance(theta, dict):
        raise TypeError("theta must be a dictionary")
    if not all(isinstance(val, float) for val in theta.values()):
        raise TypeError("Each value of theta must be a float")

    return -_mop_internal(
        theta=jnp.array(list(theta.values())),
        ys=jnp.array(ys),
        J=J,
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        dmeasure=dmeas.struct_pf,
        covars=jnp.array(covars) if covars is not None else None,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _mop_internal(theta, ys, J, rinitializer, rprocess, dmeasure, covars, alpha, key):
    """
    Internal function for MOP algorithm, which calls function 'mop_helper'
    iteratively.
    """
    # if key is None:
    # key = jax.random.PRNGKey(np.random.choice(int(1e18)))

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesF = rinitializer(theta, keys, covars)
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

    measurements = dmeasure(ys[t], particlesP, theta, covars)
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

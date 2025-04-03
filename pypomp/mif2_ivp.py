import jax
import pypomp
import jax.numpy as jnp
import numpy as np
from jax import jit
from tqdm import tqdm
from functools import partial

import pypomp.pfilter
import pypomp.internal_functions 
from pypomp.internal_functions import _normalize_weights
from pypomp.internal_functions import _pfilter_internal
from pypomp.internal_functions import _rinits_internal
from pypomp.internal_functions import _resampler_thetas
from pypomp.internal_functions import _no_resampler_thetas                                                   

MONITORS = 1

def _perfilter_helper(t, inputs, rprocesses, dmeasures):

    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)

    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = jnp.array(keys).reshape(J, covars.shape[1], 2).astype(jnp.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = jnp.array(keys)

    thetas += sigmas * jnp.array(np.random.normal(size=thetas.shape))

    # Get prediction particles
    # r processes: particleF and thetas are both vectorized (J times)
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars)  # if t>0 else particlesF
    else:
        particlesP = rprocesses(particlesF, thetas, keys, None)  # if t>0 else particlesF

    measurements = jnp.nan_to_num(dmeasures(ys[t], particlesP, thetas).squeeze(),
                                  nan=jnp.log(1e-18))  # shape (Np,)

    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)

    loglik += loglik_t
    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    counts, particlesF, norm_weights, thetas = jax.lax.cond(oddr > thresh,
                                                            _resampler_thetas,
                                                            _no_resampler_thetas,
                                                            counts, particlesP, norm_weights, 
                                                            thetas)

    return [particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key]


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, ndim, covars=None, 
                        thresh=100, key=None):
    
    loglik = 0
    # not remove
    thetas = theta + sigmas * np.random.normal(size=(J,) + theta.shape[-ndim:])
    # thetas = theta + sigmas * onp.random.normal(size=(J,) + theta.shape[1:])
    particlesF = _rinits_internal(rinit, thetas, 1, covars=covars)
    weights = jnp.log(jnp.ones(J) / J)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    if key is None:
        key = jax.random.PRNGKey(np.random.choice(int(1e18)))
    perfilter_helper_2 = partial(_perfilter_helper, rprocesses=rprocesses, dmeasures=dmeasures)
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = \
    jax.lax.fori_loop(lower=0, upper=len(ys), body_fun=perfilter_helper_2,
        init_val=[particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, 
                  thresh, key])

    return -loglik, thetas

def _perfilter_helper_ivp(t, inputs, rprocesses, dmeasures):
   
    particlesF, thetas, sigmas, a, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)

    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = jnp.array(keys).reshape(J, covars.shape[1], 2).astype(jnp.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = jnp.array(keys)

    # thetas_P - cooling fraction 
    sigmas = sigmas * a # add cooling fraction here
    thetas += sigmas * jnp.array(np.random.normal(size=thetas.shape))

    # Get prediction particles
    # r processes: particleF and thetas are both vectorized (J times)
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars)  # if t>0 else particlesF
    else:
        particlesP = rprocesses(particlesF, thetas, keys, None)  # if t>0 else particlesF

    measurements = jnp.nan_to_num(dmeasures(ys[t], particlesP, thetas).squeeze(),
                                  nan=jnp.log(1e-18))  # shape (Np,)

    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)

    loglik += loglik_t
    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    counts, particlesF, norm_weights, thetas = jax.lax.cond(oddr > thresh,
                                                            _resampler_thetas,
                                                            _no_resampler_thetas,
                                                            counts, particlesP, norm_weights, 
                                                            thetas)

    return [particlesF, thetas, sigmas, a, covars, loglik, norm_weights, counts, ys, thresh, key]


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal_ivp(thetas, ys, J, sigmas, a, rinit, rprocesses, dmeasures, ndim, covars=None, 
                        thresh=100, key=None):
    
    loglik = 0
    # thetas = theta + sigmas * np.random.normal(size=(J,) + theta.shape[-ndim:]) # remove 
    particlesF = _rinits_internal(rinit, thetas, 1, covars=covars)
    weights = jnp.log(jnp.ones(J) / J)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    if key is None:
        key = jax.random.PRNGKey(np.random.choice(int(1e18)))
    perfilter_helper_2 = partial(_perfilter_helper_ivp, rprocesses=rprocesses, dmeasures=dmeasures)
    particlesF, thetas, sigmas, a, covars, loglik, norm_weights, counts, ys, thresh, key = \
    jax.lax.fori_loop(lower=0, upper=len(ys), body_fun=perfilter_helper_2,
        init_val=[particlesF, thetas, sigmas, a, covars, loglik, norm_weights, counts, ys, 
                  thresh, key])

    return -loglik, thetas

def _mif_internal_ivp(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, 
                  sigmas_init, covars=None, M=10, a=0.95, J=100, thresh=100, monitor=False,
                  verbose=False, key=jax.random.PRNGKey(0)):

    logliks = []
    params = []
    ndim = theta.ndim
    thetas = jnp.tile(theta, (J,) + (1,) * theta.ndim)
    params.append(thetas)
    if monitor:
        loglik = jnp.mean(
            jnp.array([_pfilter_internal(thetas.mean(0), ys, J, rinit, rprocess, dmeasure,
                                         covars=covars, thresh=thresh)
                       for i in range(MONITORS)]))
        logliks.append(loglik)

    for m in tqdm(range(M)):
        # sigmas_init - IVP 
        sigmas_init *= a # cooling fraction
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(key=subkey, shape=(J,) + theta.shape[-ndim:])
        thetas = thetas + sigmas_init * noise
        
        # update sigmas for filtering
        sigmas *= a
        
        key, subkey = jax.random.split(key)
        # thetas += sigmas * np.random.normal(size=thetas.shape)
        loglik_ext, thetas = _perfilter_internal(thetas, ys, J, sigmas, rinit, rprocesses,
                                                 dmeasures, ndim=ndim, covars=covars, 
                                                 thresh=thresh, key=subkey)

        params.append(thetas)
        key, subkey = jax.random.split(key)

        if monitor:
            loglik = jnp.mean(jnp.array(
                [_pfilter_internal(thetas.mean(0), ys, J, rinit, rprocess, dmeasure, 
                                   covars=covars, thresh=thresh, key=subkey)
                 for i in range(MONITORS)]))

            logliks.append(loglik)

            if verbose:
                print(loglik)
                print(thetas.mean(0))

    return jnp.array(logliks), jnp.array(params)

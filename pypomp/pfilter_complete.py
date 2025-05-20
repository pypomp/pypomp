"""
pfilter_complete is an adaptation of pfilter that can return states
(i.e., values for each particle), filter mean, effective sample size.

This is designed for diagnostics and other data analysis investigations.
It is not designed for performance. Use pfilter within inference algorithms.

At some future point, pfilter_complete might be incorporated within
pfilter.

Currently, pfilter_complete is somewhat experimental, reflecting the
initial emphasis on methodology over data analysis for pypomp.
"""

import jax
import numpy as np
import jax.numpy as jnp

from functools import partial
from pypomp.internal_functions import _normalize_weights, _resampler, _no_resampler


def _pfilter_helper_complete(t, inputs, rprocess, dmeasure):
    [
        particlesF,
        theta,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
        logliks_arr,
        particles_arr,
        filter_mean_arr,
        ess_arr,
        traj_arr,
    ] = inputs
    J = len(particlesF)

    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = jnp.array(keys).reshape(J, covars.shape[1], 2).astype(jnp.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = jnp.array(keys)

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

    # save loglik_t (conditional log-likelihood) particlesP (states)
    logliks_arr = logliks_arr.at[t].set(loglik_t)
    particles_arr = particles_arr.at[t].set(particlesP)

    # calculate filtering mean (filt.mean)
    # \sum (w_j * particle_j)
    filter_mean_t = (particlesP * jnp.exp(norm_weights[:, None])).sum(axis=0)
    filter_mean_arr = filter_mean_arr.at[t].set(filter_mean_t)

    # create filter.traj
    # variable "counts" records the source particle of particle i (1 - J) at
    # time t from the previous step
    traj_arr = traj_arr.at[t].set(counts)

    # calculate effective sample size (ess)
    ess_t = 1.0 / jnp.sum(jnp.exp(norm_weights) ** 2)
    ess_arr = ess_arr.at[t].set(ess_t)

    key, subkey = jax.random.split(key)
    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    counts, particlesF, norm_weights = jax.lax.cond(
        oddr > thresh,
        _resampler,
        _no_resampler,
        counts,
        particlesP,
        norm_weights,
        subkey,
    )

    return [
        particlesF,
        theta,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
        logliks_arr,
        particles_arr,
        filter_mean_arr,
        ess_arr,
        traj_arr,
    ]


def _pfilter_internal_complete(
    theta, ys, J, rinit, rprocess, dmeasure, covars=None, thresh=100, key=None
):
    if key is None:
        key = jax.random.PRNGKey(np.random.choice(int(1e18)))

    particlesF = rinit(theta, J, covars=covars)
    weights = jnp.log(jnp.ones(J) / J)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0

    logliks_arr = jnp.zeros(len(ys))  # save loglik_t (conditional log-likelihood)
    particles_arr = jnp.zeros((len(ys), J, particlesF.shape[-1]))  # save states
    filter_mean_arr = jnp.zeros((len(ys), particlesF.shape[-1]))
    ess_arr = jnp.zeros(len(ys))
    traj_arr = jnp.zeros((len(ys), J))

    pfilter_helper_2 = partial(
        _pfilter_helper_complete, rprocess=rprocess, dmeasure=dmeasure
    )
    [
        particlesF,
        theta,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
        logliks_arr,
        particles_arr,
        filter_mean_arr,
        ess_arr,
        traj_arr,
    ] = jax.lax.fori_loop(
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
            logliks_arr,
            particles_arr,
            filter_mean_arr,
            ess_arr,
            traj_arr,
        ],
    )

    # filt.t:
    b = jax.random.choice(key, jnp.arange(J), p=jnp.exp(norm_weights))
    filt_traj = jnp.zeros((len(ys), particlesF.shape[-1]))
    for t in range(len(ys) - 1, -1, -1):
        # extract the state value corresponding the b th particle at time t
        # and plug it into filt_traj
        filt_traj = filt_traj.at[t].set(particles_arr[t, b])
        # find the corresponding particle from the previous step from traj_arr
        # and update b
        b = traj_arr[t, b]
        b = jnp.int32(b)

    # return 1. negative log-likelihood,
    # 2. mean log-likelihood,
    # 3. conditional log-likelihood
    # 4. save states
    # 5. filter mean
    # 6. effective sample size
    # 7. filter_traj
    return [
        -loglik,
        -loglik / len(ys),
        logliks_arr,
        particles_arr,
        filter_mean_arr,
        ess_arr,
        filt_traj,
    ]


def pfilter_complete(
    pomp_object=None,
    J=50,
    rinit=None,
    rprocess=None,
    dmeasure=None,
    theta=None,
    ys=None,
    covars=None,
    thresh=100,
    key=None,
):
    if pomp_object is not None:
        return pomp_object.pfilter(J, thresh, key)
    elif (
        rinit is not None
        and rprocess is not None
        and dmeasure is not None
        and theta is not None
        and ys is not None
    ):
        return _pfilter_internal_complete(
            theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key
        )
    else:
        raise ValueError("Invalid Arguments Input")

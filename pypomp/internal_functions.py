"""
This module implements internal functions for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

# from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

# tfd = tfp.distributions
# tfb = tfp.bijectors
# tfpk = tfp.math.psd_kernels

"""resampling functions"""


def _keys_helper(key, J, covars):
    """
    This function is a helper for generating random keys for resampling in the
    particle filtering algorithms.
    """
    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = jnp.array(keys).reshape(J, covars.shape[1], 2).astype(jnp.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = jnp.array(keys)
    return key, keys


def _rinits_internal(rinit, thetas, J, covars):
    """
    Simulator for the initial-state distribution, specifically for the perturbed
    particle filtering method.

    Args:
        rinit (function): Simulator for the initial-state distribution for the
            unperturbed particle filtering method.
        thetas (array-like): Array of parameters used in the likelihood-based
            inference.
        J (int): The number of particles.
        covars (array-like or None): Covariates or None if not applicable.

    Returns:
        array-like: The simulated initial latent states.
    """
    return rinit(thetas[0], len(thetas), covars)


def _resample(norm_weights, subkey):
    """
    Systematic resampling method based on input normalized weights.

    Args:
        norm_weights (array-like): The array containing the logarithm of
            normalized weights.
        subkey (jax.random.PRNGKey): The random key for sampling.

    Returns:
        array-like: An array containing the resampled indices from the
            systematic resampling given the input normalized weights.
    """
    J = norm_weights.shape[-1]
    unifs = (jax.random.uniform(key=subkey) + jnp.arange(J)) / J
    csum = jnp.cumsum(jnp.exp(norm_weights))
    counts = jnp.repeat(
        jnp.arange(J),
        jnp.histogram(
            unifs, bins=jnp.pad(csum / csum[-1], pad_width=(1, 0)), density=False
        )[0].astype(int),
        total_repeat_length=J,
    )
    return counts


def _normalize_weights(weights):
    """
    Acquires the normalized log-weights and calculates the log-likelihood.

    Args:
        weights (array-like): Logarithm of unnormalized weights.

    Returns:
        tuple: A tuple containing:
            - norm_weights (array-like): The normalized log-weights.
            - loglik_t (float): The log of the sum of all particle likelihoods,
                when the weights are associate with particles, which is
                equivalent to the total log-likelihood under the specific
                assumptions.
    """
    mw = jnp.max(weights)
    loglik_t = mw + jnp.log(jnp.nansum(jnp.exp(weights - mw)))
    norm_weights = weights - loglik_t
    return norm_weights, loglik_t


def _resampler(counts, particlesP, norm_weights, subkey):
    """
    Resamples the particles based on the weighted resampling rule determined by
    norm_weights and the original particles generated from the previous
    prediction.

    Args:
        counts (array-like): Indices of the resampled particles from a previous
            resampling procedure.
        particlesP (array-like): The original particles before resampling
            generated from a prediction procedure.
        norm_weights (array-like): The normalized log-weights of the particles.
        subkey (jax.random.PRNGKey): The random key for sampling.

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the resampled particles after
                the latest resampling.
            - particlesF (array-like): The particles after resampling generated
                from the filtering procedure.
            - norm_weights (array-like): The normalized log-weights of the
                resampled particles.
    """
    J = norm_weights.shape[-1]
    counts = _resample(norm_weights, subkey=subkey)
    particlesF = particlesP[counts]
    norm_weights = (
        norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - jnp.log(J)
    )
    return counts, particlesF, norm_weights


def _no_resampler(counts, particlesP, norm_weights, subkey):
    """
    Obtains the original input arguments without resampling.

    Args:
        counts (array-like): Indices of the resampled particles from a previous
            resampling procedure.
        particlesP (array-like): The original particles before resampling
            generated from a prediction procedure.
        norm_weights (array-like): The normalized log-weights of the particles.
        subkey (jax.random.PRNGKey): The random key for sampling. This is not
            used in this function, but is included to maintain the same
            arguments as _resampler().

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the particles, unchanged.
            - particlesP (array-like): The original particles, unchanged.
            - norm_weights (array-like): The normalized log-weights, unchanged.
    """
    return counts, particlesP, norm_weights


def _resampler_thetas(counts, particlesP, norm_weights, thetas, subkey):
    """
    Resamples the particles for perturbed particle filtering method, with their
    corresponding parameters also resampled

    Args:
        counts (array-like): Indices of the resampled particles from a previous
            resampling procedure.
        particlesP (array-like): The original particles before resampling
            generated from a prediction procedure.
        norm_weights (array-like): The normalized log-weights of the particles.
        thetas (array-like): Perturbed parameters associated with the particles.
        subkey (jax.random.PRNGKey): The random key for sampling.

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the resampled particles after
                the latest resampling.
            - particlesF (array-like): The particles after resampling generated
                from the filtering procedure.
            - norm_weights (array-like): The normalized log-weights of the
                resampled particles.
            - thetasF (array-like): The perturbed parameters corresponding to
                the latest perturbed particles (particlesF).
    """
    J = norm_weights.shape[-1]
    counts = _resample(norm_weights, subkey=subkey)
    particlesF = particlesP[counts]
    norm_weights = (
        norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - jnp.log(J)
    )
    thetasF = thetas[counts]
    return counts, particlesF, norm_weights, thetasF


def _no_resampler_thetas(counts, particlesP, norm_weights, thetas, subkey):
    """
    Obtains the original input arguments without resampling for perturbed
    filtering method.

    Args:
        counts (array-like): Indices of the resampled particles from a previous
            resampling procedure.
        particlesP (array-like): The original particles before resampling
            generated from a prediction procedure.
        norm_weights (array-like): The normalized log-weights of the particles.
        thetas (array-like): Perturbed parameters associated with the particles.
        subkey (jax.random.PRNGKey): The random key for sampling. This is not
            used in this function, but is included to maintain the same
            arguments as _resampler_thetas().

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the particles, unchanged.
            - particlesP (array-like): The original particles, unchanged.
            - norm_weights (array-like): The normalized log-weights, unchanged.
            - thetas (array-like): The perturbed parameters, unchanged.
    """
    return counts, particlesP, norm_weights, thetas


"""internal filtering functions - pt.1"""


def _mop_helper(t, inputs, rprocess, dmeasure):
    """
     Helper functions for MOP algorithm, which conducts a single iteration of
     filtering and is called in function 'mop_internal'.

    Args:
         t (int): The current iteration index representing the time
         inputs (list): A list containing the following elements:
             - particlesF (array-like): The particles from the previous filtering
                 procedure.
             - theta (array-like): Parameters involved in the POMP model.
             - covars (array-like or None): Covariates or None if not applicable.
             - loglik (float): The accumulated log-likelihood value.
             - weightsF (array-like): The weights of the particles after the
                 previous filtering procedure.
             - counts (array-like): Indices of particles after resampling.
             - ys (array-like): The entire measurement array.
             - alpha (float): Discount factor.
             - key (jax.random.PRNGKey): The random key for sampling.
         rprocess (function): Simulator procedure for the process model.
         dmeasure (function): Density evaluation for the measurement model.

     Returns:
         list: A list containing updated inputs for next iteration.
             - particlesF (array-like): The updated filtering particles.
             - theta (array-like): Parameters involved in the POMP model.
             - covars (array-like or None):  Covariates or None if not applicable.
             - loglik (float): The updated accumulated log-likelihood value.
             - weightsF (array-like): The updated weights of the particles after
                 the latest iteration.
             - counts (array-like): The updated indices of particles after
                 resampling.
             - ys (array-like): The entire measurement array.
             - alpha (float): Discount factor.
             - key (jax.random.PRNGKey): The updated random key for sampling.
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


@partial(jit, static_argnums=(2, 3, 4, 5))
def _mop_internal(
    theta, ys, J, rinit, rprocess, dmeasure, covars=None, alpha=0.97, key=None
):
    """
    Internal functions for MOP algorithm, which calls function 'mop_helper'
    iteratively.

    Args:
        theta (array-like): Parameters involved in the POMP model.
        ys (array-like): The measurement array.
        J (int): The number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: Negative log-likelihood value.
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
    Internal functions for calculating the mean result using MOP algorithm
    across the measurements.

    Args:
        theta (array-like): Parameters involved in the POMP model.
        ys (array-like): The measurement array.
        J (int): The number of particles.
        rinit (function): simulator for the initial-state distribution.
        rprocess (function): simulator for the process model.
        dmeasure (function): density evaluation for the measurement model.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: The mean of negative log-likelihood value across the
            measurements.
    """
    return _mop_internal(
        theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key
    ) / len(ys)


def _pfilter_helper(t, inputs, rprocess, dmeasure):
    """
    Helper functions for particle filtering algorithm in POMP, which conducts a
    single iteration of filtering and is called in function 'pfilter_internal'.

    Args:
        t (int): The current iteration index representing the time.
        inputs (list): A list containing the following elements:
            - particlesF (array-like): The particles from the previous filtering
                procedure.
            - theta (array-like): Parameters involved in the POMP model.
            - covars (array-like or None): Covariates or None if not applicable.
            - loglik (float): The accumulated log-likelihood value.
            - norm_weights (array-like): The previous normalized weights.
            - counts (array-like): Indices of particles after resampling.
            - ys (array-like): The entire measurement array.
            - thresh (float): Threshold value to determine whether to resample
                particles.
            - key (jax.random.PRNGKey): The random key for sampling.
        rprocess (function): Simulator procedure for the process model.
        dmeasure (function): Density evaluation for the measurement model.

    Returns:
        list: A list containing updated inputs for next iteration.
            - particlesF (array-like): The updated filtering particles.
            - theta (array-like): Parameters involved in the POMP model.
            - covars (array-like or None):  Covariates or None if not
                applicable.
            - loglik (float): The updated accumulated log-likelihood value.
            - norm_weights (array-like): The updated normalized weights of the
                particles after the latest iteration.
            - counts (array-like): The updated indices of particles after
                resampling.
            - ys (array-like): The entire measurement array.
            - thresh (float): Threshold value to determine whether to resample
                particles.
            - key (jax.random.PRNGKey): The updated random key for sampling.
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


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_internal(
    theta, ys, J, rinit, rprocess, dmeasure, covars=None, thresh=100, key=None
):
    """
    Internal functions for particle filtering algorithm, which calls function
    'pfilter_helper' iteratively.

    Args:
        theta (array-like): Parameters involved in the POMP model.
        ys (array-like): The measurement array.
        J (int): The number of particles.
        rinit (function): simulator for the initial-state distribution.
        rprocess (function): simulator for the process model.
        dmeasure (function): density evaluation for the measurement model.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        thresh (float, optional): Threshold value to determine whether to
            resample particles. Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: Negative log-likelihood value
    """
    # if key is None:
    # key = jax.random.PRNGKey(np.random.choice(int(1e18)))

    particlesF = rinit(theta, J, covars=covars)
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
    theta, ys, J, rinit, rprocess, dmeasure, covars=None, thresh=100, key=None
):
    """
    Internal functions for calculating the mean result using particle filtering
    algorithm across the measurements.

    Args:
        theta (array-like): Parameters involved in the POMP model.
        ys (array-like): The measurement array.
        J (int): The number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        thresh (float, optional): Threshold value to determine whether to
            resample particles. Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: The mean of negative log-likelihood value across the
            measurements.
    """
    return _pfilter_internal(
        theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key
    ) / len(ys)


def _perfilter_helper(t, inputs, rprocesses, dmeasures):
    """
    Helper functions for perturbed particle filtering algorithm, which conducts
    a single iteration of filtering and is called in function
    'perfilter_internal'.

    Args:
        t (int): The current iteration index representing the time.
        inputs (list): A list containing the following elements:
            - particlesF (array-like): The particles from the previous filtering
                procedure.
            - thetas (array-like): Perturbed parameters involved in the POMP
                model.
            - sigmas (float): Perturbed factor.
            - covars (array-like or None): Covariates or None if not applicable.
            - loglik (float): The accumulated log-likelihood value.
            - norm_weights (array-like): The previous normalized weights.
            - counts (array-like): Indices of particles after resampling.
            - ys (array-like): The entire measurement array.
            - thresh (float): Threshold value to determine whether to resample
                particles.
            - key (jax.random.PRNGKey): The random key for sampling.
        rprocesses (function): Simulator procedure for the process model.
        dmeasures (function): Density evaluation for the measurement model.

    Returns:
        list: A list containing updated inputs for next iteration.
            - particlesF (array-like): The updated filtering particles.
            - thetas (array-like): Updated perturbed parameters involved in the
                POMP model.
            - sigmas (float): Perturbed factor.
            - covars (array-like or None):  Covariates or None if not
                applicable.
            - loglik (float): The updated accumulated log-likelihood value.
            - norm_weights (array-like): The updated normalized weights of the
                particles after the latest iteration.
            - counts (array-like): The updated indices of particles after
                resampling.
            - ys (array-like): The entire measurement array.
            - thresh (float): Threshold value to determine whether to resample
                particles.
            - key (jax.random.PRNGKey): The updated random key for sampling.
    """
    (
        particlesF,
        thetas,
        sigmas,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
    ) = inputs
    J = len(particlesF)

    key, keys = _keys_helper(key=key, J=J, covars=covars)

    key, subkey = jax.random.split(key)
    thetas += sigmas * jnp.array(jax.random.normal(shape=thetas.shape, key=subkey))

    # Get prediction particles
    # r processes: particleF and thetas are both vectorized (J times)
    if covars is not None:
        particlesP = rprocesses(
            particlesF, thetas, keys, covars
        )  # if t>0 else particlesF
    else:
        particlesP = rprocesses(
            particlesF, thetas, keys, None
        )  # if t>0 else particlesF

    measurements = jnp.nan_to_num(
        dmeasures(ys[t], particlesP, thetas).squeeze(), nan=jnp.log(1e-18)
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
        counts,
        particlesP,
        norm_weights,
        thetas,
        subkey,
    )

    return [
        particlesF,
        thetas,
        sigmas,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
    ]


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal(
    theta,
    ys,
    J,
    sigmas,
    rinit,
    rprocesses,
    dmeasures,
    ndim,
    covars=None,
    thresh=100,
    key=None,
):
    """
    Internal functions for perturbed particle filtering algorithm, which calls
    function 'perfilter_helper' iteratively.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        sigmas (float): Perturbed factor
        rinit (function): Simulator for the initial-state distribution
        rprocesses (function): Simulator for the process model
        dmeasures (function): Density evaluation for the measurement model
        ndim (int): The number of dimensions of theta before perturbation
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        thresh (float, optional): Threshold value to determine whether to
            resample particles. Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - Negative log-likelihood value.
        - An updated perturbed array of parameters.
    """
    loglik = 0
    key, subkey = jax.random.split(key)
    thetas = theta + sigmas * jax.random.normal(
        shape=(J,) + theta.shape[-ndim:], key=subkey
    )
    particlesF = _rinits_internal(rinit, thetas, 1, covars=covars)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    # if key is None:
    # key = jax.random.PRNGKey(np.random.choice(int(1e18)))
    perfilter_helper_2 = partial(
        _perfilter_helper, rprocesses=rprocesses, dmeasures=dmeasures
    )
    (
        particlesF,
        thetas,
        sigmas,
        covars,
        loglik,
        norm_weights,
        counts,
        ys,
        thresh,
        key,
    ) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=perfilter_helper_2,
        init_val=[
            particlesF,
            thetas,
            sigmas,
            covars,
            loglik,
            norm_weights,
            counts,
            ys,
            thresh,
            key,
        ],
    )

    return -loglik, thetas


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal_mean(
    theta,
    ys,
    J,
    sigmas,
    rinit,
    rprocesses,
    dmeasures,
    ndim,
    covars=None,
    thresh=100,
    key=None,
):
    """
    Internal functions for calculating the mean result using perturbed particle
    filtering algorithm across the measurements.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        sigmas (float): Perturbed factor
        rinit (function): Simulator for the initial-state distribution
        rprocesses (function): Simulator for the process model
        dmeasures (function): Density evaluation for the measurement model
        ndim (int): The number of dimensions of theta before perturbation
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        thresh (float, optional): Threshold value to determine whether to
            resample particles. Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - Mean of negative log-likelihood value across the measurements
        - An updated array of parameters.
    """
    value, thetas = _perfilter_internal(
        theta, ys, J, sigmas, rinit, rprocesses, dmeasures, ndim, covars, thresh, key
    )
    return value / len(ys), thetas


"""gradient functions"""


def _line_search(
    obj,
    curr_obj,
    pt,
    grad,
    direction,
    k=1,
    eta=0.9,
    xi=10,
    tau=10,
    c=0.1,
    frac=0.5,
    stoch=False,
):
    """
    Conducts line search algorithm to determine the step size under stochastic
    Quasi-Newton methods. The implentation of the algorithm refers to
    https://arxiv.org/pdf/1909.01238.pdf.

    Args:
        obj (function): The objective function aiming to minimize
        curr_obj (float): The value of the objective function at the current
            point.
        pt (array-like): The array containing current parameter values.
        grad (array-like): The gradient of the objective function at the current
            point.
        direction (array-like): The direction to update the parameters.
        k (int, optional): Iteration index. Defaults to 1.
        eta (float, optional): Initial step size. Defaults to 0.9.
        xi (int, optional): Reduction limit. Defaults to 10.
        tau (int, optional): The maximum number of iterations. Defaults to 10.
        c (float, optional): The user-defined Armijo condition constant.
            Defaults to 0.1.
        frac (float, optional): The fact. Defaults to 0.5.
        stoch (bool, optional): Boolean argument controlling whether to adjust
            the initial step size. Defaults to False.

    Returns:
        float: optimal step size
    """
    itn = 0
    eta = min([eta, xi / k]) if stoch else eta  # if stoch is false, do not change
    next_obj = obj(pt + eta * direction)
    # check whether the new point(new_obj)satisfies the stochastic Armijo condition
    # if not, repeat until the condition is met
    # previous: grad.T @ direction
    while next_obj > curr_obj + eta * c * jnp.sum(grad * direction) or jnp.isnan(
        next_obj
    ):
        eta *= frac
        itn += 1
        if itn > tau:
            break
    return eta


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jgrad(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the gradient of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable
        thresh (float): Threshold value to determine whether to resample
            particles.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the gradient of the pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.grad(_pfilter_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jvg(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    Calculates the both the value and gradient of a mean particle filter
    objective (function 'pfilter_internal_mean') w.r.t. the current estimated
    parameter value using JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        thresh (float): Threshold value to determine whether to resample
            particles.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using pfilter_internal_mean function.
        - The gradient of the function pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.value_and_grad(_pfilter_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jgrad_mop(
    theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None
):
    """
    Calculates the gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the gradient of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.grad(_mop_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jvg_mop(
    theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None
):
    """
    calculates the both the value and gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using mop_internal_mean function.
        - The gradient of the function mop_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.value_and_grad(_mop_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jhess(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the Hessian matrix of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        thresh (float): Threshold value to determine whether to resample
            particles.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the Hessian matrix of the pfilter_internal_mean function
            w.r.t. theta_ests.
    """
    return jax.hessian(_pfilter_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        thresh=thresh,
        key=key,
    )


# get the hessian matrix from mop
@partial(jit, static_argnums=(2, 3, 4, 5))
def _jhess_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha, key=None):
    """
    calculates the Hessian matrix of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        alpha (float): Discount factor.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the Hessian matrix of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.hessian(_mop_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        alpha=alpha,
        key=key,
    )


"""internal filtering functions - pt.2"""
MONITORS = 1


def _mif_internal(
    theta,
    ys,
    rinit,
    rprocess,
    dmeasure,
    rprocesses,
    dmeasures,
    sigmas,
    sigmas_init,
    covars=None,
    M=10,
    a=0.95,
    J=100,
    thresh=100,
    monitor=False,
    verbose=False,
    key=None,
):
    """
    Internal function for conducting the iterated filtering (IF2) algorithm.
    This is called in the '_fit_internal' function.

    Args:
        theta (array-like): Parameters involved in the POMP model.
        ys (array-like): The measurement array.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        rprocesses (function): Simulator for the perturbed process model.
        dmeasures (function): Density evaluation for the perturbed measurement
            model.
        sigmas (float): Perturbed factor.
        sigmas_init (float): Initial perturbed factor.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None.
        M (int, optional): Algorithm Iteration. Defaults to 10.
        a (float, optional): Decay factor for sigmas. Defaults to 0.95.
        J (int, optional): The number of particles. Defaults to 100.
        thresh (float, optional): Threshold value to determine whether to
            resample particles. Defaults to 100.
        monitor (bool, optional): Boolean flag controlling whether to monitor
            the log-likelihood value. Defaults to False.
        verbose (bool, optional): Boolean flag controlling whether to print out
            the log-likehood and parameter information. Defaults to False.

    Returns:
        tuple: A tuple containing:
        - An array of negative log-likelihood through the iterations.
        - An array of parameters through the iterations.
    """
    logliks = []
    params = []

    ndim = theta.ndim
    thetas = jnp.tile(theta, (J,) + (1,) * ndim)
    params.append(thetas)

    if monitor:
        key, subkey = jax.random.split(key=key)
        loglik = jnp.mean(
            jnp.array(
                [
                    _pfilter_internal(
                        thetas.mean(0),
                        ys,
                        J,
                        rinit,
                        rprocess,
                        dmeasure,
                        covars=covars,
                        thresh=thresh,
                        key=subkey,
                    )
                    for i in range(MONITORS)
                ]
            )
        )
        logliks.append(loglik)

    for m in tqdm(range(M)):
        # TODO: Cool sigmas between time-iterations.
        key, *subkeys = jax.random.split(key=key, num=3)
        sigmas = a * sigmas
        sigmas_init = a * sigmas_init
        thetas += sigmas_init * jax.random.normal(shape=thetas.shape, key=subkeys[0])
        loglik_ext, thetas = _perfilter_internal(
            thetas,
            ys,
            J,
            sigmas,
            rinit,
            rprocesses,
            dmeasures,
            ndim=ndim,
            covars=covars,
            thresh=thresh,
            key=subkeys[1],
        )

        params.append(thetas)

        if monitor:
            key, subkey = jax.random.split(key=key)
            loglik = jnp.mean(
                jnp.array(
                    [
                        _pfilter_internal(
                            thetas.mean(0),
                            ys,
                            J,
                            rinit,
                            rprocess,
                            dmeasure,
                            covars=covars,
                            thresh=thresh,
                            key=subkey,
                        )
                        for i in range(MONITORS)
                    ]
                )
            )

            logliks.append(loglik)

            if verbose:
                print(loglik)
                print(thetas.mean(0))

    return jnp.array(logliks), jnp.array(params)

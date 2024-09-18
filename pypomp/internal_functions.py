from functools import partial

import jax
import jax.numpy as np
import numpy as onp
from jax import jit
# from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

# tfd = tfp.distributions
# tfb = tfp.bijectors
# tfpk = tfp.math.psd_kernels

'''resampling functions'''


def _rinits_internal(rinit, thetas, J, covars):
    """
    Simulator for the initial-state distribution, specifically for the perturbed particle filtering method.

    Args:
        rinit (function): Simulator for the initial-state distribution for the unperturbed particle filtering method
        thetas (array-like): Array of parameters used in the likelihood-based inference
        J (int): The number of particles
        covars (array-like or None): Covariates or None if not applicable

    Returns:
        array-like: The simulated initial latent states.
    """
    return rinit(thetas[0], len(thetas), covars)


def _resample(norm_weights):
    """
    Systematic resampling method based on input normalized weights.

    Args:
        norm_weights (array-like): The array containing the logarithm of normalized weights

    Returns:
        array-like: An array containing the resampled indices from the systematic resampling 
                    given the input normalized weights.
    """
    J = norm_weights.shape[-1]
    unifs = (onp.random.uniform() + np.arange(J)) / J
    csum = np.cumsum(np.exp(norm_weights))
    counts = np.repeat(np.arange(J),
                       np.histogram(unifs,
                                    bins=np.pad(csum / csum[-1], pad_width=(1, 0)),
                                    density=False)[0].astype(int),
                       total_repeat_length=J)
    return counts


def _normalize_weights(weights):
    """
    Acquires the normalized log-weights and calculates the log-likelihood.

    Args:
        weights (array-like): Logarithm of unnormalized weights

    Returns:
        tuple: A tuple containing:
            - norm_weights (array-like): The normalized log-weights
            - loglik_t (float): The log of the sum of all particle likelihoods, when the weights
                                are associate with particles, which is equivalent to the total
                                log-likelihood under the specific assumptions
    """
    mw = np.max(weights)
    loglik_t = mw + np.log(np.nansum(np.exp(weights - mw)))
    norm_weights = weights - loglik_t
    return norm_weights, loglik_t


def _resampler(counts, particlesP, norm_weights):
    """
    Resamples the particles based on the weighted resampling rule determined by norm_weights 
    and the original particles generated from the previous prediction.

    Args:
        counts (array-like): Indices of the resampled particles from a previous resampling 
                             procedure
        particlesP (array-like): The original particles before resampling generated from a 
                                 prediction procedure
        norm_weights (array-like): The normalized log-weights of the particles

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the resampled particles after the latest resampling
            - particlesF (array-like): The particles after resampling generated from the filtering 
                                       procedure
            - norm_weights (array-like): The normalized log-weights of the resampled particles
    """
    J = norm_weights.shape[-1]
    counts = _resample(norm_weights)
    particlesF = particlesP[counts]
    norm_weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    return counts, particlesF, norm_weights


def _no_resampler(counts, particlesP, norm_weights):
    """
    Obtains the original input arguments without resampling.

    Args:
        counts (array-like): Indices of the resampled particles from a previous resampling 
                             procedure
        particlesP (array-like): The original particles before resampling generated from a 
                                 prediction procedure
        norm_weights (array-like): The normalized log-weights of the particles

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the particles, unchanged.
            - particlesP (array-like): The original particles, unchanged.
            - norm_weights (array-like): The normalized log-weights, unchanged.
    """
    return counts, particlesP, norm_weights


def _resampler_thetas(counts, particlesP, norm_weights, thetas):
    """
    Resamples the particles for perturbed particle filtering method, with their corresponding parameters 
    also resampled 
    
    Args:
        counts (array-like): Indices of the resampled particles from a previous resampling 
                             procedure
        particlesP (array-like): The original particles before resampling generated from a 
                                 prediction procedure
        norm_weights (array-like): The normalized log-weights of the particles
        thetas (array-like): Perturbed parameters associated with the particles

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the resampled particles after the latest resampling
            - particlesF (array-like): The particles after resampling generated from the filtering 
                                       procedure
            - norm_weights (array-like): The normalized log-weights of the resampled particles
            - thetasF (array-like): The perturbed parameters corresponding to the latest perturbed
                                    particles (particlesF)
    """
    J = norm_weights.shape[-1]
    counts = _resample(norm_weights)
    particlesF = particlesP[counts]
    norm_weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    thetasF = thetas[counts]
    return counts, particlesF, norm_weights, thetasF


def _no_resampler_thetas(counts, particlesP, norm_weights, thetas):
    """
    Obtains the original input arguments without resampling for perturbed filtering method

    Args:
        counts (array-like): Indices of the resampled particles from a previous resampling 
                             procedure
        particlesP (array-like): The original particles before resampling generated from a 
                                 prediction procedure
        norm_weights (array-like): The normalized log-weights of the particles.
        thetas (array-like): Perturbed parameters associated with the particles

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the particles, unchanged.
            - particlesP (array-like): The original particles, unchanged.
            - norm_weights (array-like): The normalized log-weights, unchanged.
            - thetas (array-like): The perturbed parameters, unchanged.
    """
    return counts, particlesP, norm_weights, thetas


def _resampler_pf(counts, particlesP, norm_weights):
    """
    Resamples the particles for function 'pfilter_pf_internal', with weight equalization

    Args:
        counts (array-like): Indices of the resampled particles from a previous resampling 
                             procedure
        particlesP (array-like): The original particles before resampling generated from a 
                                 prediction procedure
        norm_weights (array-like): The normalized log-weights of the particles

    Returns:
        tuple: A tuple containing:
            - counts (array-like): The indices of the resampled particles after the latest resampling
            - particlesF (array-like): The particles after resampling generated from the filtering 
                                       procedure
            - norm_weights (array-like): The normalized log-weights of the resampled particles. Set to
                                         the equal weights.
    """
    J = norm_weights.shape[-1]
    counts = _resample(norm_weights)
    particlesF = particlesP[counts]
    norm_weights = np.log(np.ones(J)) - np.log(J)
    return counts, particlesF, norm_weights


'''internal filtering functions - pt.1'''


def _mop_helper(t, inputs, rprocess, dmeasure):
    """
    Helper functions for MOP algorithm, which conducts a single iteration of filtering and is called
    in function 'mop_internal'.

   Args:
        t (int): The current iteration index representing the time
        inputs (list): A list containing the following elements:
            - particlesF (array-like): The particles from the previous filtering procedure
            - theta (array-like): Parameters involved in the POMP model
            - covars (array-like or None): Covariates or None if not applicable
            - loglik (float): The accumulated log-likelihood value
            - weightsF (array-like): The weights of the particles after the previous filtering procedure
            - counts (array-like): Indices of particles after resampling
            - ys (array-like): The entire measurement array
            - alpha (float): Discount factor
            - key (jax.random.PRNGKey): The random key for sampling
        rprocess (function): Simulator procedure for the process model
        dmeasure (function): Density evaluation for the measurement model

    Returns:
        list: A list containing updated inputs for next iteration.
            - particlesF (array-like): The updated filtering particles
            - theta (array-like): Parameters involved in the POMP model
            - covars (array-like or None):  Covariates or None if not applicable
            - loglik (float): The updated accumulated log-likelihood value
            - weightsF (array-like): The updated weights of the particles after the latest iteration.
            - counts (array-like): The updated indices of particles after resampling
            - ys (array-like): The entire measurement array.
            - alpha (float): Discount factor
            - key (jax.random.PRNGKey): The updated random key for sampling
    """
    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = inputs
    J = len(particlesF)
    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = np.array(keys)

    weightsP = alpha * weightsF

    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)
    else:
        particlesP = rprocess(particlesF, theta, keys, None)

    measurements = dmeasure(ys[t], particlesP, theta)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    loglik += (jax.scipy.special.logsumexp(weightsP + measurements)
               - jax.scipy.special.logsumexp(weightsP))

    norm_weights, loglik_phi_t = _normalize_weights(jax.lax.stop_gradient(measurements))

    counts, particlesF, norm_weightsF = _resampler(counts, particlesP, norm_weights)

    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    return [particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key]


@partial(jit, static_argnums=(2, 3, 4, 5))
def _mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars=None, alpha=0.97, key=None):
    """
    Internal functions for MOP algorithm, which calls function 'mop_helper' iteratively.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: Negative log-likelihood value
    """
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))

    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J) / J)
    weightsF = np.log(np.ones(J) / J)
    counts = np.ones(J).astype(int)
    loglik = 0

    mop_helper_2 = partial(_mop_helper, rprocess=rprocess, dmeasure=dmeasure)

    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = jax.lax.fori_loop(
        lower=0, upper=len(ys), body_fun=mop_helper_2,
        init_val=[particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key])

    return -loglik


@partial(jit, static_argnums=(2, 3, 4, 5))
def _mop_internal_mean(theta, ys, J, rinit, rprocess, dmeasure, covars=None, alpha=0.97, key=None):
    """
    Internal functions for calculating the mean result using MOP algorithm across the measurements.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        rinit (function): simulator for the initial-state distribution
        rprocess (function): simulator for the process model
        dmeasure (function): density evaluation for the measurement model
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: The mean of negative log-likelihood value across the measurements.
    """
    return _mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key) / len(ys)


def _pfilter_helper(t, inputs, rprocess, dmeasure):
    """
    Helper functions for particle filtering algorithm in POMP, which conducts a single iteration 
    of filtering and is called in function 'pfilter_internal'.

    Args:
        t (int): The current iteration index representing the time
        inputs (list): A list containing the following elements:
            - particlesF (array-like): The particles from the previous filtering procedure
            - theta (array-like): Parameters involved in the POMP model.
            - covars (array-like or None): Covariates or None if not applicable.
            - loglik (float): The accumulated log-likelihood value.
            - norm_weights (array-like): The previous normalized weights
            - counts (array-like): Indices of particles after resampling.
            - ys (array-like): The entire measurement array.
            - thresh (float): Threshold value to determine whether to resample particles
            - key (jax.random.PRNGKey): The random key for sampling
        rprocess (function): Simulator procedure for the process model
        dmeasure (function): Density evaluation for the measurement model

    Returns:
        list: A list containing updated inputs for next iteration.
            - particlesF (array-like): The updated filtering particles
            - theta (array-like): Parameters involved in the POMP model
            - covars (array-like or None):  Covariates or None if not applicable
            - loglik (float): The updated accumulated log-likelihood value
            - norm_weights (array-like): The updated normalized weights of the particles 
                                         after the latest iteration
            - counts (array-like): The updated indices of particles after resampling
            - ys (array-like): The entire measurement array.
            - thresh (float): Threshold value to determine whether to resample particles.
            - key (jax.random.PRNGKey): The updated random key for sampling.
    """
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)

    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = np.array(keys)

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

    oddr = np.exp(np.max(norm_weights)) / np.exp(np.min(norm_weights))
    counts, particlesF, norm_weights = jax.lax.cond(oddr > thresh,
                                                    _resampler,
                                                    _no_resampler,
                                                    counts, particlesP, norm_weights)

    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars=None, thresh=100, key=None):
    """
    Internal functions for particle filtering algorithm, which calls function 'pfilter_helper' iteratively.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        rinit (function): simulator for the initial-state distribution
        rprocess (function): simulator for the process model
        dmeasure (function): density evaluation for the measurement model
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        thresh (float, optional): Threshold value to determine whether to resample particles. 
                                Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: Negative log-likelihood value
    """
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))

    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J) / J)
    norm_weights = np.log(np.ones(J) / J)
    counts = np.ones(J).astype(int)
    loglik = 0

    pfilter_helper_2 = partial(_pfilter_helper, rprocess=rprocess, dmeasure=dmeasure)
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
        lower=0, upper=len(ys), body_fun=pfilter_helper_2,
        init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])

    return -loglik


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_internal_mean(theta, ys, J, rinit, rprocess, dmeasure, covars=None, thresh=100, key=None):
    """
    Internal functions for calculating the mean result using particle filtering algorithm across the measurements.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        rinit (function): simulator for the initial-state distribution
        rprocess (function): simulator for the process model
        dmeasure (function): density evaluation for the measurement model
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        thresh (float, optional): Threshold value to determine whether to resample particles. 
                                  Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: The mean of negative log-likelihood value across the measurements.
    """
    return _pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key) / len(ys)


def _perfilter_helper(t, inputs, rprocesses, dmeasures):
    """
    Helper functions for perturbed particle filtering algorithm, which conducts a single iteration of 
    filtering and is called in function 'perfilter_internal'.

    Args:
        t (int): The current iteration index representing the time.
        inputs (list): A list containing the following elements:
            - particlesF (array-like): The particles from the previous filtering procedure
            - thetas (array-like): Perturbed parameters involved in the POMP model
            - sigmas (float): Perturbed factor.
            - covars (array-like or None): Covariates or None if not applicable
            - loglik (float): The accumulated log-likelihood value.
            - norm_weights (array-like): The previous normalized weights
            - counts (array-like): Indices of particles after resampling
            - ys (array-like): The entire measurement array
            - thresh (float): Threshold value to determine whether to resample particles.
            - key (jax.random.PRNGKey): The random key for sampling
        rprocesses (function): Simulator procedure for the process model
        dmeasures (function): Density evaluation for the measurement model

    Returns:
        list: A list containing updated inputs for next iteration
            - particlesF (array-like): The updated filtering particles
            - thetas (array-like): Updated perturbed parameters involved in the POMP model.
            - sigmas (float): Perturbed factor.
            - covars (array-like or None):  Covariates or None if not applicable.
            - loglik (float): The updated accumulated log-likelihood value.
            - norm_weights (array-like): The updated normalized weights of the particles 
                                         after the latest iteration.
            - counts (array-like): The updated indices of particles after resampling
            - ys (array-like): The entire measurement array
            - thresh (float): Threshold value to determine whether to resample particles.
            - key (jax.random.PRNGKey): The updated random key for sampling
    """
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)

    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = np.array(keys)

    thetas += sigmas * np.array(onp.random.normal(size=thetas.shape))

    # Get prediction particles
    # r processes: particleF and thetas are both vectorized (J times)
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars)  # if t>0 else particlesF
    else:
        particlesP = rprocesses(particlesF, thetas, keys, None)  # if t>0 else particlesF

    measurements = np.nan_to_num(dmeasures(ys[t], particlesP, thetas).squeeze(),
                                 nan=np.log(1e-18))  # shape (Np,)

    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)

    loglik += loglik_t
    oddr = np.exp(np.max(norm_weights)) / np.exp(np.min(norm_weights))
    counts, particlesF, norm_weights, thetas = jax.lax.cond(oddr > thresh,
                                                            _resampler_thetas,
                                                            _no_resampler_thetas,
                                                            counts, particlesP, norm_weights, thetas)

    return [particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key]


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, ndim, covars=None, thresh=100,
                       key=None):
    """
    Internal functions for perturbed particle filtering algorithm, which calls function 'perfilter_helper'
    iteratively.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        sigmas (float): Perturbed factor
        rinit (function): Simulator for the initial-state distribution
        rprocesses (function): Simulator for the process model
        dmeasures (function): Density evaluation for the measurement model
        ndim (int): The number of dimensions of theta before perturbation
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        thresh (float, optional): Threshold value to determine whether to resample particles. 
                                Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - Negative log-likelihood value
        - An updated perturbed array of parameters.
    """
    loglik = 0
    thetas = theta + sigmas * onp.random.normal(size=(J,) + theta.shape[-ndim:])
    # thetas = theta + sigmas * onp.random.normal(size=(J,) + theta.shape[1:])
    particlesF = _rinits_internal(rinit, thetas, 1, covars=covars)
    weights = np.log(np.ones(J) / J)
    norm_weights = np.log(np.ones(J) / J)
    counts = np.ones(J).astype(int)
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    perfilter_helper_2 = partial(_perfilter_helper, rprocesses=rprocesses, dmeasures=dmeasures)
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
        lower=0, upper=len(ys), body_fun=perfilter_helper_2,
        init_val=[particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key])

    return -loglik, thetas


@partial(jit, static_argnums=(2, 4, 5, 6, 7))
def _perfilter_internal_mean(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, ndim, covars=None, thresh=100,
                            key=None):
    """
    Internal functions for calculating the mean result using perturbed particle filtering algorithm across the 
    measurements.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        sigmas (float): Perturbed factor
        rinit (function): Simulator for the initial-state distribution
        rprocesses (function): Simulator for the process model
        dmeasures (function): Density evaluation for the measurement model
        ndim (int): The number of dimensions of theta before perturbation
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        thresh (float, optional): Threshold value to determine whether to resample particles. 
                                Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - Mean of negative log-likelihood value across the measurements
        - An updated array of parameters.
    """
    value, thetas = _perfilter_internal(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, ndim, covars, thresh, key)
    return value / len(ys), thetas


def _pfilter_helper_pf(t, inputs, rprocess, dmeasure):
    """
    Helper functions for particle filtering algorithm with weight equalization in POMP, which conducts a 
    single iteration of filtering and is called in function 'pfilter_pf_internal'.

    Args:
        t (int): The current iteration index representing the time
        inputs (list): A list containing the following elements:
            - particlesF (array-like): The particles from the previous filtering procedure
            - theta (array-like): Parameters involved in the POMP model
            - covars (array-like or None): Covariates or None if not applicable
            - loglik (float): The accumulated log-likelihood value
            - norm_weights (array-like): The previous normalized weights
            - counts (array-like): Indices of particles after resampling
            - ys (array-like): The entire measurement array
            - thresh (float): Threshold value to determine whether to resample particles
            - key (jax.random.PRNGKey): The random key for sampling
        rprocess (function): Simulator procedure for the process model
        dmeasure (function): Density evaluation for the measurement model

    Returns:
        list: A list containing updated inputs for next iteration
            - particlesF (array-like): The updated filtering particles
            - theta (array-like): Parameters involved in the POMP model
            - covars (array-like or None):  Covariates or None if not applicable
            - loglik (float): The updated accumulated log-likelihood value
            - norm_weights (array-like): The updated normalized weights of the particles 
                                         after the latest iteration (weight equalization)
            - counts (array-like): The updated indices of particles after resampling
            - ys (array-like): The entire measurement array
            - thresh (float): Threshold value to determine whether to resample particles
            - key (jax.random.PRNGKey): The updated random key for sampling
    """
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)

    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J * covars.shape[1] + 1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:
        key, *keys = jax.random.split(key, num=J + 1)
        keys = np.array(keys)

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

    oddr = np.exp(np.max(norm_weights)) / np.exp(np.min(norm_weights))
    counts, particlesF, norm_weights = jax.lax.cond(oddr > thresh,
                                                    _resampler_pf,
                                                    _no_resampler,
                                                    counts, particlesP, norm_weights)

    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_pf_internal(theta, ys, J, rinit, rprocess, dmeasure, covars=None, thresh=100, key=None):
    """
    Internal functions for calculating the mean result using particle filtering algorithm with weight equalization 
    across the measurements, which calls function 'pfilter_pf_helper()' iteratively.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        J (int): The number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        thresh (float, optional): Threshold value to determine whether to resample particles. 
                                  Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        float: The mean of negative log-likelihood value across the measurements
    """
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))

    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J) / J)
    norm_weights = np.log(np.ones(J) / J)
    counts = np.ones(J).astype(int)
    loglik = 0

    pfilter_pf_helper_2 = partial(_pfilter_helper_pf, rprocess=rprocess, dmeasure=dmeasure)

    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
        lower=0, upper=len(ys), body_fun=pfilter_pf_helper_2,
        init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])

    return -loglik / len(ys)


'''gradient functions'''


def _line_search(obj, curr_obj, pt, grad, direction, k=1, eta=0.9, xi=10, tau=10, c=0.1, frac=0.5, stoch=False):
    """
    Conducts line search algorithm to determine the step size under stochastic Quasi-Newton methods. 
    The implentation of the algorithm refers to https://arxiv.org/pdf/1909.01238.pdf

    Args:
        obj (function): The objective function aiming to minimize
        curr_obj (float): The value of the objective function at the current point
        pt (array-like): The array containing current parameter values
        grad (array-like): The gradient of the objective function at the current point
        direction (array-like): The direction to update the parameters
        k (int, optional): Iteration index. Defaults to 1.
        eta (float, optional): Initial step size. Defaults to 0.9.
        xi (int, optional): Reduction limit. Defaults to 10.
        tau (int, optional): The maximum number of iterations. Defaults to 10.
        c (float, optional): The user-defined Armijo condition constant. Defaults to 0.1.
        frac (float, optional): The fact. Defaults to 0.5.
        stoch (bool, optional): Boolean argument controlling whether to adjust the initial step size. 
                                Defaults to False.

    Returns:
        float: optimal step size
    """
    itn = 0
    eta = min([eta, xi / k]) if stoch else eta  # if stoch is false, do not change
    next_obj = obj(pt + eta * direction)
    # check whether the new point(new_obj)satisfies the stochastic Armijo condition
    # if not, repeat until the condition is met
    # previous: grad.T @ direction
    while next_obj > curr_obj + eta * c * np.sum(grad * direction) or np.isnan(next_obj):
        eta *= frac
        itn += 1
        if itn > tau:
            break
    return eta


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jgrad_pf(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the gradient of a mean particle filter objective with weight equalization 
    (function 'pfilter_pf_internal') w.r.t. the current estimated parameter value using JAX's automatic 
    differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable 
        thresh (float): Threshold value to determine whether to resample particles. 
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the gradient of the function pfilter_pf_internal() w.r.t. theta_ests
    """
    return jax.grad(_pfilter_pf_internal)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh,
                                         key=key)


# return the value and gradient at the same time
@partial(jit, static_argnums=(2, 3, 4, 5))
def _jvg_pf(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the both the value and gradient of a mean particle filter objective with weight equalization 
    (function 'pfilter_pf_internal') w.r.t. the current estimated parameter value using JAX's automatic 
    differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable
        thresh (float): Threshold value to determine whether to resample particles. 
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements using pfilter_pf_internal().
        - The gradient of the function pfilter_pf_internal() w.r.t. theta_ests
    """
    return jax.value_and_grad(_pfilter_pf_internal)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars,
                                                   thresh=thresh, key=key)


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jgrad(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the gradient of a mean particle filter objective (function 'pfilter_internal_mean') w.r.t. 
    the current estimated parameter value using JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable 
        thresh (float): Threshold value to determine whether to resample particles. 
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the gradient of the pfilter_internal_mean function w.r.t. theta_ests
    """
    return jax.grad(_pfilter_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh,
                                           key=key)


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jvg(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the both the value and gradient of a mean particle filter objective (function 'pfilter_internal_mean') 
    w.r.t. the current estimated parameter value using JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable
        thresh (float): Threshold value to determine whether to resample particles. 
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements using pfilter_internal_mean function.
        - The gradient of the function pfilter_internal_mean function w.r.t. theta_ests
    """
    return jax.value_and_grad(_pfilter_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars,
                                                     thresh=thresh, key=key)


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jgrad_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None):
    """
    calculates the gradient of a mean MOP objective (function 'mop_internal_mean') w.r.t. the current estimated
    parameter value using JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the gradient of the mop_internal_mean function w.r.t. theta_ests
    """
    return jax.grad(_mop_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha,
                                       key=key)


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jvg_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None):
    """
    calculates the both the value and gradient of a mean MOP objective (function 'mop_internal_mean') w.r.t. 
    the current estimated parameter value using JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements using mop_internal_mean function.
        - The gradient of the function mop_internal_mean function w.r.t. theta_ests
    """
    return jax.value_and_grad(_mop_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars,
                                                 alpha=alpha, key=key)


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jhess(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the Hessian matrix of a mean particle filter objective (function 'pfilter_internal_mean') w.r.t. 
    the current estimated parameter value using JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable 
        thresh (float): Threshold value to determine whether to resample particles. 
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the Hessian matrix of the pfilter_internal_mean function w.r.t. theta_ests
    """
    return jax.hessian(_pfilter_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars,
                                              thresh=thresh, key=key)


# get the hessian matrix from mop
@partial(jit, static_argnums=(2, 3, 4, 5))
def _jhess_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha, key=None):
    """
    calculates the Hessian matrix of a mean MOP objective (function 'mop_internal_mean') w.r.t. the current 
    estimated parameter value using JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable
        alpha (float): Discount factor
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the Hessian matrix of the mop_internal_mean function w.r.t. theta_ests
    """
    return jax.hessian(_mop_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha,
                                          key=key)


'''internal filtering functions - pt.2'''
MONITORS = 1


def _mif_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init, covars=None, M=10,
                 a=0.95, J=100, thresh=100, monitor=False, verbose=False):
    """
    Internal functions for conducting iterated filtering (IF2) algorithm, is called in 'fit_internal' function.

    Args:
        theta (array-like): Parameters involved in the POMP model
        ys (array-like): The measurement array
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        rprocesses (function): Simulator for the perturbed process model
        dmeasures (function): Density evaluation for the perturbed measurement model
        sigmas (float): Perturbed factor
        sigmas_init (float): Initial perturbed factor
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        M (int, optional): Algorithm Iteration. Defaults to 10.
        a (float, optional): Decay factor for sigmas. Defaults to 0.95.
        J (int, optional): The number of particles. Defaults to 100.
        thresh (float, optional): Threshold value to determine whether to resample particles. Defaults to 100.
        monitor (bool, optional): Boolean flag controlling whether to monitor the log-likelihood value. 
                                  Defaults to False.
        verbose (bool, optional): Boolean flag controlling whether to print out the log-likehood and parameter 
                                  information. Defaults to False.

    Returns:
        tuple: A tuple containing:
        - An array of negative log-likelihood through the iterations
        - An array of parameters through the iterations 
    """
    logliks = []
    params = []

    ndim = theta.ndim
    thetas = theta + sigmas_init * onp.random.normal(size=(J,) + theta.shape[-ndim:])
    # thetas = theta + sigmas_init*onp.random.normal(size=(J, theta.shape[-1]))
    params.append(thetas)
    if monitor:
        loglik = np.mean(
            np.array([_pfilter_internal(thetas.mean(0), ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh)
                      for i in range(MONITORS)]))
        logliks.append(loglik)

    for m in tqdm(range(M)):
        sigmas *= a
        thetas += sigmas * onp.random.normal(size=thetas.shape)
        loglik_ext, thetas = _perfilter_internal(thetas, ys, J, sigmas, rinit, rprocesses, dmeasures, ndim=ndim,
                                                covars=covars, thresh=thresh)

        params.append(thetas)

        if monitor:
            loglik = np.mean(np.array(
                [_pfilter_internal(thetas.mean(0), ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh)
                 for i in range(MONITORS)]))

            logliks.append(loglik)

            if verbose:
                print(loglik)
                print(thetas.mean(0))

    return np.array(logliks), np.array(params)


def _train_internal(theta_ests, ys, rinit, rprocess, dmeasure, covars=None, J=5000, Jh=1000, method='Newton', itns=20,
                   beta=0.9, eta=0.0025, c=0.1, max_ls_itn=10, thresh=100, verbose=False, scale=False, ls=False,
                   alpha=1):
    """
    Internal function for conducting the MOP gradient estimate method, is called in 'fit_internal'
    function.

    Args:
        theta_ests (array-like): Initial value of parameter values before SGD
        ys (array-like): The measurement array
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None. Defaults to None.
        J (int, optional): The number of particles in the MOP objective for obtaining the gradient. Defaults to 5000.
        Jh (int, optional): The number of particles in the MOP objective for obtaining the Hessian matrix. 
                            Defaults to 1000.
        method (str, optional): The optimization method to use, including Newton method, weighted Newton method
                                BFGS method, gradient descent. Defaults to 'Newton'.
        itns (int, optional): Maximum iteration for the gradient descent optimization. Defaults to 20.
        beta (float, optional): Initial step size for the line search algorithm. Defaults to 0.9.
        eta (float, optional): Initial step size. Defaults to 0.0025.
        c (float, optional): The user-defined Armijo condition constant. Defaults to 0.1.
        max_ls_itn (int, optional): The maximum number of iterations for the line search algorithm. Defaults to 10.
        thresh (int, optional): Threshold value to determine whether to resample particles in pfilter function.
                                Defaults to 100.
        verbose (bool, optional): Boolean flag controlling whether to print out the log-likelihood and parameter
                                  information. Defaults to False.
        scale (bool, optional): Boolean flag controlling normalizing the direction or not. Defaults to False.
        ls (bool, optional): Boolean flag controlling using the line search or not. Defaults to False.
        alpha (int, optional): Discount factor. Defaults to 1.

    Returns:
        tuple: A tuple containing:
        - An array of negative log-likelihood through the iterations
        - An array of parameters through the iterations 
    """
    Acopies = []
    grads = []
    hesses = []
    logliks = []
    hess = np.eye(theta_ests.shape[-1])  # default one

    for i in tqdm(range(itns)):
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
        if MONITORS == 1:
            loglik, grad = _jvg_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)

            loglik *= len(ys)
        else:
            grad = _jgrad_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)
            loglik = np.mean(np.array([_pfilter_internal(theta_ests, ys, J, rinit, rprocess, dmeasure,
                                                        covars=covars, thresh=-1, key=key)
                                       for i in range(MONITORS)]))

        if method == 'Newton':
            hess = _jhess_mop(theta_ests, ys, Jh, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)

            # flatten
            theta_flat = theta_ests.flatten()
            grad_flat = grad.flatten()
            hess_flat = hess.reshape(theta_flat.size, theta_flat.size)
            hess_flat_pinv = np.linalg.pinv(hess_flat)
            direction_flat = -hess_flat_pinv @ grad_flat
            direction = direction_flat.reshape(theta_ests.shape)

            # direction = -np.linalg.pinv(hess) @ grad
        elif method == 'WeightedNewton':
            if i == 0:
                hess = _jhess_mop(theta_ests, ys, Jh, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)
                theta_flat = theta_ests.flatten()
                grad_flat = grad.flatten()
                hess_flat = hess.reshape(theta_flat.size, theta_flat.size)
                hess_flat_pinv = np.linalg.pinv(hess_flat)
                direction_flat = -hess_flat_pinv @ grad_flat
                direction = direction_flat.reshape(theta_ests.shape)
                # direction = -np.linalg.pinv(hess) @ grad
            else:
                hess = _jhess_mop(theta_ests, ys, Jh, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)
                wt = (i ** onp.log(i)) / ((i + 1) ** (onp.log(i + 1)))
                theta_flat = theta_ests.flatten()
                grad_flat = grad.flatten()
                weighted_hess = wt * hesses[-1] + (1 - wt) * hess
                weighted_hess_flat = weighted_hess.reshape(theta_flat.size, theta_flat.size)
                weighted_hess_flat_pinv = np.linalg.pinv(weighted_hess_flat)
                direction_flat = -weighted_hess_flat_pinv @ grad_flat
                direction = direction_flat.reshape(theta_ests.shape)
                # direction = -np.linalg.pinv(wt * hesses[-1] + (1-wt) * hess) @ grad

        elif method == 'BFGS' and i > 1:
            s_k = et * direction
            # not grad but grads
            y_k = grad - grads[-1]
            rho_k = np.reciprocal(np.dot(y_k, s_k))
            sy_k = s_k[:, np.newaxis] * y_k[np.newaxis, :]
            w = np.eye(theta_ests.shape[-1], dtype=rho_k.dtype) - rho_k * sy_k
            # H_(k+1) = W_k^T@H_k@W_k + pho_k@s_k@s_k^T 
            hess = (np.einsum('ij,jk,lk', w, hess, w)
                    + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :])
            hess = np.where(np.isfinite(rho_k), hess, hess)

            theta_flat = theta_ests.flatten()
            grad_flat = grad.flatten()
            hess_flat = hess.reshape(theta_flat.size, theta_flat.size)

            direction_flat = -hess_flat @ grad_flat
            direction = direction_flat.reshape(theta_ests.shape)

            # direction = -hess @ grad

        else:
            direction = -grad

        Acopies.append(theta_ests)
        logliks.append(loglik)
        grads.append(grad)
        hesses.append(hess)

        if scale:
            direction = direction / np.linalg.norm(direction)

        eta = _line_search(
            partial(_pfilter_internal, ys=ys, J=J, rinit=rinit, rprocess=rprocess, dmeasure=dmeasure, covars=covars,
                    thresh=thresh, key=key),
            loglik, theta_ests, grad, direction, k=i + 1, eta=beta, c=c, tau=max_ls_itn) if ls else eta
        try:
            et = eta if len(eta) == 1 else eta[i]
        except:
            et = eta
        if i % 1 == 0 and verbose:
            print(theta_ests, et, logliks[i])

        theta_ests += et * direction

    logliks.append(np.mean(np.array(
        [_pfilter_internal(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh) for i in
         range(MONITORS)])))
    Acopies.append(theta_ests)

    return np.array(logliks), np.array(Acopies)


def _fit_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses=None, dmeasures=None, sigmas=None, sigmas_init=None,
                 covars=None, M=10, a=0.9,
                 J=100, Jh=1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1,
                 max_ls_itn=10, thresh_mif=100, thresh_tr=100, verbose=False, scale=False, ls=False, alpha=0.1,
                 monitor=True, mode="IFAD"):
    """
    Internal function for executing the iterated filtering (IF2), MOP gradient-based iterative optimization method (GD), 
    and iterated filtering with automatic differentiation (IFAD) for likelihood maximization algorithm of POMP model.

    Args:
        theta (array-like): Initial parameters involved in the POMP model
        ys (array-like): The measurement array
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        rprocesses (function, optional): Simulator for the perturbed process model Defaults to None.
        dmeasures (function, optional): Density evaluation for the perturbed measurement model. Defaults to None.
        sigmas (float, optional): Perturbed factor. Defaults to None.
        sigmas_init (float, optional): Initial perturbed factor. Defaults to None.
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        M (int, optional): Maximum algorithm iteration for iterated filtering. Defaults to 10.
        a (float, optional): Decay factor for sigmas. Defaults to 0.9.
        J (int, optional): The number of particles in iterated filtering and the number of particles in the MOP
         objective for obtaining the gradient in gradient optimization. Defaults to 100.
        Jh (int, optional): The number of particles in the MOP objective for obtaining the Hessian matrix. Defaults to
         1000.
        method (str, optional): The gradient optimization method to use, including Newton method, weighted Newton method
                                BFGS method, gradient descent. Defaults to 'Newton'.
        itns (int, optional): Maximum iteration for the gradient optimization. Defaults to 20.
        beta (float, optional): Initial step size. Defaults to 0.9.
        eta (float, optional): Initial step size. Defaults to 0.0025.
        c (float, optional): The user-defined Armijo condition constant. Defaults to 0.1.
        max_ls_itn (int, optional): The maximum number of iterations for the line search algorithm. Defaults to 10.
        thresh_mif (int, optional): Threshold value to determine whether to resample particles in iterated filtering.
                                    Defaults to 100.
        thresh_tr (int, optional): Threshold value to determine whether to resample particles in gradient optimization.
                                   Defaults to 100.
        verbose (bool, optional):  Boolean flag controlling whether to print out the log-likelihood and parameter
                                  information. Defaults to False.
        scale (bool, optional): Boolean flag controlling normalizing the direction or not. Defaults to False.
        ls (bool, optional): Boolean flag controlling using the line search or not. Defaults to False.
        alpha (float, optional): Discount factor. Defaults to 0.1.
        monitor (bool, optional): Boolean flag controlling whether to monitor the log-likelihood value. Defaults to
         True.
        mode (str, optional): The optimization algorithm to use, including 'IF2', 'GD', and 'IFAD'. Defaults to "IFAD".

    Raises:
        TypeError: Missing the required arguments in iterated filtering
        TypeError: Missing the required arguments in gradient optimization method
        TypeError: Invalid mode input

    Returns:
        tuple: A tuple containing:
        - An array of negative log-likelihood through the iterations
        - An array of parameters through the iterations
    """
    if mode == 'IF2':
        if rprocesses is not None and dmeasures is not None and sigmas is not None and sigmas_init is not None:
            # Directly call mif_internal and return the results
            mif_logliks_warm, mif_params_warm = _mif_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses,
                                                             dmeasures, sigmas,
                                                             sigmas_init, covars, M, a, J, thresh_mif, monitor=monitor,
                                                             verbose=verbose)
            return np.array(mif_logliks_warm), np.array(mif_params_warm)
        else:
            raise TypeError(f"Unknown parameter")

    elif mode == 'GD':
        # Directly call train_internal and return the results
        gd_logliks, gd_ests = _train_internal(theta, ys, rinit, rprocess, dmeasure, covars, J, Jh, method, itns, beta,
                                             eta, c,
                                             max_ls_itn, thresh_tr, verbose, scale, ls, alpha)
        return np.array(gd_logliks), np.array(gd_ests)

    elif mode == 'IFAD':
        # The original logic combining both mif_internal and train_internal
        if rprocesses is not None and dmeasures is not None and sigmas is not None and sigmas_init is not None:
            mif_logliks_warm, mif_params_warm = _mif_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses,
                                                             dmeasures, sigmas,
                                                             sigmas_init, covars, M, a, J, thresh_mif, monitor=True,
                                                             verbose=verbose)
            theta_ests = mif_params_warm[mif_logliks_warm.argmin()].mean(0)
            gd_logliks, gd_ests = _train_internal(theta_ests, ys, rinit, rprocess, dmeasure, covars, J, Jh, method, itns,
                                                 beta, eta, c,
                                                 max_ls_itn, thresh_tr, verbose, scale, ls, alpha)
            return np.array(gd_logliks), np.array(gd_ests)
        else:
            raise TypeError(f"Unknown parameter")

    else:
        raise TypeError(f"Unknown mode: {mode}. Choose from 'IF2', 'GD', or 'IFAD'.")

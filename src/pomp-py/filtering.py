

import jax
import itertools
import numpy as onp
import jax.numpy as np
import ipywidgets as widgets

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from functools import partial
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

from tqdm.notebook import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')

from resampling import *
from pomps import rinit, rprocess, dmeasure, rinits, rprocesses, dmeasures

def resampler(counts, particlesP, norm_weights):
    J = norm_weights.shape[-1]
    counts = resample(norm_weights)
    particlesF = particlesP[counts]
    norm_weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    return counts, particlesF, norm_weights

def no_resampler(counts, particlesP, norm_weights):
    return counts, particlesP, norm_weights

def resampler_thetas(counts, particlesP, norm_weights, thetas):
    J = norm_weights.shape[-1]
    counts = resample(norm_weights)
    particlesF = particlesP[counts]
    norm_weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    thetasF = thetas[counts]
    return counts, particlesF, norm_weights, thetasF

def no_resampler_thetas(counts, particlesP, norm_weights, thetas):
    return counts, particlesP, norm_weights, thetas
    
'''
# Resampling condition
if np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights)) > thresh:
    resamples += 1 #tracker
    # Systematic resampling
    counts = resample(norm_weights, J)
    particlesF = particlesP[counts]
    weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
else:
    particlesF = particlesP
    weights = norm_weights
'''


def mop_helper(t, inputs):
    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = inputs
    J = len(particlesF)
    
    if len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
        key, *keys = jax.random.split(key, num=J+1)
        keys = np.array(keys)
        
    # Discount weights by alpha in logspace
    weightsP = alpha*weightsF
    
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys, None)
        
    measurements = dmeasure(ys[t], particlesP, theta)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
    
    # Using before-resampling conditional likelihood
    loglik += (jax.scipy.special.logsumexp(weightsP + measurements) 
               - jax.scipy.special.logsumexp(weightsP))
    
    # Obtain normalized measurement likelihoods for resampling
    norm_weights, loglik_phi_t = normalize_weights(jax.lax.stop_gradient(measurements))

    # Systematic resampling according to normalized measurement likelihoods
    counts, particlesF, norm_weightsF = resampler(counts, particlesP, norm_weights)
    
    # Correct for theta/phi and resample
    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    #jax.debug.print(loglik, loglik_t)
    return [particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key]

    
# test on linear gaussian toy model again
@partial(jit, static_argnums=2)
def mop(theta, ys, J, covars=None, alpha=0.97, key=None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    weightsF = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0
    
    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=mop_helper, 
                 init_val=[particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key])
    
    return -loglik

# test on linear gaussian toy model again
@partial(jit, static_argnums=2)
def mop_mean(theta, ys, J, covars=None, alpha=0.97, key=None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    weightsF = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0
    
    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=mop_helper, 
                 init_val=[particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key])
    
    return -loglik/len(ys)



def pfilter_helper(t, inputs):
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)
    
    if len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
        key, *keys = jax.random.split(key, num=J+1)
        keys = np.array(keys)
        
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys, None)
        
    measurements = dmeasure(ys[t], particlesP, theta)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
    
    # Multiply weights by measurement model result
    weights = norm_weights + measurements

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)
    loglik += loglik_t

    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    # Systematic resampling
    # Here we resample before calculating dmeasure at timestep t!
    # so we resample with the old weights, not the new ones! wrong.
    # if resampling, resample with dmeasure
    # if not resampling, just propagate
    counts, particlesF, norm_weights = jax.lax.cond(oddr > thresh, 
                                               resampler, 
                                               no_resampler, 
                                               counts, particlesP, norm_weights)

    #jax.debug.print(loglik, loglik_t)
    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]
    
# test on linear gaussian toy model again
@partial(jit, static_argnums=2)
def pfilter(theta, ys, J, covars=None, thresh=100, key=None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0
    
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=pfilter_helper, 
                 init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik

# test on linear gaussian toy model again
@partial(jit, static_argnums=2)
def pfilter_mean(theta, ys, J, covars=None, thresh=100, key=None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0
    
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=pfilter_helper, 
                 init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik/len(ys)

def perfilter_helper(t, inputs):
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)
    
    if len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
        key, *keys = jax.random.split(key, num=J+1)
        keys = np.array(keys)
    
    # Perturb parameters
    thetas += sigmas*np.array(onp.random.normal(size=thetas.shape))
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars)# if t>0 else particlesF
    else:
        particlesP = rprocesses(particlesF, thetas, keys)# if t>0 else particlesF
        
    
    measurements = np.nan_to_num(dmeasures(ys[t], particlesP, thetas, keys=keys).squeeze(),
                            nan=np.log(1e-18)) #shape (Np,)
    
    
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
    # Multiply weights by measurement model result
    weights = norm_weights + measurements 
                             
    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)
    
    # Sum up loglik
    loglik += loglik_t
    
    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    # Systematic resampling
    counts, particlesF, norm_weights, thetas = jax.lax.cond(oddr > thresh, 
                                               resampler_thetas, 
                                               no_resampler_thetas, 
                                               counts, particlesP, norm_weights, thetas)
    
    
    return [particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key]


@partial(jit, static_argnums=2)
def perfilter(theta, ys, J, sigmas, covars=None, a=0.9, thresh=100, key=None):
    
    loglik = 0
    thetas = theta + sigmas*onp.random.normal(size=(J, theta.shape[-1]))
    particlesF = rinits(thetas, 1, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=perfilter_helper, 
                 init_val=[particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik, thetas

@partial(jit, static_argnums=2)
def perfilter_mean(theta, ys, J, sigmas, covars=None, a=0.9, thresh=100, key=None):
    
    loglik = 0
    thetas = theta + sigmas*onp.random.normal(size=(J, theta.shape[-1]))
    particlesF = rinits(thetas, 1, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=perfilter_helper, 
                 init_val=[particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik/len(ys), thetas



def resampler_pf(counts, particlesP, norm_weights):
    J = norm_weights.shape[-1]
    counts = resample(norm_weights)
    particlesF = particlesP[counts]
    return counts, particlesF, np.log(np.ones(J)) - np.log(J)


def pfilter_helper_pf(t, inputs):
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)
    
    if len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
        key, *keys = jax.random.split(key, num=J+1)
        keys = np.array(keys)
        
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys, None)
        
    measurements = dmeasure(ys[t], particlesP, theta)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
    
    # Multiply weights by measurement model result
    weights = norm_weights + measurements

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)
    loglik += loglik_t

    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    # Systematic resampling
    # Here we resample before calculating dmeasure at timestep t!
    # so we resample with the old weights, not the new ones! wrong.
    # if resampling, resample with dmeasure
    # if not resampling, just propagate
    counts, particlesF, norm_weights = jax.lax.cond(oddr > thresh, 
                                               resampler_pf, 
                                               no_resampler, 
                                               counts, particlesP, norm_weights)

    #jax.debug.print(loglik, loglik_t)
    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]


# test on linear gaussian toy model again
@partial(jit, static_argnums=2)
def pfilter_pf(theta, ys, J, covars=None, thresh=100, key=None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0
    
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=pfilter_helper_pf, 
                 init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik/len(ys)



#PFILTER
'''  
for t in tqdm(range(len(ys))):
    keys = np.array([jax.random.PRNGKey(onp.random.choice(int(1e18))) for j in range(J)])
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars[t])# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys)

    resamples += 1 #tracker
    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    # Systematic resampling
    counts, particlesF, weights = jax.lax.cond(oddr > thresh, 
                                               partial(resampler, J=J), 
                                               partial(no_resampler, J=J), 
                                               counts, particlesP, norm_weights)



    # Multiply weights by measurement model result
    keys = np.array([jax.random.PRNGKey(onp.random.choice(int(1e18))) for j in range(J)])
    weights += dmeasure(ys[t], particlesP, theta, keys=keys) #shape (Np,)

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)

    # Sum up loglik
    loglik += loglik_t
'''

#PERFILTER
'''
# inner filtering loop
for t in tqdm(range(len(ys))):
    keys = np.array([jax.random.PRNGKey(onp.random.choice(int(1e18))) for j in range(J)])

    # Perturb parameters
    thetas += sigmas*np.array(onp.random.normal(size=thetas.shape))
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars[t])# if t>0 else particlesF
    else:
        particlesP = rprocesses(particlesF, thetas, keys)# if t>0 else particlesF

    # Resampling condition
    if np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights)) > thresh:
        resamples += 1 #tracker
        # Systematic resampling
        counts = resample(norm_weights, J)
        particlesF = particlesP[counts]
        thetas = thetas[counts]
        weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    else:
        particlesF = particlesP
        weights = norm_weights


    keys = np.array([jax.random.PRNGKey(onp.random.choice(int(1e18))) for j in range(J)])
    # Multiply weights by measurement model result
    weights += dmeasures(ys[t], particlesP, thetas, keys=keys).squeeze() #shape (Np,)

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)

    # Sum up loglik
    loglik += loglik_t
'''

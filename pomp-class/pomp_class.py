import os 
import jax
import itertools
import numpy as onp
import jax.numpy as np
import ipywidgets as widgets
import ptitprince as pt
import pandas as pd

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain
from functools import partial

from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

'''resampling functions'''
def rinits_internal(rinit, thetas, J, covars):
    return rinit(thetas[0], len(thetas), covars)

def resample(norm_weights):
    J = norm_weights.shape[-1]
    unifs = (onp.random.uniform()+np.arange(J)) / J
    csum = np.cumsum(np.exp(norm_weights))
    counts = np.repeat(np.arange(J), 
                        np.histogram(unifs, 
                        bins=np.pad(csum/csum[-1], pad_width=(1,0)), 
                            density=False)[0].astype(int),
                        total_repeat_length=J)
    return counts


def normalize_weights(weights):
    mw = np.max(weights)
    loglik_t = mw + np.log(np.nansum(np.exp(weights - mw))) 
    norm_weights = weights - loglik_t
    return norm_weights, loglik_t
        
  
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

def resampler_pf(counts, particlesP, norm_weights):
    J = norm_weights.shape[-1]
    counts = resample(norm_weights)
    particlesF = particlesP[counts]
    return counts, particlesF, np.log(np.ones(J)) - np.log(J)


'''internal filtering functions - pt.1'''
def mop_helper(t, inputs, rprocess, dmeasure):
    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = inputs
    J = len(particlesF)
    if covars is not None and len(covars.shape) > 2:
            key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
            keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
            key, *keys = jax.random.split(key, num=J+1)
            keys = np.array(keys)
        
    weightsP = alpha*weightsF
        
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)
    else:
        particlesP = rprocess(particlesF, theta, keys, None)
        
    measurements = dmeasure(ys[t], particlesP, theta) 
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
        
    
    loglik += (jax.scipy.special.logsumexp(weightsP + measurements) 
                - jax.scipy.special.logsumexp(weightsP))

    norm_weights, loglik_phi_t = normalize_weights(jax.lax.stop_gradient(measurements))

    counts, particlesF, norm_weightsF = resampler(counts, particlesP, norm_weights)
    
    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    return [particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key]


@partial(jit, static_argnums=(2, 3, 4, 5))
def mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars=None, alpha=0.97, key=None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    weightsF = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0

    mop_helper_2 = partial(mop_helper, rprocess = rprocess, dmeasure = dmeasure)
    
    particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key = jax.lax.fori_loop(
            lower=0, upper=len(ys), body_fun=mop_helper_2, 
            init_val=[particlesF, theta, covars, loglik, weightsF, counts, ys, alpha, key])
    
    return -loglik

@partial(jit, static_argnums=(2, 3, 4, 5))
def mop_internal_mean(theta, ys, J, rinit, rprocess, dmeasure, covars=None, alpha=0.97, key=None):
    return mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key)/len(ys)


def pfilter_helper(t, inputs, rprocess, dmeasure):
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)
    
    if covars is not None and len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
        key, *keys = jax.random.split(key, num=J+1)
        keys = np.array(keys)
        
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys, None)
        
    measurements = dmeasure(ys[t], particlesP, theta)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
    
    weights = norm_weights + measurements

    norm_weights, loglik_t = normalize_weights(weights)
    loglik += loglik_t

    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    counts, particlesF, norm_weights = jax.lax.cond(oddr > thresh, 
                                               resampler, 
                                               no_resampler, 
                                               counts, particlesP, norm_weights)

    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]
    

    
    
@partial(jit, static_argnums=(2,3,4,5))
def pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars = None, thresh = 100, key = None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))

    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0
    
    pfilter_helper_2 = partial(pfilter_helper, rprocess = rprocess, dmeasure = dmeasure)
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=pfilter_helper_2, 
                 init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik

@partial(jit, static_argnums=(2,3,4,5))
def pfilter_internal_mean(theta, ys, J, rinit, rprocess, dmeasure, covars = None, thresh = 100, key = None):
    return pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key)/len(ys)

def perfilter_helper(t, inputs, rprocesses, dmeasures):
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)

    if len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
        key, *keys = jax.random.split(key, num=J+1)
        keys = np.array(keys)
    
    thetas += sigmas*np.array(onp.random.normal(size=thetas.shape))

    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars)
    else:
        particlesP = rprocesses(particlesF, thetas, keys)
        
    
    measurements = np.nan_to_num(dmeasures(ys[t], particlesP, thetas, keys=keys).squeeze(),
                    nan=np.log(1e-18))
    
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
     

    weights = norm_weights + measurements                         
    norm_weights, loglik_t = normalize_weights(weights)
    
    loglik += loglik_t
    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    counts, particlesF, norm_weights, thetas = jax.lax.cond(oddr > thresh, 
                                            resampler_thetas, 
                                            no_resampler_thetas, 
                                            counts, particlesP, norm_weights, thetas)
    
    
    return [particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key]

@partial(jit, static_argnums=(2, 4, 5, 6))
def perfilter_internal(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, covars=None, a = 0.95, thresh=100, key=None):
    loglik = 0
    thetas = theta + sigmas*onp.random.normal(size=(J, theta.shape[-1]))
    particlesF = rinits_internal(rinit, thetas, 1, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    perfilter_helper_2 = partial(perfilter_helper, rprocesses = rprocesses, dmeasures = dmeasures)
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
            lower=0, upper=len(ys), body_fun = perfilter_helper_2,
            init_val=[particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik, thetas

@partial(jit, static_argnums=(2, 4, 5, 6))
def perfilter_internal_mean(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, covars=None, a = 0.95, thresh=100, key=None):
    value, thetas = perfilter_internal(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, covars, thresh, key)
    return value/len(ys), thetas


def pfilter_helper_pf(t, inputs, rprocess, dmeasure):
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = inputs
    J = len(particlesF)
    
    if len(covars.shape) > 2:
        key, *keys = jax.random.split(key, num=J*covars.shape[1]+1)
        keys = np.array(keys).reshape(J, covars.shape[1], 2).astype(np.uint32)
    else:    
        key, *keys = jax.random.split(key, num=J+1)
        keys = np.array(keys)
        
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars)# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys, None)
        
    measurements = dmeasure(ys[t], particlesP, theta)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)
    
    weights = norm_weights + measurements

    norm_weights, loglik_t = normalize_weights(weights)
    loglik += loglik_t

    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    counts, particlesF, norm_weights = jax.lax.cond(oddr > thresh, 
                                            resampler_pf, 
                                            no_resampler, 
                                            counts, particlesP, norm_weights)

    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key]

@partial(jit, static_argnums=(2, 3, 4, 5))
def pfilter_pf_internal(theta, ys, J, rinit, rprocess, dmeasure, covars=None, thresh=100, key=None):
    if key is None:
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    loglik = 0
        
    pfilter_pf_helper_2 = partial(pfilter_helper_pf, rprocess = rprocess, dmeasure = dmeasure)

    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key = jax.lax.fori_loop(
            lower=0, upper=len(ys), body_fun=pfilter_pf_helper_2,
            init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh, key])
    
    return -loglik/len(ys) 

'''gradient functions'''
def line_search(obj, curr_obj, pt, grad, direction, k=1, eta=0.9, xi=10, tau = 10, c=0.1, frac=0.5, stoch=False):
    itn = 0
    eta = min([eta, xi/k]) if stoch else eta #if stoch if false, do not change
    next_obj = obj(pt + eta*direction)
    # check whether the new point(new_obj)satisfies the stochastic Armijo condition
    # if not, repeat until the condition is met
    while next_obj > curr_obj + eta*c*grad.T @ direction or np.isnan(next_obj):
        eta *= frac
        itn += 1
        if itn > tau: 
            break
    return eta
    
@partial(jit, static_argnums=(2, 3, 4, 5))
def jgrad_pf(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    return jax.grad(pfilter_pf_internal)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh, key=key)

#return the value and gradient at the same time
@partial(jit, static_argnums=(2, 3, 4, 5))
def jvg_pf(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    return jax.value_and_grad(pfilter_pf_internal)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh, key=key)

@partial(jit, static_argnums=(2, 3, 4, 5))
def jgrad(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    return jax.grad(pfilter_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh, key=key)

@partial(jit, static_argnums=(2, 3, 4, 5))
def jvg(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    return jax.value_and_grad(pfilter_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh, key=key)

@partial(jit, static_argnums=(2, 3, 4, 5))
def jgrad_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None):
    return jax.grad(mop_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)

@partial(jit, static_argnums=(2, 3, 4, 5))
def jvg_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None):
    return jax.value_and_grad(mop_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)

#get the hessian matrix from pfilter
@partial(jit, static_argnums=(2, 3, 4, 5))
def jhess(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    return jax.hessian(pfilter_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh, key=key)

#get the hessian matrix from mop
@partial(jit, static_argnums=(2, 3, 4, 5))
def jhess_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha, key=None):
        return jax.hessian(mop_internal_mean)(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha = alpha, key=key)


'''internal filtering functions - pt.2'''
MONITORS = 1

def mif_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init, covars=None, M=10, 
    a=0.95, J=100, thresh=100, monitor=False, verbose=False):
    
    logliks = []
    params = []
    
    thetas = theta + sigmas_init*onp.random.normal(size=(J, theta.shape[-1]))
    params.append(thetas)
    if monitor:
        loglik = np.mean(np.array([pfilter_internal(thetas.mean(0), ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh) 
                                for i in range(MONITORS)]))
        logliks.append(loglik)
        
    
    for m in tqdm(range(M)):
        sigmas *= a
        thetas += sigmas*onp.random.normal(size=thetas.shape)
        loglik_ext, thetas = perfilter_internal(thetas, ys, J, sigmas, rinit, rprocesses, dmeasures, covars=covars, a=a, thresh=thresh)
        
        params.append(thetas)
        
        if monitor:
            loglik = np.mean(np.array([pfilter_internal(thetas.mean(0), ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh) 
                                        for i in range(MONITORS)]))
            logliks.append(loglik)
                   
            if verbose:
                print(loglik)
                print(thetas.mean(0))
        
    return np.array(logliks), np.array(params)

def train_internal(theta_ests, ys, rinit, rprocess, dmeasure, covars=None, J=5000, Jh=1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, max_ls_itn=10, thresh=100, verbose=False, scale=False, ls=False, alpha=1): 
    Acopies = []
    grads = []
    hesses = []
    logliks = []
    hess = np.eye(theta_ests.shape[-1])
    
    
    for i in tqdm(range(itns)):
        print("new iteration")
        key = jax.random.PRNGKey(onp.random.choice(int(1e18)))
        if MONITORS == 1:
            print("!")
            loglik, grad = jvg_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)
            print("Value type:", type(loglik))
            print("Grad type:", type(grad))
            loglik *= len(ys) 
        else:
            grad = jgrad_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, alpha=alpha, key=key)
            loglik = np.mean(np.array([pfilter_internal(theta_ests, ys, J, rinit, rprocess, dmeasure, 
                                                covars=covars, thresh=-1, key=key) 
                                        for i in range(MONITORS)])) 
     
        if method=='Newton':
            hess = jhess_mop(theta_ests, ys, Jh, rinit, rprocess, dmeasure, covars=covars, alpha = alpha, key=key)
            direction = -np.linalg.pinv(hess) @ grad
        elif method == 'WeightedNewton':
            if i == 0:
                hess = jhess(theta_ests, ys, Jh, rinit, rprocess, dmeasure, covars=covars, thresh=thresh, key=key)
                direction = -np.linalg.pinv(hess) @ grad
            else:
                hess = jhess(theta_ests, ys, Jh, rinit, rprocess, dmeasure, covars=covars, thresh=thresh, key=key)
                wt = (i**onp.log(i))/((i+1)**(onp.log(i+1)))
                direction = -np.linalg.pinv(wt * hesses[-1] + (1-wt) * hess) @ grad

        elif method=='BFGS' and i > 1:
            s_k = et * direction 
            y_k = grad - grad[-1] 
            rho_k = np.reciprocal(np.dot(y_k, s_k))
            sy_k = s_k[:, np.newaxis] * y_k[np.newaxis, :]
            w = np.eye(theta_ests.shape[-1], dtype=rho_k.dtype) - rho_k * sy_k
            # H_(k+1) = W_k^T@H_k@W_k + pho_k@s_k@s_k^T 
            hess = (np.einsum('ij,jk,lk', w, hess, w)
                        + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :])
            hess = np.where(np.isfinite(rho_k), hess, hess) 
            direction = -hess @ grad 
        else:
            direction = -grad
            
        Acopies.append(theta_ests) 
        logliks.append(loglik)
        grads.append(grad)
        hesses.append(hess)
            
        if scale:
            direction = direction/np.linalg.norm(direction)
        
        eta = line_search(partial(pfilter_internal, ys=ys, J=J, rinit=rinit, rprocess = rprocess, dmeasure = dmeasure, covars=covars, thresh=thresh, key=key), 
                          loglik, theta_ests, grad, direction, k=i+1, eta=beta, c=c, tau=max_ls_itn) if ls else eta
        try:
            et = eta if len(eta) == 1 else eta[i]
        except: 
            et = eta
        if i%1==0 and verbose:
            print(theta_ests, et, logliks[i])

        theta_ests += et*direction
    logliks.append(np.mean(np.array([pfilter_internal(theta_ests, ys, J, rinit, rprocess, dmeasure, covars=covars, thresh=thresh) for i in range(MONITORS)])))
    Acopies.append(theta_ests)
    
    return np.array(logliks), np.array(Acopies)

def fit_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init, covars = None, M = 10, a = 0.9, 
        J =100, Jh = 1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, 
        max_ls_itn=10, thresh_mif = 100, thresh_tr = 100, verbose = False, scale = False, ls = False, alpha = 0.1):
        
    mif_logliks_warm, mif_params_warm = mif_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, 
                                sigmas_init, covars, M , a , J, thresh_mif,  monitor = True, verbose = verbose) 
    theta_ests = mif_params_warm[mif_logliks_warm.argmin()].mean(0)
    gd_logliks, gd_ests = train_internal(theta_ests, ys, rinit, rprocess, dmeasure, covars, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_tr, verbose, scale, ls, alpha)
        
    return np.array(gd_logliks), np.array(gd_ests)

'''OOP layer - pomp class'''
class Pomp:
    MONITORS = 1

    def __init__(self, rinit, rproc, dmeas, ys, theta, covars=None):
        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.ys = ys
        self.theta = theta
        self.covars = covars
        self.rprocess = jax.vmap(self.rproc, (0, None, 0, None))
        self.rprocesses = jax.vmap(rproc, (0, 0, 0, None))
        self.dmeasure = jax.vmap(self.dmeas, (None,0,None))
        self.dmeasures = jax.vmap(self.dmeas, (None,0,0))
    
    def rinits(self, thetas, J, covars):
        return rinits_internal(self.rinit, thetas, J, covars)

    def mop(self, J, alpha=0.97, key=None):
        return mop_internal(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha, key)

    def mop_mean(self, J, alpha=0.97, key=None):
        return mop_internal_mean(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha, key)
    
    def pfilter(self, J, thresh=100, key=None):
        return pfilter_internal(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh, key)
        
    
    def pfilter_mean(self, J, thresh=100, key=None):
        return pfilter_internal_mean(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh, key)
    
    def perfilter(self, J, sigmas, a = 0.9, thresh=100, key=None):
    
        return perfilter_internal(self.theta, self.ys, J, sigmas, self.rinit, self.rprocesses, self.dmeasures, self.covars, a, thresh, key)
    
    def perfilter_mean(self, J, sigmas, a = 0.9, thresh=100, key=None):
    
        return perfilter_internal_mean(self.theta, self.ys, J, sigmas, self.rinit, self.rprocesses, self.dmeasures, self.covars, a, thresh, key)
    

    def pfilter_pf(self, J, thresh=100, key=None):
        return pfilter_pf_internal(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh, key)
    
    
    def mif(self, sigmas, sigmas_init, M=10, a=0.9, J=100, thresh=100, monitor=False, verbose=False):
    
        return mif_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                            sigmas, sigmas_init, self.covars, M, a, J, thresh, monitor, verbose)

    def train(self, theta_ests, J=5000, Jh=1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, max_ls_itn=10, thresh=100, verbose=False, scale=False, ls=False, alpha=1): 
           
        return train_internal(theta_ests, self.ys, self.rinit, self.rprocess, self.dmeasure, self.covars, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh, verbose, scale, ls, alpha)
        

    def fit(self, sigmas, sigmas_init, M = 10, a = 0.9, 
            J =100, Jh = 1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, 
            max_ls_itn=10, thresh_mif = 100, thresh_tr = 100, verbose = False, scale = False, ls = False, alpha = 0.1):
        
        return fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures, sigmas, sigmas_init, self.covars, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, 
                           thresh_tr, verbose, scale, ls, alpha)
    
def pfilter(pomp_object = None, J = 50, rinit = None, rprocess = None, dmeasure = None, theta = None, ys = None, covars = None, thresh = 100, key = None):
    if pomp_object is not None:
        return pomp_object.pfilter(J, thresh, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key)
    else:
        raise ValueError("Invalid Arguments Input")
    
def perfilter(pomp_object = None, J = 50, rinit = None, rprocesses = None, dmeasures = None, theta = None, ys = None, sigmas = None, covars = None, thresh = 100, key = None):
    if pomp_object is not None:
        return pomp_object.perfilter(J, sigmas, thresh, key)
    elif rinit is not None and rprocesses is not None and dmeasures is not None and theta is not None and ys is not None:
        return perfilter_internal(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, covars, thresh, key)
    else:
        raise ValueError("Invalid Arguments Input")    
    
def mop(pomp_object = None, J = 50, rinit = None, rprocess = None, dmeasure = None, theta = None, ys = None, covars = None,  alpha = 0.97, key = None):
    if pomp_object is not None:
        return pomp_object.mop( J, alpha, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key)
    else:
        raise ValueError("Invalid Arguments Input")    
    
def pfilter_pf(pomp_object = None, J = 50, rinit = None, rprocess = None, dmeasure = None, theta = None, ys = None, covars = None, thresh = 100, key = None):
    if pomp_object is not None:
        return pomp_object.pfilter_pf(J, thresh, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return pfilter_pf_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key)
    else:
        raise ValueError("Invalid Arguments Input")   

def mif(pomp_object = None, J = 50, rinit = None, rprocess = None, dmeasure = None, rprocesses = None, dmeasures = None, theta = None, ys = None, sigmas = None, sigmas_init = None, 
        covars = None, M = None, a = None, thresh = 100, monitor = False, verbose = False):
    if pomp_object is not None:
        return pomp_object.mif(sigmas, sigmas_init, M, a, J, thresh, monitor, verbose)
    elif rinit is not None and rprocess is not None and dmeasure is not None and rprocesses is not None and dmeasures is not None and theta is not None and ys is not None:
        return mif_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init, covars, M, a, J, thresh, monitor, verbose)
    else:
        raise ValueError("Invalid Arguments Input")   

def train(pomp_object = None, J = 50, theta_ests = None, rinit = None, rprocess = None, dmeasure = None, ys = None, covars = None, Jh = 1000, method = 'Newton', itns = 20, beta = 0.9, 
          eta = 0.0025, c = 0.1, max_ls_itn = 10, thresh = 100, verbose = False, scale = False, ls = False, alpha = 1):
    if pomp_object is not None and theta_ests is not None:
        return pomp_object.train(theta_ests, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh, verbose, scale, ls, alpha)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta_ests is not None and ys is not None:
        return train_internal(theta_ests, ys, rinit, rprocess, dmeasure, covars, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh, verbose, scale, ls, alpha)
    else:
        raise ValueError("Invalid Arguments Input")   

def fit(pomp_object = None, J = 100, Jh = 1000, rinit = None, rprocess = None, dmeasure = None, rprocesses = None, dmeasures = None, theta = None, ys = None, sigmas = None , sigmas_init = None, covars = None, M = 10, a = 0.9, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, 
        max_ls_itn=10, thresh_mif = 100, thresh_tr = 100, verbose = False, scale = False, ls = False, alpha = 0.1):
    if pomp_object is not None and sigmas is not None and sigmas_init is not None:
        return pomp_object.fit(sigmas, sigmas_init, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, thresh_tr, verbose, scale, ls, alpha)
    elif rinit is not None and rprocess is not None and dmeasure is not None and rprocesses is not None and dmeasures is not None and theta is not None and ys is not None and sigmas is not None and sigmas_init is not None:
        return fit_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init, covars, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, thresh_tr, verbose, scale, ls, alpha)
    else:
        raise ValueError("Invalid Arguments Input")   
    
    


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

from internal_functions import *

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
    
        return perfilter_internal(self.theta, self.ys, J, sigmas, self.rinit, self.rprocesses, self.dmeasures, ndim = self.theta.ndim, covars = self.covars, a = a, thresh = thresh, key = key)
    
    def perfilter_mean(self, J, sigmas, a = 0.9,thresh=100, key=None):
    
        return perfilter_internal_mean(self.theta, self.ys, J, sigmas, self.rinit, self.rprocesses, self.dmeasures, ndim = self.theta.dim, covars = self.covars, a = a, thresh = thresh, key = key)
    

    def pfilter_pf(self, J, thresh=100, key=None):
        return pfilter_pf_internal(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh, key)
    
    
    def mif(self, sigmas, sigmas_init, M=10, a=0.9, J=100, thresh=100, monitor=False, verbose=False):
    
        return mif_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                            sigmas, sigmas_init, self.covars, M, a, J, thresh, monitor, verbose)

    def train(self, theta_ests, J=5000, Jh=1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, max_ls_itn=10, thresh=100, verbose=False, scale=False, ls=False, alpha=1): 
           
        return train_internal(theta_ests, self.ys, self.rinit, self.rprocess, self.dmeasure, self.covars, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh, verbose, scale, ls, alpha)
        

    def fit(self, sigmas, sigmas_init, M = 10, a = 0.9, 
            J =100, Jh = 1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, 
            max_ls_itn=10, thresh_mif = 100, thresh_tr = 100, verbose = False, scale = False, ls = False, alpha = 0.1, mode = "IFAD"):
        
        return fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures, sigmas, sigmas_init, self.covars, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, 
                           thresh_tr, verbose, scale, ls, alpha, monitor = True, mode = mode)
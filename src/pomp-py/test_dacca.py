
# note: copy pomps_dacca.py to pomps.py before running this with
# python3 test_dacca.py 

# this file is adapted from cholera.ipynb 
# jupyter nbconvert --to script cholera.ipynb 
# [NbConvertApp] Converting notebook cholera.ipynb to script
# the plotting routines are excluded

n_trials = 2
J = 50
Iterations = 2

import os
import jax
import itertools
import numpy as onp
import ptitprince as pt

import jax.numpy as np
import ipywidgets as widgets
import pandas as pd

from jax.numpy.linalg import inv, pinv
from jax.scipy.optimize import minimize
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain
from functools import partial

from tqdm.notebook import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

from pomps import *
from resampling import *
from filtering import *
from optim import *

import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')
onp.set_printoptions(suppress=True)

dataset = pd.read_csv('data/dacca.csv', index_col=0).reset_index(drop=True)
ys = np.array(dataset['cholera.deaths'].values)
dataset = pd.read_csv('data/dacca-covars.csv', index_col=0).reset_index(drop=True)
dataset.index = pd.read_csv('data/dacca-covart.csv', index_col=0).reset_index(drop=True).squeeze()
dataset = dataset.reindex(onp.array([1891 + i*(1/240) for i in range(12037)])).interpolate()
covars = np.array(dataset.values)

gamma = 20.8
epsilon = 19.1
rho = 0
delta = 0.02
m = 0.06
c = np.array(1)
beta_trend = -0.00498
bs = np.array([0.747, 6.38, -3.44, 4.23, 3.33, 4.55])
sigma = 3.13 #3.13 # 0.77
tau = 0.23 
omega = onp.exp(-4.5)
omegas = np.log(np.array([0.184, 0.0786, 0.0584, 0.00917, 0.000208, 0.0124]))


theta = transform_thetas(gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, omegas)

pfilter(theta, ys, J, covars, thresh=-1)

mop(theta, ys, J, covars, alpha=0.5)

def get_rand_theta():
    return transform_thetas(onp.random.uniform(10.00, 40.00),
                onp.random.uniform(0.03, 0.60), 
                         rho, 
                 onp.random.uniform(0.20, 30.00), 
                         omega, 
                         c, 
                 onp.random.uniform(-1.00, 0.00)*0.01, 
                 onp.random.uniform(1.00, 5.00), 
                 onp.random.uniform(0.10, 0.50), 
                 onp.random.uniform(0,8,size=6)+np.array([-4,0,-4,0,0,0]), 
                 onp.random.uniform(-10,0,size=6))

def get_sds():    
    lows = transform_thetas(10.00,0.03, rho, 0.20, omega, c, 
             -1.00*0.01, 1.00, 0.10, 
             onp.zeros(6)+onp.array([-4,0,-4,0,0,0]), 
            -10*onp.ones(6))
    highs = transform_thetas(40.00,0.60, rho, 30.00, omega, c, 
                 0.00, 5.00, 0.50, 
                 8*onp.ones(6)+onp.array([-4,0,-4,0,0,0]), 
                onp.zeros(6))
    return (highs-lows)/100


def get_rand_theta(J=J):
    lows = transform_thetas(10.00,0.03, rho, 0.20, omega, c, 
             -1.00*0.01, 1.00, 0.10, 
             onp.zeros(6)+onp.array([-4,0,-4,0,0,0]), 
            -10*onp.ones(6))
    highs = transform_thetas(40.00,0.60, rho, 30.00, omega, c, 
                 0.00, 5.00, 0.50, 
                 8*onp.ones(6)+onp.array([-4,0,-4,0,0,0]), 
                onp.zeros(6))
    rands = onp.array(onp.repeat(((lows+highs)/2)[None,:], J, axis=0)).T
    rands[~onp.isinf(lows)] = onp.random.uniform(lows[~onp.isinf(lows)], 
                       highs[~onp.isinf(highs)],
                       size=(J, len(highs[~onp.isinf(highs)]))).T
    return rands.T
                       
                       
sigmas = (np.abs(theta)/600)
theta_ests = theta + 60*sigmas*onp.random.normal(size=theta.shape)

print(pfilter(theta_ests, ys, J, covars, thresh=-1))

def log_in_bbox(theta):    
    valids = np.array([i for i in range(len(theta)) if i not in [2,5]])
    lows = transform_thetas(10.00,0.03, rho, 0.20, omega, c, 
             -1.00*0.01, 1.00, 0.10, 
             onp.zeros(6)+onp.array([-4,0,-4,0,0,0]), 
            -10*onp.ones(6))
    highs = transform_thetas(40.00,0.60, rho, 30.00, omega, c, 
                 0.00, 5.00, 0.50, 
                 8*onp.ones(6)+onp.array([-4,0,-4,0,0,0]), 
                onp.zeros(6))
    return -100*np.log(1-(np.logical_or(np.any(lows[valids] > theta[valids]),
                                        np.any(theta[valids] > highs[valids])))+1e-43)

log_in_bbox(theta+9999)

close = False

original_logliks = []
original_theta_ests = []

mif_logliks_trials = []
mif_params_trials = []

mif_logliks_warm_trials = []
mif_params_warm_trials = []

gd_logliks_trials = []
gd_ests_trials = []

gd_logliks_pf_trials = []
gd_ests_pf_trials = []

gd_logliks_mop_trials = []
gd_ests_mop_trials = []

gd_logliks_raw_trials = []
gd_ests_raw_trials = []
gd_logliks_pf_raw_trials = []
gd_ests_pf_raw_trials = []

for trial in tqdm(range(n_trials)):
    
    if close:
        sigmas = (np.abs(theta)/600)
        theta_ests = theta + 60*sigmas*onp.random.normal(size=theta.shape) 
        orig_loglik = pfilter(theta_ests, ys, J, covars, thresh=-1)
        gd_logliks, gd_ests = train(theta_ests, ys, covars, beta=0.9, 
                            eta=np.flip(np.linspace(0.0001,0.1,n_trials)), 
                            verbose=False, itns=Iterations, J=J, thresh=100, 
                            method='SGD', ls=False, scale=True)
    else:
        bbox = get_rand_theta(1).squeeze()
        orig_loglik = pfilter(bbox, ys, J, covars, thresh=-1)
        mif_logliks_warm, mif_params_warm = mif(bbox, ys, sigmas=0.02, 
                                  sigmas_init = 1e-20, covars=covars, verbose=False,
                                  M=Iterations, J=J, a=0.95, monitor=True, thresh=-1) 
        theta_ests = mif_params_warm[mif_logliks_warm.argmin()].mean(0)
        
        
        gd_logliks, gd_ests = train(theta_ests, ys, covars, beta=0.9, 
                            eta=0.01, 
                            verbose=False, itns=Iterations, J=J, thresh=100, 
                            method='SGD', ls=False, scale=False, alpha=1)  
        gd_logliks_pf, gd_ests_pf = train(theta_ests, ys, covars, beta=0.9, 
                            eta=0.05, 
                            verbose=False, itns=Iterations, J=J, thresh=100, 
                            method='SGD', ls=False, scale=False, alpha=0) 
        gd_logliks_mop, gd_ests_mop = train(theta_ests, ys, covars, beta=0.9, 
                            eta=0.2, 
                            verbose=False, itns=Iterations, J=J, thresh=100, 
                            method='SGD', ls=False, scale=False, alpha=0.97)
                
        mif_logliks, mif_params = mif(bbox, ys, sigmas=0.02, 
                                  sigmas_init = 1e-20, covars=covars, verbose=False,
                                  M=Iterations, J=J, a=0.95, monitor=True, thresh=-1)
        
    original_logliks.append(orig_loglik)
    original_theta_ests.append(theta_ests)
    
    
 

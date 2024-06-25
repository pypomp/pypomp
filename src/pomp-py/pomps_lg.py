


import jax
import itertools
import numpy as onp
import jax.numpy as np
import ipywidgets as widgets

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


def sigmoid(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))

def get_thetas(theta):
    A = theta[0]
    C = theta[1]
    Q = theta[2]
    R = theta[3]
    return A, C, Q, R

def transform_thetas(theta):
    return np.array([A, C, Q, R])

def rproc(state, theta, key, covars=None):
    A, C, Q, R = get_thetas(theta)
    return jax.random.multivariate_normal(key=key, 
                                          mean=A@state, cov=Q)

rprocesses = jax.jit(jax.vmap(rproc, (0, 0, 0, None)))
rprocess = jax.jit(jax.vmap(rproc, (0, None, 0, None)))

def dmeas(y, preds, theta):
    A, C, Q, R = get_thetas(theta)
    return jax.scipy.stats.multivariate_normal.logpdf(y, preds, R)

dmeasure = jax.vmap(dmeas, (None,0,None))
dmeasures = jax.vmap(dmeas, (None,0,0))


def rinit(theta, J, covars=None):
    return np.ones((J, 2))


def rinits(thetas, J, covars=None):
    return rinit(thetas[0], len(thetas), covars)
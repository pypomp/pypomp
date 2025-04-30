import os
import csv
import jax
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp

def get_thetas(theta):
    """
    Cast a theta vector into A, C, Q, and R matrices as if casting iron.
    """
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return A, C, Q, R


def transform_thetas(A, C, Q, R):
    """
    Take A, C, Q, and R matrices and melt them into a single 1D array.
    """
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])


fixed = False
key = jax.random.PRNGKey(111)
angle = 0.2
angle2 = angle if fixed else -0.5
A = jnp.array([[jnp.cos(angle2), -jnp.sin(angle)],
                [jnp.sin(angle), jnp.cos(angle2)]])
C = jnp.eye(2)
Q = jnp.array([[1, 1e-4],
               [1e-4, 1]]) / 100
R = jnp.array([[1, .1],
               [.1, 1]]) / 10
theta =  transform_thetas(A, C, Q, R)
covars = None

def Generate_data(T = 4, key = key):
    xs = []
    ys = []
    for i in tqdm(range(T)):
        x = jnp.ones(2)
        key, subkey = jax.random.split(key)
        x = jax.random.multivariate_normal(key=subkey, mean=A @ x, cov=Q)
        key, subkey = jax.random.split(key)
        y = jax.random.multivariate_normal(key=subkey, mean=C @ x, cov=R)
        xs.append(x)
        ys.append(y)
    xs = jnp.array(xs)
    ys = jnp.array(ys)
    return ys

def rinit(theta, J, covars=None):
    return jnp.ones((J, 2))

def rproc(state, theta, key, covars=None):
    A, C, Q, R = get_thetas(theta)
    key, subkey = jax.random.split(key)
    return jax.random.multivariate_normal(
        key=subkey,
        mean=A @ state, cov=Q
    )
    
def dmeas(y, preds, theta):
    A, C, Q, R = get_thetas(theta)
    return jax.scipy.stats.multivariate_normal.logpdf(y, preds, R)

rprocess = jax.vmap(rproc, (0, None, 0, None))
dmeasure = jax.vmap(dmeas, (None, 0, None))
rprocesses = jax.vmap(rproc, (0, 0, 0, None))
dmeasures = jax.vmap(dmeas, (None, 0, 0))

def LG_internal(T=4):
    ys = Generate_data(T=T, key=key)
    LG_obj = Pomp(rinit, rproc, dmeas, ys, theta, covars)
    return LG_obj, ys, theta, covars, rinit, rproc, dmeas, rprocess, dmeasure, rprocesses, dmeasures

def LG(T=4):
    """
    Initialize a Pomp object with the linear Gaussian model.

    Parameters
    ----------
    T : int, optional
        The number of time steps to generate data for. Defaults to 4.

    Returns
    -------
    LG_obj : Pomp
        A Pomp object initialized with the linear Gaussian model parameters and
        the generated data.
    """
    ys = Generate_data(T=T, key=key)
    LG_obj = Pomp(rinit, rproc, dmeas, ys, theta, covars)
    return LG_obj

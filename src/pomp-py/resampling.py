

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
from functools import partial

from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')



    
def resample(norm_weights):
    J = norm_weights.shape[-1]
    #counts = jax.random.categorical(key=jax.random.PRNGKey(randint), 
    #                   logits=jax.lax.stop_gradient(norm_weights),
    #                    shape=(J,))
    
    #unifs = 0.5+np.linspace(0,1,J)
    #unifs = np.where(unifs>=1, x=unifs-1, y=unifs) #when true x, else y
    
    #ARCHIVE
    #unifs = unifs.at[unifs>=1].set(unifs[unifs>=1]-1)
    
    unifs = (onp.random.uniform()+np.arange(J)) / J
    
    csum = np.cumsum(np.exp(norm_weights))
    counts = np.repeat(np.arange(J), 
                       np.histogram(unifs, 
                        bins=np.pad(csum/csum[-1], pad_width=(1,0)), 
                            density=False)[0].astype(int),
                      total_repeat_length=J)
    
    #if len(counts)<J:
    #    counts = np.hstack([counts, np.zeros(J-len(counts))]).astype(int)
    return counts


def normalize_weights(weights):
    mw = np.max(weights)
    loglik_t = mw + np.log(np.nansum(np.exp(weights - mw))) # p(y_t | x_{t,1:J}, \theta)
    norm_weights = weights - loglik_t
    return norm_weights, loglik_t



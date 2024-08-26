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

from src.internal_functions import *
from src.pomp_class import *

def mop(pomp_object = None, J = 50, rinit = None, rprocess = None, dmeasure = None, theta = None, ys = None, covars = None,  alpha = 0.97, key = None):
    if pomp_object is not None:
        return pomp_object.mop(J, alpha, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key)
    else:
        raise ValueError("Invalid Arguments Input")    
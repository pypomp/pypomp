
# note: copy pomps_ly.py to pomps.py before running this with
# python3 test_lg.py > test_lg.out

# this file is adapted from pfilterLGTest.ipynb 
# jupyter nbconvert --to script pfilterLGTest.ipynb 
# [NbConvertApp] Converting notebook pfilterLGTest.ipynb to script

import jax
import itertools
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')
import ipywidgets as widgets

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

from pomps import *
from resampling import *
from filtering import *
from optim import *

from tqdm import tqdm

fixed = False

angle = 0.2
angle2 = angle if fixed else -0.5
A = np.array([[np.cos(angle2), -np.sin(angle)], 
              [np.sin(angle), np.cos(angle2)]])
C = np.eye(2)
Q = np.array([[1, 1e-4], 
              [1e-4, 1]])/100
R = np.array([[1, .1], 
              [.1, 1]])/10
x = np.ones(2)

xs = []
ys = []
T = 4
J = 100
for i in tqdm(range(T)):
    randint = onp.random.randint(0, 10000)
    x = jax.random.multivariate_normal(key=jax.random.PRNGKey(randint), mean=A@x, cov=Q)
    randint = onp.random.randint(0, 10000)
    y = jax.random.multivariate_normal(key=jax.random.PRNGKey(randint), mean=C@x, cov=R)
    xs.append(x)
    ys.append(y)

xs = np.array(xs)
ys = np.array(ys)
# plt.figure(figsize=(16,9))
# plt.plot(xs)
# plt.plot(ys)

loss_grad = (jax.vmap(jax.grad(dmeasure, argnums=1), in_axes=(None,0,None)))

import pykalman
kf = pykalman.KalmanFilter(transition_matrices=A, observation_matrices=C, 
                      transition_covariance=Q, observation_covariance=R)#.filter(ys)
print("kf loglik =", kf.loglikelihood(ys))

theta = np.array([A, C, Q, R])
pfilter_loglik = -pfilter(theta, ys, J, None, 0)
print("pfilter loglik = ", pfilter_loglik, " (J=", J,")", sep="")


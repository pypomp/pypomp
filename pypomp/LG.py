"""This module implements a linear Gaussian model for POMP."""

from functools import partial
import jax
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.model_struct import RInit
from pypomp.model_struct import RProc
from pypomp.model_struct import DMeas
from pypomp.model_struct import RMeas


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


# TODO: Replace this with a simulate function.
def Generate_data(
    T=4,
    A=jnp.array([[jnp.cos(0.2), -jnp.sin(0.2)], [jnp.sin(0.2), jnp.cos(0.2)]]),
    C=jnp.eye(2),
    Q=jnp.array([[1, 1e-4], [1e-4, 1]]) / 100,
    R=jnp.array([[1, 0.1], [0.1, 1]]) / 10,
    key=jax.random.PRNGKey(111),
):
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


# TODO: Add custom starting position.
@RInit
def rinit(theta_, key, covars=None, t0=None):
    """Initial state process simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.random.multivariate_normal(key=key, mean=jnp.array([0, 0]), cov=Q)


@RProc
def rproc(X_, theta_, key, covars=None, t=None, dt=None):
    """Process simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.random.multivariate_normal(key=key, mean=A @ X_, cov=Q)


@DMeas
def dmeas(Y_, X_, theta_, covars=None, t=None):
    """Measurement model distribution for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.scipy.stats.multivariate_normal.logpdf(Y_, X_, R)


@partial(RMeas, ydim=4)
def rmeas(X_, theta_, key, covars=None, t=None):
    """Measurement simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.random.multivariate_normal(key=key, mean=C @ X_, cov=R)


def LG(
    T=4,
    A=jnp.array([[jnp.cos(0.2), -jnp.sin(0.2)], [jnp.sin(0.2), jnp.cos(0.2)]]),
    C=jnp.eye(2),
    Q=jnp.array([[1, 1e-4], [1e-4, 1]]) / 100,
    R=jnp.array([[1, 0.1], [0.1, 1]]) / 10,
    key=jax.random.key(111),
):
    """
    Initialize a Pomp object with the linear Gaussian model.

    Parameters
    ----------
    T : int, optional
        The number of time steps to generate data for. Defaults to 4.
    A : array-like, optional
        The transition matrix. Defaults to the identity matrix.
    C : array-like, optional
        The measurement matrix. Defaults to the identity matrix.
    Q : array-like, optional
        The covariance matrix of the state noise. Defaults to the identity
        matrix.
    R : array-like, optional
        The covariance matrix of the measurement noise. Defaults to the identity
        matrix.
    key : PRNGKey, optional
        The random key used to generate the data. Defaults to
        jax.random.PRNGKey(111).

    Returns
    -------
    LG_obj : Pomp
        A Pomp object initialized with the linear Gaussian model parameters and
        the generated data.
    """
    theta_names = [f"{name}{i}" for name in "ACQR" for i in range(1, 5)]
    theta = dict(zip(theta_names, transform_thetas(A, C, Q, R).tolist()))

    covars = None
    ys = Generate_data(T=T, A=A, C=C, Q=Q, R=R, key=key)
    LG_obj = Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=ys,
        theta=theta,
        covars=covars,
    )
    return LG_obj

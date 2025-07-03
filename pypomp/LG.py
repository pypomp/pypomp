"""This module implements a linear Gaussian model for POMP."""

from functools import partial
import jax
import jax.numpy as jnp
import pandas as pd

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


# TODO: Add custom starting position.
@partial(RInit, t0=0.0)
def rinit(theta_, key, covars=None, t0=None):
    """Initial state process simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.random.multivariate_normal(key=key, mean=jnp.array([0, 0]), cov=Q)


@partial(RProc, step_type="fixedstep", nstep=1)
def rproc(X_, theta_, key, covars=None, t=None, dt=None):
    """Process simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.random.multivariate_normal(key=key, mean=A @ X_, cov=Q)


@DMeas
def dmeas(Y_, X_, theta_, covars=None, t=None):
    """Measurement model distribution for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.scipy.stats.multivariate_normal.logpdf(Y_, X_, R)


@partial(RMeas, ydim=2)
def rmeas(X_, theta_, key, covars=None, t=None):
    """Measurement simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    return jax.random.multivariate_normal(key=key, mean=C @ X_, cov=R)


def LG(
    T: int = 4,
    A: jax.Array = jnp.array(
        [[jnp.cos(0.2), -jnp.sin(0.2)], [jnp.sin(0.2), jnp.cos(0.2)]]
    ),
    C: jax.Array = jnp.eye(2),
    Q: jax.Array = jnp.array([[1, 1e-4], [1e-4, 1]]) / 100,
    R: jax.Array = jnp.array([[1, 0.1], [0.1, 1]]) / 10,
    key: jax.Array = jax.random.key(111),
):
    """
    Initialize a Pomp object with the linear Gaussian model.

    Parameters
    ----------
    T : int, optional
        The number of time steps to generate data for. Defaults to 4.
    A : jax.Array, optional
        The transition matrix. Defaults to the identity matrix.
    C : jax.Array, optional
        The measurement matrix. Defaults to the identity matrix.
    Q : jax.Array, optional
        The covariance matrix of the state noise. Defaults to the identity
        matrix.
    R : jax.Array, optional
        The covariance matrix of the measurement noise. Defaults to the identity
        matrix.
    key : jax.Array, optional
        The random key used to generate the data. Defaults to
        jax.random.key(111).

    Returns
    -------
    LG_obj : Pomp
        A Pomp object initialized with the linear Gaussian model parameters and
        the generated data.
    """
    theta_names = [f"{name}{i}" for name in "ACQR" for i in range(1, 5)]
    theta = dict(zip(theta_names, transform_thetas(A, C, Q, R).tolist()))

    ys_temp = pd.DataFrame(0, index=range(1, T + 1), columns=pd.Index(["Y1", "Y2"]))

    LG_obj_temp = Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=ys_temp,
        theta=theta,
        covars=None,
    )
    sims = LG_obj_temp.simulate(key=key)

    LG_obj = Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=pd.DataFrame(
            sims[0]["Y_sims"].squeeze(),
            index=range(1, T + 1),
            columns=pd.Index(["Y1", "Y2"]),
        ),
        theta=theta,
        covars=None,
    )

    return LG_obj

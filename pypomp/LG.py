"""This module implements a linear Gaussian model for POMP."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pypomp.pomp_class import Pomp


def get_thetas(theta):
    """
    Cast a theta vector into A, C, Q, and R matrices as if casting iron.
    """
    A = jnp.array([theta["A1"], theta["A2"], theta["A3"], theta["A4"]]).reshape(2, 2)
    C = jnp.array([theta["C1"], theta["C2"], theta["C3"], theta["C4"]]).reshape(2, 2)
    Q = jnp.array([theta["Q1"], theta["Q2"], theta["Q3"], theta["Q4"]]).reshape(2, 2)
    R = jnp.array([theta["R1"], theta["R2"], theta["R3"], theta["R4"]]).reshape(2, 2)
    return A, C, Q, R


def transform_thetas(A, C, Q, R):
    """
    Take A, C, Q, and R matrices and melt them into a single 1D array.
    """
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])


# TODO: Add custom starting position.
def rinit(theta_, key, covars=None, t0=None):
    """Initial state process simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    result = jax.random.multivariate_normal(key=key, mean=jnp.array([0, 0]), cov=Q)
    return {"X1": result[0], "X2": result[1]}


def rproc(X_, theta_, key, covars=None, t=None, dt=None):
    """Process simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    X_array = jnp.array([X_["X1"], X_["X2"]])
    result = jax.random.multivariate_normal(key=key, mean=A @ X_array, cov=Q)
    return {"X1": result[0], "X2": result[1]}


def dmeas(Y_, X_, theta_, covars=None, t=None):
    """Measurement model distribution for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    X_array = jnp.array([X_["X1"], X_["X2"]])
    return jax.scipy.stats.multivariate_normal.logpdf(Y_, X_array, R)


def rmeas(X_, theta_, key, covars=None, t=None):
    """Measurement simulator for the linear Gaussian model"""
    A, C, Q, R = get_thetas(theta_)
    X_array = jnp.array([X_["X1"], X_["X2"]])
    return jax.random.multivariate_normal(key=key, mean=C @ X_array, cov=R)


def LG(
    T: int = 4,
    A: jax.Array = jnp.array(
        [[jnp.cos(0.2), -jnp.sin(0.2)], [jnp.sin(0.2), jnp.cos(0.2)]]
    ),
    C: jax.Array = jnp.eye(2),
    Q: jax.Array = jnp.array([[1, 2e-2], [2e-2, 1]]) / 100,
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

    ys_temp = pd.DataFrame(
        0, index=np.arange(1, T + 1, dtype=float), columns=pd.Index(["Y1", "Y2"])
    )

    LG_obj_temp = Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=ys_temp,
        t0=0.0,
        nstep=1,
        dt=None,
        ydim=2,
        theta=theta,
        covars=None,
        statenames=["X1", "X2"],
    )
    _, Y_sims = LG_obj_temp.simulate(key=key)
    Y_sims = Y_sims.rename(columns={"obs_0": "Y1", "obs_1": "Y2"})
    Y_sims = Y_sims[["time", "Y1", "Y2"]]
    Y_sims.set_index("time", inplace=True)
    assert isinstance(Y_sims, pd.DataFrame)

    LG_obj = Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=Y_sims,
        t0=0.0,
        nstep=1,
        dt=None,
        ydim=2,
        theta=theta,
        covars=None,
        statenames=["X1", "X2"],
    )

    return LG_obj

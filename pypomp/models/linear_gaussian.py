"""This module implements a linear Gaussian model for POMP."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pypomp.core.pomp import Pomp
from pypomp.core.par_trans import ParTrans
from pypomp.types import (
    StateDict,
    ParamDict,
    CovarDict,
    TimeFloat,
    StepSizeFloat,
    RNGKey,
    ObservationDict,
    InitialTimeFloat,
)


def _get_thetas(theta):
    A = jnp.array([theta["A11"], theta["A12"], theta["A21"], theta["A22"]]).reshape(
        2, 2
    )
    C = jnp.array([theta["C11"], theta["C12"], theta["C21"], theta["C22"]]).reshape(
        2, 2
    )

    def make_pd(m11_val, m12_val, m22_val):
        m11 = jnp.maximum(m11_val, 1e-12)
        m22 = jnp.maximum(m22_val, 1e-12)
        limit = 0.999 * jnp.sqrt(m11 * m22)
        m_off_clipped = jnp.clip(m12_val, -limit, limit)
        return jnp.array([[m11, m_off_clipped], [m_off_clipped, m22]])

    Q = make_pd(theta["Q11"], theta["Q12"], theta["Q22"])
    R = make_pd(theta["R11"], theta["R12"], theta["R22"])
    return A, C, Q, R


def _transform_thetas(A, C, Q, R):
    return jnp.concatenate(
        [
            A.flatten(),
            C.flatten(),
            jnp.array([Q[0, 0], Q[0, 1], Q[1, 1]]),
            jnp.array([R[0, 0], R[0, 1], R[1, 1]]),
        ]
    )


# TODO: Add custom starting position.
def _rinit(
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t0: InitialTimeFloat,
):
    A, C, Q, R = _get_thetas(theta_)
    result = jax.random.multivariate_normal(key=key, mean=jnp.array([0, 0]), cov=Q)
    return {"X1": result[0], "X2": result[1]}


def _rproc(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t: TimeFloat,
    dt: StepSizeFloat,
):
    A, C, Q, R = _get_thetas(theta_)
    X_array = jnp.array([X_["X1"], X_["X2"]])
    result = jax.random.multivariate_normal(key=key, mean=A @ X_array, cov=Q)
    return {"X1": result[0], "X2": result[1]}


def _dmeas(
    Y_: ObservationDict,
    X_: StateDict,
    theta_: ParamDict,
    covars: CovarDict,
    t: TimeFloat,
):
    A, C, Q, R = _get_thetas(theta_)
    X_array = jnp.array([X_["X1"], X_["X2"]])
    Y_array = jnp.array([Y_["Y1"], Y_["Y2"]])
    return jax.scipy.stats.multivariate_normal.logpdf(Y_array, X_array, R)


def _rmeas(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t: TimeFloat,
):
    A, C, Q, R = _get_thetas(theta_)
    X_array = jnp.array([X_["X1"], X_["X2"]])
    res = jax.random.multivariate_normal(key=key, mean=C @ X_array, cov=R)
    return {"Y1": res[0], "Y2": res[1]}


def _to_est(theta: ParamDict) -> ParamDict:
    new_theta = {**theta}
    for name in "ACQR":
        new_theta[f"{name}11"] = jnp.log(theta[f"{name}11"])
        new_theta[f"{name}22"] = jnp.log(theta[f"{name}22"])
    return new_theta


def _from_est(theta: ParamDict) -> ParamDict:
    new_theta = {**theta}
    for name in "ACQR":
        new_theta[f"{name}11"] = jnp.exp(theta[f"{name}11"])
        new_theta[f"{name}22"] = jnp.exp(theta[f"{name}22"])
    return new_theta


def LG(
    T: int = 4,
    A: np.ndarray = np.array(
        [[jnp.cos(0.2), -jnp.sin(0.2)], [jnp.sin(0.2), jnp.cos(0.2)]]
    ),
    C: np.ndarray = np.eye(2),
    Q: np.ndarray = np.array([[1, 2e-2], [2e-2, 1]]) / 100,
    R: np.ndarray = np.array([[1, 0.1], [0.1, 1]]) / 10,
    key: jax.Array = jax.random.key(1),
) -> Pomp:
    """
    Initialize a Pomp object with the linear Gaussian model.

    Parameters
    ----------
    T : int, optional
        The number of time steps to generate data for. Defaults to 4.
    A : np.ndarray, optional
        The transition matrix.
    C : np.ndarray, optional
        The measurement matrix.
    Q : np.ndarray, optional
        The covariance matrix of the state noise.
    R : np.ndarray, optional
        The covariance matrix of the measurement noise.
    key : jax.Array, optional
        The random key used to generate the data.

    Returns
    -------
    A Pomp object initialized with the linear Gaussian model parameters and the generated data.
    """
    # Validate covariance matrices Q and R
    for name, mat in [("Q", Q), ("R", R)]:
        mat_np = np.asarray(mat)
        if not np.allclose(mat_np, mat_np.T, atol=1e-8, rtol=1e-5):
            raise ValueError(f"Covariance matrix {name} must be symmetric.")
        try:
            np.linalg.cholesky(mat_np)
        except np.linalg.LinAlgError:
            raise ValueError(f"Covariance matrix {name} must be positive-definite.")

    theta_names = [
        "A11",
        "A12",
        "A21",
        "A22",
        "C11",
        "C12",
        "C21",
        "C22",
        "Q11",
        "Q12",
        "Q22",
        "R11",
        "R12",
        "R22",
    ]
    theta = dict(zip(theta_names, _transform_thetas(A, C, Q, R).tolist()))

    ys_temp = pd.DataFrame(
        0, index=np.arange(1, T + 1, dtype=float), columns=pd.Index(["Y1", "Y2"])
    )

    from pypomp.core.parameters import PompParameters

    LG_obj_temp = Pomp(
        rinit=_rinit,
        rproc=_rproc,
        dmeas=_dmeas,
        rmeas=_rmeas,
        ys=ys_temp,
        t0=0.0,
        nstep=1,
        dt=None,
        theta=PompParameters(theta),
        covars=None,
        statenames=["X1", "X2"],
        par_trans=ParTrans(to_est=_to_est, from_est=_from_est),
    )
    LG_obj = LG_obj_temp.simulate(key=key, nsim=1, as_pomp=True)
    assert isinstance(LG_obj, Pomp)

    return LG_obj

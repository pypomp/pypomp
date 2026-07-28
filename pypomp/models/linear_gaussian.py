"""This module implements a linear Gaussian model for POMP.

The model is

.. math::

    X_0    &\\sim N(X_0^{init}, Q) \\\\
    X_t    &\\sim N(A X_{t-1}, Q) \\\\
    Y_t    &\\sim N(C X_t, R)

for a state of dimension ``dx`` and an observation of dimension ``dy``. Both
dimensions are inferred from the shapes of the arrays passed to :func:`LG`, so
1-D, 2-D, or higher-dimensional variants are all available from the same
constructor.

Parameter naming
----------------
``A`` and ``C`` contribute one parameter per entry, named ``A{row}{col}`` and
``C{row}{col}`` with 1-based indices. ``Q`` and ``R`` are stored as the
lower-triangular Cholesky factors ``L`` with ``Q = L @ L.T``, which is positive
definite for any values of ``L``; their parameters are the lower-triangular
entries ``Q{row}{col}`` (``row >= col``). The initial state mean contributes
``X0_{i}``. Because the row and column indices are concatenated without a
separator, dimensions above :data:`MAX_DIM` are rejected as ambiguous.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pypomp.core.par_trans import ParTrans
from pypomp.core.pomp import Pomp
from pypomp.types import (
    CovarDict,
    InitialTimeFloat,
    ObservationDict,
    ParamDict,
    RNGKey,
    StateDict,
    StepSizeFloat,
    TimeFloat,
)

#: Largest state or observation dimension supported. Above this, the ``A11``
#: style parameter names become ambiguous (``A11`` could mean ``A[1, 1]`` or
#: ``A[1, 11]``).
MAX_DIM = 9


# --- Parameter naming -------------------------------------------------------


def _matrix_names(name: str, nrow: int, ncol: int) -> list[str]:
    """Names for every entry of a ``nrow`` by ``ncol`` matrix, row-major."""
    return [f"{name}{i}{j}" for i in range(1, nrow + 1) for j in range(1, ncol + 1)]


def _tril_names(name: str, d: int) -> list[str]:
    """Names for the lower triangle of a ``d`` by ``d`` matrix, row-major."""
    return [f"{name}{i}{j}" for i in range(1, d + 1) for j in range(1, i + 1)]


def _x0_names(dx: int) -> list[str]:
    """Names for the initial state mean."""
    return [f"X0_{i}" for i in range(1, dx + 1)]


def _statenames(dx: int) -> list[str]:
    return [f"X{i}" for i in range(1, dx + 1)]


def _obsnames(dy: int) -> list[str]:
    return [f"Y{j}" for j in range(1, dy + 1)]


# --- Unpacking theta --------------------------------------------------------


def _dims(theta: ParamDict) -> tuple[int, int]:
    """Recover ``(dx, dy)`` from the parameter names.

    The names are static at trace time, so this is safe to call from inside
    jitted model components.
    """
    dx = sum(1 for k in theta if k.startswith("X0_"))
    if dx == 0:
        raise ValueError("theta contains no 'X0_i' entries; cannot infer state dim.")
    n_c = sum(1 for k in theta if k.startswith("C"))
    return dx, n_c // dx


def _matrix_from_flat(name: str, nrow: int, ncol: int, theta: ParamDict) -> jax.Array:
    return jnp.array(
        [
            [theta[f"{name}{i}{j}"] for j in range(1, ncol + 1)]
            for i in range(1, nrow + 1)
        ]
    )


def _tril_from_flat(name: str, d: int, theta: ParamDict) -> jax.Array:
    """Assemble a lower-triangular matrix from the ``name{i}{j}`` entries."""
    L = jnp.zeros((d, d))
    for i in range(1, d + 1):
        for j in range(1, i + 1):
            L = L.at[i - 1, j - 1].set(theta[f"{name}{i}{j}"])
    return L


def _unpack(
    theta: ParamDict,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``(A, C, L_Q, L_R, X0)``, with ``L_Q``/``L_R`` Cholesky factors.

    The factors rather than the covariances are returned because sampling uses
    ``mean + L @ z``, which avoids re-factorizing a covariance on every call.
    """
    dx, dy = _dims(theta)
    A = _matrix_from_flat("A", dx, dx, theta)
    C = _matrix_from_flat("C", dy, dx, theta)
    L_Q = _tril_from_flat("Q", dx, theta)
    L_R = _tril_from_flat("R", dy, theta)
    X0 = jnp.array([theta[n] for n in _x0_names(dx)])
    return A, C, L_Q, L_R, X0


def _get_thetas(
    theta: ParamDict,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``(A, C, Q, R, X0)`` with ``Q`` and ``R`` as full covariances."""
    A, C, L_Q, L_R, X0 = _unpack(theta)
    return A, C, L_Q @ L_Q.T, L_R @ L_R.T, X0


# --- Model components -------------------------------------------------------


def _rinit(
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t0: InitialTimeFloat,
):
    A, C, L_Q, L_R, X0 = _unpack(theta_)
    dx = X0.shape[0]
    result = X0 + L_Q @ jax.random.normal(key, (dx,))
    return {name: result[i] for i, name in enumerate(_statenames(dx))}


def _rproc(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t: TimeFloat,
    dt: StepSizeFloat,
):
    A, C, L_Q, L_R, X0 = _unpack(theta_)
    dx = X0.shape[0]
    X_array = jnp.array([X_[name] for name in _statenames(dx)])
    result = A @ X_array + L_Q @ jax.random.normal(key, (dx,))
    return {name: result[i] for i, name in enumerate(_statenames(dx))}


def _dmeas(
    Y_: ObservationDict,
    X_: StateDict,
    theta_: ParamDict,
    covars: CovarDict,
    t: TimeFloat,
):
    A, C, L_Q, L_R, X0 = _unpack(theta_)
    dx, dy = X0.shape[0], L_R.shape[0]
    X_array = jnp.array([X_[name] for name in _statenames(dx)])
    Y_array = jnp.array([Y_[name] for name in _obsnames(dy)])
    return jax.scipy.stats.multivariate_normal.logpdf(Y_array, C @ X_array, L_R @ L_R.T)


def _rmeas(
    X_: StateDict,
    theta_: ParamDict,
    key: RNGKey,
    covars: CovarDict,
    t: TimeFloat,
):
    A, C, L_Q, L_R, X0 = _unpack(theta_)
    dx, dy = X0.shape[0], L_R.shape[0]
    X_array = jnp.array([X_[name] for name in _statenames(dx)])
    res = C @ X_array + L_R @ jax.random.normal(key, (dy,))
    return {name: res[j] for j, name in enumerate(_obsnames(dy))}


# --- Parameter transformations ----------------------------------------------


def _is_cov_diagonal(name: str) -> bool:
    """True for the diagonal Cholesky entries of Q and R, e.g. ``Q11``, ``R22``."""
    return len(name) == 3 and name[0] in ("Q", "R") and name[1] == name[2]


def _to_est(theta: ParamDict) -> ParamDict:
    new_theta = {**theta}
    for name in theta:
        if _is_cov_diagonal(name):
            new_theta[name] = jnp.log(theta[name])
    return new_theta


def _from_est(theta: ParamDict) -> ParamDict:
    new_theta = {**theta}
    for name in theta:
        if _is_cov_diagonal(name):
            new_theta[name] = jnp.exp(theta[name])
    return new_theta


# --- Default matrices -------------------------------------------------------


def _default_A(dx: int) -> np.ndarray:
    """Block-diagonal rotation by 0.2 radians.

    At ``dx = 2`` this is the plane rotation matrix used historically; an odd
    trailing dimension gets ``cos(0.2)``.
    """
    A = np.zeros((dx, dx))
    c, s = float(np.cos(0.2)), float(np.sin(0.2))
    i = 0
    while i + 1 < dx:
        A[i : i + 2, i : i + 2] = np.array([[c, -s], [s, c]])
        i += 2
    if i < dx:
        A[i, i] = c
    return A


def _default_C(dy: int, dx: int) -> np.ndarray:
    """Observe the leading ``dy`` state coordinates."""
    return np.eye(dy, dx)


def _default_cov(d: int, diag: float, offdiag: float) -> np.ndarray:
    """``diag`` on the diagonal, ``offdiag / (d - 1)`` elsewhere.

    Dividing the off-diagonal by ``d - 1`` keeps the matrix strictly diagonally
    dominant, hence positive definite, at every dimension, while reproducing the
    historical 2-D defaults exactly.
    """
    M = np.eye(d) * diag
    if d > 1:
        M = M + (np.ones((d, d)) - np.eye(d)) * (offdiag / (d - 1))
    return M


def _default_Q(dx: int) -> np.ndarray:
    return _default_cov(dx, 1.0, 0.02) / 100


def _default_R(dy: int) -> np.ndarray:
    return _default_cov(dy, 1.0, 0.1) / 10


# --- Dimension inference ----------------------------------------------------


def _square_dim(name: str, mat: np.ndarray) -> int:
    arr = np.asarray(mat)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a square 2-D array, but has shape {arr.shape}."
        )
    return int(arr.shape[0])


def _resolve(claims: dict[str, int], kind: str) -> int | None:
    """Collapse per-argument dimension claims into one, or raise on conflict."""
    if not claims:
        return None
    if len(set(claims.values())) > 1:
        detail = ", ".join(f"{k} implies {v}" for k, v in sorted(claims.items()))
        raise ValueError(f"Inconsistent {kind} dimension: {detail}.")
    return next(iter(claims.values()))


def _infer_dims(
    A: np.ndarray | None,
    C: np.ndarray | None,
    Q: np.ndarray | None,
    R: np.ndarray | None,
    X0: np.ndarray | None,
) -> tuple[int, int]:
    dx_claims: dict[str, int] = {}
    dy_claims: dict[str, int] = {}

    if A is not None:
        dx_claims["A"] = _square_dim("A", A)
    if Q is not None:
        dx_claims["Q"] = _square_dim("Q", Q)
    if X0 is not None:
        x0_arr = np.asarray(X0)
        if x0_arr.ndim != 1:
            raise ValueError(f"X0 must be a 1-D array, but has shape {x0_arr.shape}.")
        dx_claims["X0"] = int(x0_arr.shape[0])
    if C is not None:
        c_arr = np.asarray(C)
        if c_arr.ndim != 2:
            raise ValueError(
                f"C must be a 2-D array of shape (dy, dx), but has shape {c_arr.shape}."
            )
        dx_claims["the number of columns of C"] = int(c_arr.shape[1])
        dy_claims["the number of rows of C"] = int(c_arr.shape[0])
    if R is not None:
        dy_claims["R"] = _square_dim("R", R)

    dx = _resolve(dx_claims, "state")
    dy = _resolve(dy_claims, "observation")

    # Nothing constrains the state dimension: follow the observation dimension
    # if one was given, otherwise fall back to the historical 2-D model.
    if dx is None:
        dx = dy if dy is not None else 2
    if dy is None:
        dy = dx

    for kind, d in (("state", dx), ("observation", dy)):
        if d < 1:
            raise ValueError(f"The {kind} dimension must be at least 1, but is {d}.")
        if d > MAX_DIM:
            raise ValueError(
                f"LG supports dimensions up to {MAX_DIM}, but the {kind} dimension is "
                f"{d}. Above {MAX_DIM} the 'A11' style parameter names are ambiguous."
            )
    return dx, dy


def LG(
    T: int = 4,
    A: np.ndarray | None = None,
    C: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    X0: np.ndarray | None = None,
    key: jax.Array | None = None,
) -> Pomp:
    """
    Initialize a Pomp object with the linear Gaussian model.

    The state dimension ``dx`` and observation dimension ``dy`` are inferred
    from the shapes of whichever arrays are supplied; any argument left as
    ``None`` is generated at the inferred size. Supplying nothing reproduces the
    two-dimensional model. Both dimensions are capped at :data:`MAX_DIM`.

    Parameters
    ----------
    T : int, optional
        The number of time steps to generate data for. Defaults to 4.
    A : np.ndarray, optional
        The ``(dx, dx)`` transition matrix. Defaults to a block-diagonal
        rotation by 0.2 radians.
    C : np.ndarray, optional
        The ``(dy, dx)`` measurement matrix. Defaults to ``np.eye(dy, dx)``,
        which observes the leading ``dy`` state coordinates.
    Q : np.ndarray, optional
        The ``(dx, dx)`` covariance matrix of the state noise. Must be symmetric
        positive-definite.
    R : np.ndarray, optional
        The ``(dy, dy)`` covariance matrix of the measurement noise. Must be
        symmetric positive-definite.
    X0 : np.ndarray, optional
        The mean of the initial state distribution, of length ``dx``. Defaults
        to zeros. The initial state is drawn from ``N(X0, Q)``, so with the
        default ``Q`` the process starts tightly concentrated around ``X0``.
        These values enter ``theta`` as ``X0_1 ... X0_{dx}`` and so are
        estimable.
    key : jax.Array, optional
        The random key used to generate the data.

    Returns
    -------
    A Pomp object initialized with the linear Gaussian model parameters and the
    generated data.

    Examples
    --------
    >>> LG()                                  # 2-D state, 2-D observation
    >>> LG(A=np.array([[0.9]]))               # 1-D
    >>> LG(A=np.eye(3), C=np.eye(2, 3))       # 3-D state, 2-D observation
    >>> LG(X0=np.array([5.0, -5.0]))          # start away from the origin
    """
    if key is None:
        key = jax.random.key(1)

    dx, dy = _infer_dims(A, C, Q, R, X0)

    A_ = _default_A(dx) if A is None else np.asarray(A, dtype=float)
    C_ = _default_C(dy, dx) if C is None else np.asarray(C, dtype=float)
    Q_ = _default_Q(dx) if Q is None else np.asarray(Q, dtype=float)
    R_ = _default_R(dy) if R is None else np.asarray(R, dtype=float)
    X0_ = np.zeros(dx) if X0 is None else np.asarray(X0, dtype=float)

    # Validate covariance matrices Q and R
    for name, mat in [("Q", Q_), ("R", R_)]:
        if not np.allclose(mat, mat.T, atol=1e-8, rtol=1e-5):
            raise ValueError(f"Covariance matrix {name} must be symmetric.")
        try:
            np.linalg.cholesky(mat)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                f"Covariance matrix {name} must be positive-definite."
            ) from e

    L_Q = np.linalg.cholesky(Q_)
    L_R = np.linalg.cholesky(R_)

    theta: dict[str, float] = {}
    for name, mat, nrow, ncol in [("A", A_, dx, dx), ("C", C_, dy, dx)]:
        for par in _matrix_names(name, nrow, ncol):
            i, j = int(par[-2]), int(par[-1])
            theta[par] = float(mat[i - 1, j - 1])
    for name, fac, d in [("Q", L_Q, dx), ("R", L_R, dy)]:
        for par in _tril_names(name, d):
            i, j = int(par[-2]), int(par[-1])
            theta[par] = float(fac[i - 1, j - 1])
    for i, par in enumerate(_x0_names(dx)):
        theta[par] = float(X0_[i])

    ys_temp = pd.DataFrame(
        0, index=np.arange(1, T + 1, dtype=float), columns=pd.Index(_obsnames(dy))
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
        statenames=_statenames(dx),
        par_trans=ParTrans(to_est=_to_est, from_est=_from_est),
    )
    LG_obj = LG_obj_temp.simulate(key=key, nsim=1, as_pomp=True)
    assert isinstance(LG_obj, Pomp)

    return LG_obj

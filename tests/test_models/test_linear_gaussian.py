from typing import cast

import jax
import numpy as np
import pytest

import pypomp as pp
from pypomp.models.linear_gaussian import (
    MAX_DIM,
    _default_A,
    _default_Q,
    _default_R,
    _from_est,
    _get_thetas,
    _to_est,
)
from pypomp.types import ParamDict


def _theta_of(model: pp.Pomp) -> ParamDict:
    """The model's single parameter replicate, as a plain dict."""
    return cast(ParamDict, dict(model.theta.params(as_list=True)[0]))


def _pfilter_loglik(model: pp.Pomp, J: int = 200, seed: int = 0) -> float:
    model.pfilter(J=J, key=jax.random.key(seed))
    return float(np.asarray(model.results_history[-1].logLiks).ravel()[0])


# --- Parameter transformations ----------------------------------------------


def test_lg_par_trans_roundtrip():
    # Q and R are stored as Cholesky factors, so only their diagonals are
    # log-transformed; everything else must pass through untouched.
    theta_orig = {
        "A11": 0.9,
        "A12": -0.1,
        "A21": 0.1,
        "A22": 0.8,
        "C11": 1.0,
        "C12": 0.0,
        "C21": 0.0,
        "C22": 1.0,
        "Q11": 0.1,
        "Q21": 0.002,
        "Q22": 0.09,
        "R11": 0.3,
        "R21": 0.03,
        "R22": 0.2,
        "X0_1": 1.5,
        "X0_2": -2.5,
    }

    theta_est = _to_est(cast(ParamDict, theta_orig))
    theta_nat = _from_est(theta_est)

    for k, v in theta_orig.items():
        assert np.allclose(theta_nat[k], v, rtol=1e-6, atol=1e-6)

    # Only the covariance diagonals are transformed.
    for k in ("A11", "C12", "Q21", "R21", "X0_1", "X0_2"):
        assert np.allclose(theta_est[k], theta_orig[k])
    for k in ("Q11", "Q22", "R11", "R22"):
        assert np.allclose(theta_est[k], np.log(theta_orig[k]))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_lg_par_trans_enforces_psd(dim):
    # Arbitrary values in the estimation space -- including negative and large
    # ones -- must still map to symmetric positive-definite Q and R, at any
    # dimension. This is what the Cholesky parameterization buys us.
    rng = np.random.default_rng(0)
    theta_est: dict[str, float] = {}
    for i in range(1, dim + 1):
        for j in range(1, dim + 1):
            theta_est[f"A{i}{j}"] = float(rng.normal(scale=3.0))
            theta_est[f"C{i}{j}"] = float(rng.normal(scale=3.0))
        for j in range(1, i + 1):
            # Kept within a range whose reconstructed covariance is still
            # resolvable in float32; the guarantee itself is scale-free, but
            # eigenvalues of a matrix with condition number >1e7 are not.
            theta_est[f"Q{i}{j}"] = float(rng.normal(scale=1.5))
            theta_est[f"R{i}{j}"] = float(rng.normal(scale=1.5))
        theta_est[f"X0_{i}"] = float(rng.normal())

    theta_nat = _from_est(cast(ParamDict, theta_est))

    # The actual invariant: the transform puts strictly positive values on the
    # Cholesky diagonal, which is what makes Q = L @ L.T positive definite.
    for name in ("Q", "R"):
        for i in range(1, dim + 1):
            assert float(theta_nat[f"{name}{i}{i}"]) > 0

    _, _, Q, R, _ = _get_thetas(theta_nat)

    for name, mat in [("Q", np.array(Q)), ("R", np.array(R))]:
        assert np.allclose(mat, mat.T, rtol=1e-6, atol=1e-12), f"{name} not symmetric"
        eigenvalues = np.linalg.eigvalsh(mat)
        assert np.all(eigenvalues > 0), (
            f"{name} eigenvalues were not positive: {eigenvalues}"
        )


def test_lg_get_thetas_recovers_covariances():
    # LG stores the Cholesky factor of what the user passes in; _get_thetas
    # must reconstruct the original covariance.
    Q_in = np.array([[0.04, 0.01], [0.01, 0.09]])
    R_in = np.array([[0.5, -0.2], [-0.2, 0.3]])
    model = pp.models.lg(Q=Q_in, R=R_in)

    A, C, Q, R, X0 = _get_thetas(_theta_of(model))

    assert np.allclose(np.array(Q), Q_in)
    assert np.allclose(np.array(R), R_in)
    assert np.allclose(np.array(A), _default_A(2))
    assert np.allclose(np.array(C), np.eye(2))
    assert np.allclose(np.array(X0), np.zeros(2))


# --- Covariance validation --------------------------------------------------


def test_lg_covariance_validation():
    # Symmetric check
    asymmetric_cov = np.array([[1.0, 0.5], [0.2, 1.0]])
    valid_cov = np.array([[1.0, 0.2], [0.2, 1.0]])

    with pytest.raises(ValueError, match="Covariance matrix Q must be symmetric"):
        pp.models.lg(Q=asymmetric_cov)

    with pytest.raises(ValueError, match="Covariance matrix R must be symmetric"):
        pp.models.lg(R=asymmetric_cov)

    # Positive-definite check
    non_pd_cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # determinant is 1 - 4 = -3 < 0
    with pytest.raises(
        ValueError, match="Covariance matrix Q must be positive-definite"
    ):
        pp.models.lg(Q=non_pd_cov)

    with pytest.raises(
        ValueError, match="Covariance matrix R must be positive-definite"
    ):
        pp.models.lg(R=non_pd_cov)

    # Valid matrices should pass without errors
    LG_obj = pp.models.lg(Q=valid_cov, R=valid_cov)
    assert isinstance(LG_obj, pp.Pomp)


# --- Dimensions -------------------------------------------------------------


def test_lg_defaults_are_two_dimensional():
    model = pp.models.lg()
    assert model.statenames == ["X1", "X2"]
    assert list(model.ys.columns) == ["Y1", "Y2"]
    assert list(model.canonical_param_names) == [
        "A11",
        "A12",
        "A21",
        "A22",
        "C11",
        "C12",
        "C21",
        "C22",
        "Q11",
        "Q21",
        "Q22",
        "R11",
        "R21",
        "R22",
        "X0_1",
        "X0_2",
    ]


def test_lg_default_matrices_match_historical_two_dimensional_values():
    # The generated defaults must reproduce the original hard-coded 2-D model.
    assert np.allclose(
        _default_A(2),
        np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]]),
    )
    assert np.allclose(_default_Q(2), np.array([[1, 2e-2], [2e-2, 1]]) / 100)
    assert np.allclose(_default_R(2), np.array([[1, 0.1], [0.1, 1]]) / 10)


@pytest.mark.parametrize("d", [1, 3, 5])
def test_lg_generated_defaults_are_positive_definite(d):
    for mat in (_default_Q(d), _default_R(d)):
        np.linalg.cholesky(mat)  # raises if not positive-definite


def test_lg_one_dimensional():
    model = pp.models.lg(A=np.array([[0.9]]))

    assert model.statenames == ["X1"]
    assert list(model.ys.columns) == ["Y1"]
    assert list(model.canonical_param_names) == ["A11", "C11", "Q11", "R11", "X0_1"]
    assert np.isfinite(_pfilter_loglik(model))


def test_lg_state_and_observation_dims_may_differ():
    # 3-D state observed through a 2-D measurement.
    model = pp.models.lg(A=np.eye(3) * 0.9, C=np.eye(2, 3))

    assert model.statenames == ["X1", "X2", "X3"]
    assert list(model.ys.columns) == ["Y1", "Y2"]
    # A: 9, C: 6, Q: 6, R: 3, X0: 3
    assert len(model.canonical_param_names) == 27
    assert np.isfinite(_pfilter_loglik(model))

    X_sims, Y_sims = model.simulate(nsim=2, key=jax.random.key(1))
    assert {"X1", "X2", "X3"}.issubset(X_sims.columns)
    assert {"Y1", "Y2"}.issubset(Y_sims.columns)


@pytest.mark.parametrize(
    "build",
    [
        lambda: pp.models.lg(A=np.eye(3) * 0.5),
        lambda: pp.models.lg(Q=np.eye(3) * 0.01),
        lambda: pp.models.lg(X0=np.zeros(3)),
        lambda: pp.models.lg(C=np.eye(3)),
        lambda: pp.models.lg(R=np.eye(3) * 0.1),
    ],
    ids=["A", "Q", "X0", "C", "R"],
)
def test_lg_dimension_inferred_from_each_argument(build):
    # Any one argument is enough to pin down the dimension of the whole model.
    model = build()
    assert model.statenames == ["X1", "X2", "X3"]
    assert list(model.ys.columns) == ["Y1", "Y2", "Y3"]


def test_lg_inconsistent_dimensions_raise():
    with pytest.raises(ValueError, match="Inconsistent state dimension"):
        pp.models.lg(A=np.eye(3) * 0.5, Q=np.eye(2) * 0.01)

    with pytest.raises(ValueError, match="Inconsistent state dimension"):
        pp.models.lg(A=np.eye(3) * 0.5, X0=np.zeros(2))

    with pytest.raises(ValueError, match="Inconsistent observation dimension"):
        pp.models.lg(C=np.eye(2, 3), R=np.eye(3) * 0.1)


def test_lg_shape_validation():
    with pytest.raises(ValueError, match="A must be a square 2-D array"):
        pp.models.lg(A=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="X0 must be a 1-D array"):
        pp.models.lg(X0=np.zeros((2, 1)))


def test_lg_dimension_cap():
    d = MAX_DIM + 1
    with pytest.raises(ValueError, match=f"LG supports dimensions up to {MAX_DIM}"):
        pp.models.lg(A=np.eye(d) * 0.5)


# --- Starting position ------------------------------------------------------


def test_lg_starting_position_is_used():
    X0 = np.array([50.0, -50.0])
    model = pp.models.lg(X0=X0, T=1)

    theta = _theta_of(model)
    assert theta["X0_1"] == 50.0
    assert theta["X0_2"] == -50.0

    X_sims, _ = model.simulate(nsim=300, key=jax.random.key(3))
    means = X_sims.reset_index().groupby("time")[["X1", "X2"]].mean()

    # At t0 the state is drawn from N(X0, Q); with the default Q the spread is
    # ~0.1 per coordinate, so 300 draws pin the mean down tightly.
    assert np.allclose(means.loc[0.0].to_numpy(), X0, atol=0.1)

    # After one step the mean is A @ X0.
    expected = _default_A(2) @ X0
    assert np.allclose(means.loc[1.0].to_numpy(), expected, atol=0.1)


def test_lg_starting_position_defaults_to_origin():
    model = pp.models.lg()
    theta = _theta_of(model)
    assert theta["X0_1"] == 0.0
    assert theta["X0_2"] == 0.0


# --- Correctness against the Kalman filter ----------------------------------


def _kalman_loglik(
    ys: np.ndarray, a: float, c: float, q: float, r: float, x0: float, p0: float
) -> float:
    """Exact log-likelihood of the 1-D linear Gaussian model.

    Uses the same convention as LG: the initial state is drawn from N(x0, p0)
    at t0, then propagated once before the first observation.
    """
    x, p, loglik = x0, p0, 0.0
    for y in ys:
        x_pred = a * x
        p_pred = a * a * p + q
        v = y - c * x_pred
        s = c * c * p_pred + r
        loglik += -0.5 * (np.log(2.0 * np.pi * s) + v * v / s)
        k = c * p_pred / s
        x = x_pred + k * v
        p = (1.0 - k * c) * p_pred
    return float(loglik)


def test_lg_pfilter_matches_kalman_filter_in_one_dimension():
    a, c, q, r = 0.9, 1.0, 0.25, 0.5
    model = pp.models.lg(
        T=20,
        A=np.array([[a]]),
        C=np.array([[c]]),
        Q=np.array([[q]]),
        R=np.array([[r]]),
        X0=np.array([1.0]),
        key=jax.random.key(7),
    )

    exact = _kalman_loglik(model.ys["Y1"].to_numpy(), a=a, c=c, q=q, r=r, x0=1.0, p0=q)

    # Average several particle filter replicates to tame Monte Carlo error.
    logliks = []
    for seed in range(5):
        logliks.append(_pfilter_loglik(model, J=5000, seed=seed))
    est = float(np.mean(logliks))

    assert abs(est - exact) < 1.0, f"pfilter={est}, kalman={exact}"

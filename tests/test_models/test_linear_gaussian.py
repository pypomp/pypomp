import jax.numpy as jnp
import numpy as np
import pytest
import pypomp as pp
from pypomp.types import ParamDict
from pypomp.models.linear_gaussian import _to_est, _from_est, _get_thetas
from typing import cast


def test_lg_par_trans_roundtrip():
    theta_orig = {
        "A11": 0.9,
        "A12": -0.1,
        "A21": 0.1,
        "A22": 0.8,
        "C11": 1.0,
        "C12": 0.0,
        "C21": 0.0,
        "C22": 1.0,
        "Q11": 0.01,
        "Q12": 0.0002,
        "Q22": 0.01,
        "R11": 0.1,
        "R12": 0.01,
        "R22": 0.1,
    }

    theta_est = _to_est(cast(ParamDict, theta_orig))

    theta_nat = _from_est(theta_est)

    # Verify that round trip recovers the original parameters exactly
    for k, v in theta_orig.items():
        assert np.allclose(theta_nat[k], v, rtol=1e-6, atol=1e-6)


def test_lg_par_trans_enforces_psd():
    # Check that any arbitrary values in the estimation space
    # (even negative or very large/small values) map to valid, symmetric,
    # positive-definite matrices Q and R in the natural space.

    theta_est = {
        "A11": 0.5,
        "A12": -1.0,
        "A21": 2.0,
        "A22": 0.0,
        "C11": -0.5,
        "C12": 1.5,
        "C21": -0.5,
        "C22": 0.8,
        "Q11": -1.5,
        "Q12": 3.0,
        "Q22": -0.5,
        "R11": 2.0,
        "R12": -5.0,
        "R22": -3.5,
    }

    theta_nat = _from_est(cast(ParamDict, theta_est))

    A, C, Q, R = _get_thetas(theta_nat)

    for name, mat in [("Q", Q), ("R", R)]:
        mat = np.array(mat)

        # 1. Symmetry check: mat should be symmetric
        assert np.allclose(mat[0, 1], mat[1, 0], rtol=1e-6, atol=1e-6)

        # 2. Positive-definite check: eigenvalues should be strictly positive
        eigenvalues = np.linalg.eigvals(mat)
        assert np.all(eigenvalues > 0), (
            f"{name} matrix eigenvalues were not positive: {eigenvalues}"
        )

        # 3. Check specific diagonal components are positive
        assert mat[0, 0] > 0
        assert mat[1, 1] > 0


def test_lg_covariance_validation():
    # Symmetric check
    asymmetric_cov = jnp.array([[1.0, 0.5], [0.2, 1.0]])
    valid_cov = jnp.array([[1.0, 0.2], [0.2, 1.0]])

    with pytest.raises(ValueError, match="Covariance matrix Q must be symmetric"):
        pp.models.LG(Q=asymmetric_cov)

    with pytest.raises(ValueError, match="Covariance matrix R must be symmetric"):
        pp.models.LG(R=asymmetric_cov)

    # Positive-definite check
    non_pd_cov = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # determinant is 1 - 4 = -3 < 0
    with pytest.raises(
        ValueError, match="Covariance matrix Q must be positive-definite"
    ):
        pp.models.LG(Q=non_pd_cov)

    with pytest.raises(
        ValueError, match="Covariance matrix R must be positive-definite"
    ):
        pp.models.LG(R=non_pd_cov)

    # Valid matrices should pass without errors
    LG_obj = pp.models.LG(Q=valid_cov, R=valid_cov)
    assert isinstance(LG_obj, pp.Pomp)

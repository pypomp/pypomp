import jax
import numpy as np
import pytest

import pypomp as pp
from pypomp.models.sir import (
    DEFAULT_THETA,
    STATENAMES,
    from_est,
    get_process_weight_index,
    periodic_bspline_basis_eval,
    to_est,
)


@pytest.fixture(scope="module")
def sir_default():
    return pp.models.sir()


def test_sir_construct_default(sir_default):
    sir = sir_default
    assert isinstance(sir, pp.Pomp)
    # 4 years of weekly observations under default `times`.
    assert len(sir.ys) == int(4 * 52)
    assert list(sir.ys.columns) == ["reports"]
    assert sir.statenames == STATENAMES
    for name in DEFAULT_THETA:
        assert name in sir.canonical_param_names


def test_sir_pfilter(sir_default):
    sir = sir_default
    sir.pfilter(J=20, key=jax.random.key(1))
    logLik = float(pp.maths.logmeanexp(sir.results_history[-1].logLiks))
    assert np.isfinite(logLik)


def test_sir_construct_explicit_args():
    # Passing R_0 and times explicitly exercises the non-default branches
    # in the constructor.
    times = np.linspace(0.05, 0.4, 8)
    sir = pp.models.sir(R_0=0.5, times=times, t0=0.0)
    assert isinstance(sir, pp.Pomp)
    assert len(sir.ys) == len(times)


def test_par_trans_roundtrip():
    est = to_est(DEFAULT_THETA)
    nat = from_est(est)
    for name, value in DEFAULT_THETA.items():
        assert float(nat[name]) == pytest.approx(value, rel=1e-5, abs=1e-6)


def test_periodic_bspline_basis_deriv_zero():
    y = periodic_bspline_basis_eval(0.5, period=1.0, degree=3, nbasis=3, deriv=0)
    # B-spline basis values sum to 1 at any x for the standard partition of unity.
    assert float(np.sum(np.asarray(y))) == pytest.approx(1.0, abs=1e-5)


def test_periodic_bspline_basis_first_derivative():
    # deriv > 0 hits the recursive derivative branch in `_bspline_eval`.
    y = periodic_bspline_basis_eval(0.5, period=1.0, degree=3, nbasis=3, deriv=1)
    # First derivatives of a partition of unity sum to 0.
    assert float(np.sum(np.asarray(y))) == pytest.approx(0.0, abs=1e-5)


def test_periodic_bspline_basis_deriv_above_degree():
    # deriv > degree should short-circuit to zero.
    y = periodic_bspline_basis_eval(0.5, period=1.0, degree=3, nbasis=3, deriv=4)
    assert np.allclose(np.asarray(y), 0.0)


def test_get_process_weight_index():
    assert get_process_weight_index() == STATENAMES.index("logw")

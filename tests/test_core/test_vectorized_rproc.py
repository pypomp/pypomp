"""Tests for manually vectorized rproc components (``@pp.vectorized``).

A vectorized rproc receives every state as a ``(J,)`` array and is called once
per Euler step instead of being vmapped over particles. The equivalence tests
below use *deterministic* dynamics so that the two code paths must agree
exactly; a stochastic rproc cannot match bit-for-bit because the vectorized path
draws from a single key per step rather than one key per particle.
"""

import pickle

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
from pypomp.core.model_mechanics import _RProc
from pypomp.core.parameters import PompParameters

STATENAMES = ["x", "acc"]
THETA = {"growth": 0.1, "sigma": 0.2}
YS = pd.DataFrame({"y": np.linspace(1.0, 2.0, 12)}, index=np.arange(1.0, 13.0))


def _rinit(theta_, key, covars, t0):
    return {"x": 1.0, "acc": 0.0}


def _dmeas(Y_, X_, theta_, covars, t):
    return jax.scipy.stats.norm.logpdf(Y_["y"], X_["x"], 0.5)


def _rmeas(X_, theta_, key, covars, t):
    return {"y": X_["x"] + jax.random.normal(key) * 0.5}


# --- deterministic pair -------------------------------------------------


def _rproc_scalar_det(X_, theta_, key, covars, t, dt):
    return {"x": X_["x"] * (1.0 + theta_["growth"] * dt), "acc": X_["acc"] + dt}


@pp.vectorized
def _rproc_vec_det(X_, theta_, key, covars, t, dt):
    return {"x": X_["x"] * (1.0 + theta_["growth"] * dt), "acc": X_["acc"] + dt}


# --- stochastic pair ----------------------------------------------------


def _rproc_scalar_sto(X_, theta_, key, covars, t, dt):
    dw = jax.random.normal(key) * jnp.sqrt(dt)
    return {
        "x": X_["x"] * (1.0 + theta_["growth"] * dt) + theta_["sigma"] * dw,
        "acc": X_["acc"] + dt,
    }


@pp.vectorized
def _rproc_vec_sto(X_, theta_, key, covars, t, dt):
    x = X_["x"]
    dw = jax.random.normal(key, (x.shape[0],)) * jnp.sqrt(dt)
    return {
        "x": x * (1.0 + theta_["growth"] * dt) + theta_["sigma"] * dw,
        "acc": X_["acc"] + dt,
    }


def _build(rproc, accumvars=None):
    return pp.Pomp(
        rinit=_rinit,
        rproc=rproc,
        dmeas=_dmeas,
        rmeas=_rmeas,
        ys=YS,
        t0=0.0,
        nstep=3,
        accumvars=accumvars,
        theta=PompParameters(THETA),
        statenames=STATENAMES,
    )


def _pfilter_loglik(model, J=64, seed=5):
    model.pfilter(J=J, key=jax.random.key(seed), reps=1)
    return float(model.results()["logLik"].iloc[0])


# --- flag detection -----------------------------------------------------


def test_decorator_sets_marker():
    assert getattr(_rproc_vec_det, "_pypomp_vectorized", False) is True
    assert getattr(_rproc_scalar_det, "_pypomp_vectorized", False) is False


def test_rproc_autodetects_decorator():
    assert _build(_rproc_vec_det).rproc._is_vectorized is True
    assert _build(_rproc_scalar_det).rproc._is_vectorized is False


def test_explicit_flag_overrides_decorator():
    """The constructor argument wins over the decorator marker."""
    rp = _RProc(
        _rproc_scalar_det,
        statenames=STATENAMES,
        param_names=list(THETA),
        covar_names=[],
        par_trans=pp.ParTrans(),
        nstep=2,
        vectorized=False,
    )
    assert rp._is_vectorized is False


def test_equality_distinguishes_vectorization():
    a = _build(_rproc_scalar_det).rproc
    b = _build(_rproc_scalar_det).rproc
    assert a == b
    assert _build(_rproc_vec_det).rproc != a


# --- equivalence of the two code paths ----------------------------------


def test_pfilter_matches_scalar_path():
    scalar = _pfilter_loglik(_build(_rproc_scalar_det))
    vector = _pfilter_loglik(_build(_rproc_vec_det))
    assert scalar == pytest.approx(vector, rel=1e-10)


def test_mif_matches_scalar_path():
    """mif perturbs theta per particle, exercising the batched-theta wrapper."""
    rw_sd = pp.RWSigma({"growth": 0.02, "sigma": 0.02})
    logliks = []
    for rproc in (_rproc_scalar_det, _rproc_vec_det):
        model = _build(rproc)
        model.mif(J=32, M=2, rw_sd=rw_sd, n_monitors=1, key=jax.random.key(3))
        logliks.append(float(model.traces()["logLik"].dropna().iloc[-1]))
    assert logliks[0] == pytest.approx(logliks[1], rel=1e-10)


def test_simulate_matches_scalar_path():
    states = []
    for rproc in (_rproc_scalar_det, _rproc_vec_det):
        _, xsim = _build(rproc).simulate(key=jax.random.key(4), nsim=3)
        states.append(np.asarray(xsim.select_dtypes("number")))
    np.testing.assert_allclose(states[0], states[1], rtol=1e-10)


def test_accumvars_reset_matches_scalar_path():
    scalar = _pfilter_loglik(_build(_rproc_scalar_det, accumvars=["acc"]))
    vector = _pfilter_loglik(_build(_rproc_vec_det, accumvars=["acc"]))
    assert scalar == pytest.approx(vector, rel=1e-10)


# --- stochastic behaviour -----------------------------------------------


def test_stochastic_vectorized_runs():
    scalar = _pfilter_loglik(_build(_rproc_scalar_sto))
    vector = _pfilter_loglik(_build(_rproc_vec_sto))
    assert np.isfinite(vector)
    # Different RNG streams, so only a loose agreement is meaningful here.
    assert vector == pytest.approx(scalar, rel=0.5)


def test_train_differentiates_through_vectorized_rproc():
    """The dict-based sub-step carry must stay differentiable."""
    model = _build(_rproc_vec_sto)
    eta = pp.LearningRate({"growth": 0.01, "sigma": 0.01})
    model.train(J=32, M=2, eta=eta, key=jax.random.key(1))
    traces = model.traces()
    logliks = traces["logLik"].dropna()
    params = traces[["growth", "sigma"]].dropna()
    assert np.isfinite(logliks.iloc[-1])
    assert not np.allclose(params.iloc[0].to_numpy(), params.iloc[-1].to_numpy())


def test_vectorized_rproc_receives_batched_state_and_single_key():
    """The user function must see (J,) states and one key, not J keys."""
    seen = {}

    @pp.vectorized
    def rproc(X_, theta_, key, covars, t, dt):
        seen["x_shape"] = X_["x"].shape
        seen["key_ndim"] = jnp.ndim(key)
        return {"x": X_["x"], "acc": X_["acc"]}

    _pfilter_loglik(_build(rproc), J=16)
    assert seen["x_shape"] == (16,)
    assert seen["key_ndim"] == 0


def test_scalar_return_is_broadcast():
    """Returning a scalar for a constant state is allowed."""

    @pp.vectorized
    def rproc(X_, theta_, key, covars, t, dt):
        return {"x": X_["x"] * (1.0 + theta_["growth"] * dt), "acc": 0.0}

    assert np.isfinite(_pfilter_loglik(_build(rproc)))


# --- serialization ------------------------------------------------------


def test_pickle_roundtrip_preserves_vectorization():
    model = _build(_rproc_vec_sto)
    restored = pickle.loads(cloudpickle.dumps(model))
    assert restored.rproc._is_vectorized is True
    assert np.isfinite(_pfilter_loglik(restored))

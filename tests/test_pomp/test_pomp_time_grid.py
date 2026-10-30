"""Observation intervals of no length, on models with an exact likelihood."""

import pickle

import jax
import numpy as np
import pandas as pd
import pytest

import pypomp as pp

SD = 0.7
X0 = 2.0


def rinit(theta_, key, covars, t0):
    return {"x": theta_["x0"], "n": 0.0}


def map_rproc(X_, theta_, key, covars, t, dt):
    """A map: each step adds one whatever dt is, so every step shows in the state."""
    return {"x": X_["x"] + 1.0, "n": X_["n"] + 1.0}


def rate_rproc(X_, theta_, key, covars, t, dt):
    """Noise per unit time, as in dW/dt, which is 0/0 at dt = 0."""
    dW = theta_["sd"] * dt * jax.random.normal(key)
    return {"x": X_["x"] + 0.01 * dW / dt, "n": X_["n"] + 1.0}


def dmeas(Y_, X_, theta_, covars, t):
    return jax.scipy.stats.norm.logpdf(
        Y_["y"], X_["x"] + X_["n"] + covars["c"], theta_["sd"]
    )


def rmeas(X_, theta_, key, covars, t):
    return {"y": X_["x"] + X_["n"] + covars["c"]}


def make(t0, times, rproc=map_rproc, **grid):
    times = np.asarray(times, dtype=float)
    ctimes = np.array([min(t0, times[0]) - 1.0, times[-1] + 1.0])
    return pp.Pomp(
        ys=pd.DataFrame({"y": np.arange(times.size, dtype=float)}, index=times),
        theta=pp.PompParameters({"x0": X0, "sd": SD}),
        statenames=["x", "n"],
        t0=t0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        covars=pd.DataFrame({"c": ctimes}, index=ctimes),
        accumvars=["n"],
        **grid,
    )


def states(steps):
    """Return x and n at each observation; n counts only that interval's steps."""
    return X0 + np.cumsum(steps), np.asarray(steps, dtype=float)


def exact_logLik(times, steps):
    x, n = states(steps)
    y = np.arange(len(times), dtype=float)
    mean = x + n + np.asarray(times, dtype=float)
    return np.sum(-0.5 * ((y - mean) / SD) ** 2 - np.log(SD * np.sqrt(2 * np.pi)))


CASES = {
    "first-empty-nstep": (0.0, [0, 1, 2, 3], {"nstep": 3}, [0, 3, 3, 3], 3, True),
    "first-empty-dt": (0.0, [0, 1, 2, 3], {"dt": 0.25}, [0, 4, 4, 4], 4, True),
    "repeat-nstep": (0.0, [1, 2, 2, 3], {"nstep": 3}, [3, 3, 0, 3], 3, True),
    "last-empty-nstep": (0.0, [1, 2, 2], {"nstep": 3}, [3, 3, 0], 3, True),
    "uneven-dt": (0.0, [0, 1, 3], {"dt": 0.25}, [0, 4, 8], None, False),
    "only-observation": (0.0, [0], {"nstep": 3}, [0], 0, False),
    "no-empty-interval": (0.0, [1, 2, 3], {"nstep": 3}, [3, 3, 3], 3, False),
}


@pytest.fixture(params=CASES.values(), ids=list(CASES))
def case(request):
    t0, times, grid, steps, nstep, skip = request.param
    return make(t0, times, **grid), times, steps, nstep, skip


def test_step_counts(case):
    model, _, steps, nstep, skip = case
    assert model._nstep_array.tolist() == steps
    assert model.rproc.nstep == nstep
    assert model.rproc.skip_empty is skip


def test_pfilter_matches_the_exact_likelihood(case):
    model, times, steps, _, _ = case
    model.pfilter(J=3, key=jax.random.key(0))
    ll = np.asarray(model.results_history[-1].logLiks.values).ravel()
    assert ll == pytest.approx(exact_logLik(times, steps), rel=1e-5)


def test_simulate_matches_the_exact_states(case):
    model, times, steps, _, _ = case
    X, Y = model.simulate(key=jax.random.key(0))
    x, n = states(steps)
    assert np.allclose(X["x"].to_numpy()[1:], x)
    assert np.allclose(X["n"].to_numpy()[1:], n)
    assert np.allclose(Y["y"].to_numpy(), x + n + np.asarray(times, dtype=float))


def test_mif_matches_the_exact_likelihood(case):
    model, times, steps, _, _ = case
    rw_sd = pp.RWSigma(sigmas={"x0": 0.0, "sd": 0.0})
    model.mif(J=3, M=1, rw_sd=rw_sd, key=jax.random.key(0))
    traces = model.results_history[-1].traces_da
    ll = float(traces.sel(variable="logLik").isel(iteration=-1).squeeze())
    assert ll == pytest.approx(exact_logLik(times, steps), rel=1e-5)


def test_pickle_keeps_empty_intervals():
    model = make(0.0, [0, 1, 2], nstep=3)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.rproc.skip_empty
    assert restored.rproc.nstep == 3


@pytest.mark.parametrize("grid", [{"nstep": 3}, {"dt": 0.25}], ids=["nstep", "dt"])
def test_train_survives_a_first_interval_of_no_length(grid):
    model = make(0.0, [0, 1, 2, 3], rate_rproc, **grid)
    model.train(
        J=4,
        M=2,
        eta=pp.LearningRate({"x0": 0.01, "sd": 0.01}),
        optimizer=pp.Adam(),
        alpha=1.0,
        key=jax.random.key(0),
    )
    traces = model.results_history[-1].traces_da
    assert np.isfinite(traces.values).all()

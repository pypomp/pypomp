import warnings
from copy import deepcopy

import jax
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
from pypomp.functional.train import _stalled_params

J_DEFAULT = 2
M_DEFAULT = 2


@pytest.fixture(scope="module")
def simple_sir_for_dpop():
    """A small SIR Pomp model, shortened to two observations to keep setup fast."""
    model = pp.models.sir(times=np.array([0.2, 0.4]))
    return model


def _eta(model):
    return pp.LearningRate({name: 0.01 for name in model.canonical_param_names})


@pytest.mark.parametrize(
    "optimizer, eta_type",
    [
        (pp.Adam(), "constant"),
        (pp.SGD(), "constant"),
        (pp.SGD(), "hyperbolic"),
        (pp.Adam(ls=True), "constant"),
        (pp.Newton(), "constant"),
    ],
    ids=["adam", "sgd", "sgd-hyperbolic", "adam-linesearch", "newton"],
)
def test_dpop_train_variants(simple_sir_for_dpop, optimizer, eta_type):
    """DPOP uses train's step, so every train optimizer is supported."""
    model = simple_sir_for_dpop
    eta = _eta(model)
    if eta_type == "hyperbolic":
        eta = eta.hyperbolic_decay(0.1, M=M_DEFAULT)

    model.results_history.clear()
    ret = model.train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer=optimizer,
        alpha=0.8,
        dpop=True,
        key=jax.random.key(1),
    )
    assert ret is None
    res = model.results_history[-1]
    assert res.method == "train"
    assert res.dpop is True
    assert res.kind == "trace"
    traces = res.traces()
    assert not traces.empty
    assert np.all(np.isfinite(traces["logLik"]))


def test_dpop_train_param_order_invariance(simple_sir_for_dpop):
    """
    Check that DPOP training is invariant to the ordering of
    parameter dictionary keys (in natural space).
    """
    model = simple_sir_for_dpop

    eta = _eta(model)
    initial_theta = deepcopy(model.theta)

    # First run: default theta ordering
    model.results_history.clear()
    model.train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer=pp.SGD(),
        alpha=0.8,
        key=jax.random.key(123),
        theta=deepcopy(initial_theta),
        dpop=True,
    )
    res1 = model.results_history[-1]

    # Build a permuted theta with reversed key order
    theta_orig = initial_theta.params(as_list=True)  # list[dict]
    param_keys = list(theta_orig[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta_orig]

    # Second run: same random key & hyper-parameters, but permuted theta
    model.train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer=pp.SGD(),
        alpha=0.8,
        key=jax.random.key(123),
        theta=pp.PompParameters(permuted_theta),
        dpop=True,
    )
    res2 = model.results_history[-1]

    # Histories should match exactly up to numerical precision
    np.testing.assert_allclose(
        res1.traces()["logLik"], res2.traces()["logLik"], atol=1e-7
    )


def test_dpop_train_alpha_cooling(simple_sir_for_dpop):
    """Cooling leaves alpha unchanged at the first step, so the first update
    matches the uncooled run while the second one differs."""
    model = simple_sir_for_dpop
    initial_theta = deepcopy(model.theta)
    kwargs = dict(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=_eta(model),
        optimizer=pp.SGD(),
        alpha=0.8,
        dpop=True,
        key=jax.random.key(321),
    )
    names = model.canonical_param_names

    model.train(theta=deepcopy(initial_theta), alpha_cooling=1.0, **kwargs)
    fixed = model.results_history[-1].traces_da.sel(theta_idx=0, variable=names)
    model.train(theta=deepcopy(initial_theta), alpha_cooling=0.1, **kwargs)
    cooled = model.results_history[-1].traces_da.sel(theta_idx=0, variable=names)

    np.testing.assert_array_equal(fixed.sel(iteration=1), cooled.sel(iteration=1))
    assert not np.allclose(fixed.sel(iteration=2), cooled.sel(iteration=2))


def test_dpop_train_final_theta_loglik_1d_and_pruned(simple_sir_for_dpop):
    """DPOP training sets self.theta.logLik as a 1D array matching the final iteration."""
    model = simple_sir_for_dpop
    model.theta = model.theta * 2

    model.results_history.clear()
    model.train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=_eta(model),
        optimizer=pp.Adam(),
        alpha=0.8,
        dpop=True,
        key=jax.random.key(1),
    )

    res = model.results_history[-1]
    final_logliks = np.asarray(res.traces_da.isel(iteration=-1).sel(variable="logLik"))

    assert model.theta.logLik.ndim == 1
    assert model.theta.logLik.shape == (2,)
    np.testing.assert_allclose(model.theta.logLik, final_logliks)

    # Verify that pruning works on the resulting theta without error
    pruned = model.theta.pruned(n=1, refill=False)
    assert pruned.logLik.ndim == 1
    assert pruned.logLik.shape == (1,)
    assert pruned.num_replicates() == 1


def test_dpop_train_warns_once(simple_sir_for_dpop):
    """The experimental warning is raised once per call, at the caller."""
    model = simple_sir_for_dpop
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.train(
            J=J_DEFAULT,
            M=1,
            eta=_eta(model),
            dpop=True,
            key=jax.random.key(0),
        )
    experimental = [w for w in caught if "experimental" in str(w.message)]
    assert len(experimental) == 1
    assert experimental[0].filename == __file__


_TIMES = np.arange(1.0, 6.0)


def _poisson_model(update_logw=True, accumvars=None, logw_step=None):
    """X grows by Poisson(lam * dt) increments; ``y`` is X plus Gaussian noise.

    ``logw_step`` replaces the Poisson log-weight with a constant per step.
    """

    def rinit(theta_, key, covars=None, t0=None):
        return {"X": 5.0, "_logw": 0.0}

    def rproc(X_, theta_, key, covars, t, dt):
        mean = theta_["lam"] * dt
        n = pp.random.fast_poisson(key, mean)
        if logw_step is not None:
            increment = logw_step
        elif update_logw:
            increment = pp.random.poisson_logw(n, mean)
        else:
            increment = 0.0
        return {"X": X_["X"] + n, "_logw": X_["_logw"] + increment}

    def dmeas(Y_, X_, theta_, covars=None, t=None):
        return jax.scipy.stats.norm.logpdf(Y_["y"], X_["X"], 3.0)

    def rmeas(X_, theta_, key, covars=None, t=None):
        return {"y": X_["X"] + 3.0 * jax.random.normal(key)}

    return pp.Pomp(
        ys=pd.DataFrame({"y": 5.0 + 2.0 * _TIMES}, index=pd.Index(_TIMES)),
        theta=pp.PompParameters({"lam": 1.0}),
        statenames=["X", "_logw"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        nstep=4,
        accumvars=accumvars,
    )


def _train_user_warnings(model, dpop):
    """Train briefly and return the messages of the UserWarnings raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.train(
            J=20,
            M=2,
            eta=pp.LearningRate({"lam": 0.01}),
            optimizer=pp.SGD(),
            dpop=dpop,
            key=jax.random.key(0),
        )
    return [str(w.message) for w in caught if w.category is UserWarning]


@pytest.mark.parametrize(
    "accumvars, expected",
    [(None, ["_logw"]), (["X"], ["X", "_logw"]), (["_logw", "X"], ["_logw", "X"])],
)
def test_logw_is_always_an_accumvar(accumvars, expected):
    model = _poisson_model(accumvars=accumvars)
    assert model.accumvars == expected
    assert model.rproc.accumvars == tuple(
        model.statenames.index(name) for name in expected
    )


def test_simulate_resets_logw_each_interval():
    """Every method, not only DPOP training, resets ``_logw``."""
    model = _poisson_model(logw_step=-1.0)
    X_sims, _ = model.simulate(key=jax.random.key(0), nsim=1)
    np.testing.assert_allclose(X_sims["_logw"].to_numpy(), [0.0] + [-4.0] * len(_TIMES))


def test_train_without_dpop_warns_for_logw_model():
    messages = _train_user_warnings(_poisson_model(), dpop=False)
    assert any("only DPOP uses" in m for m in messages)
    stalled = [m for m in messages if "did not change ['lam']" in m]
    assert len(stalled) == 1
    assert "use dpop=True" in stalled[0]


def test_dpop_train_warns_when_logw_is_not_updated():
    messages = _train_user_warnings(_poisson_model(update_logw=False), dpop=True)
    stalled = [m for m in messages if "did not change ['lam']" in m]
    assert len(stalled) == 1
    assert "rproc must add" in stalled[0]


def test_dpop_train_does_not_warn_when_params_move():
    assert _train_user_warnings(_poisson_model(), dpop=True) == []


def test_stalled_params():
    eta = pp.LearningRate({"a": 0.1, "b": 0.1, "c": 0.0, "d": 0.1})
    names = ["a", "b", "c", "d"]
    # (reps, iterations, params): a moves in one replicate only, b never
    # moves, c has a zero rate, d becomes NaN.
    traces = np.zeros((2, 3, 4))
    traces[1, 2, 0] = 1.0
    traces[0, 1:, 3] = np.nan
    assert _stalled_params(traces, eta, names, M=2) == ["b"]
    # Unit-specific traces carry a unit axis before the parameters.
    unit_traces = np.zeros((1, 3, 2, 4))
    unit_traces[0, 1, 1, 1] = 1.0
    assert _stalled_params(unit_traces, eta, names, M=2) == ["a", "d"]
    assert _stalled_params(traces[:, :1], eta, names, M=0) == []

import warnings
from copy import deepcopy

import jax
import numpy as np
import pytest

import pypomp as pp

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

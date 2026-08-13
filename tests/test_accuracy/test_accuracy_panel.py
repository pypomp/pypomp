"""Statistical accuracy of the panel algorithms against exact results.

The three cases per algorithm cover the sharing patterns a panel can have:
one parameter shared, all shared, and none shared.
"""

import jax
import numpy as np
import pytest

import pypomp as pp
from tests.helpers.kalman import lg_panel_mle
from tests.helpers.lg_accuracy import (
    A,
    ESTIMATED,
    FIXED,
    Q,
    R,
    lg_1d_loglik,
    lg_1d_panel,
    rw_sd,
    theta_bounds,
)
from tests.helpers.plotting import save_traces_plotnine

pytestmark = pytest.mark.heavy

T = 100
BOUNDS = {A: (0.1, 1.0), Q: (0.1, 1.0), R: (0.1, 1.0)}

# True parameters per unit, and the perturbed values the fit starts from.
TRUE_UNITS = {
    "unit1": {A: 0.8, Q: 0.5, R: 0.3},
    "unit2": {A: 0.8, Q: 0.4, R: 0.2},
}
START_UNITS = {
    "unit1": {A: 0.5, Q: 0.8, R: 0.6},
    "unit2": {A: 0.5, Q: 0.7, R: 0.5},
}
# Every unit identical, for the all-shared case.
TRUE_SAME = {u: dict(TRUE_UNITS["unit1"]) for u in TRUE_UNITS}
START_SAME = {u: dict(START_UNITS["unit1"]) for u in START_UNITS}

MIXED = ([A], TRUE_UNITS, START_UNITS)
SHARED_ONLY = (ESTIMATED, TRUE_SAME, START_SAME)
UNIT_ONLY = ([], TRUE_UNITS, START_UNITS)


def _panel_mle(ys_by_unit, shared_names, true_units):
    """Exact panel MLE, started from each unit's own true parameters."""
    unit_names = [n for n in ESTIMATED if n not in shared_names]
    start = {**next(iter(true_units.values())), **FIXED}
    for unit, params in true_units.items():
        start.update({f"{n}_{unit}": v for n, v in params.items()})
    return lg_panel_mle(ys_by_unit, shared_names, unit_names, FIXED, start)


def _check(panel, ys_by_unit, shared_names, targets, plot, true_units):
    final = panel.theta.params(as_list=False)
    mean_shared = final["shared"].mean(dim="theta_idx")
    mean_unit = final["unit_specific"].mean(dim="theta_idx")
    mle = _panel_mle(ys_by_unit, shared_names, true_units)

    for (param, unit), tolerance in targets.items():
        if unit is None:
            est = mean_shared.sel(parameter=param).item()
            exact = mle[param]
        else:
            est = mean_unit.sel(unit=unit, parameter=param).item()
            exact = mle[f"{param}_{unit}"]
        assert np.abs(est - exact) < tolerance, (
            f"parameter={param}, unit={unit}: est={est}, mle={exact}"
        )

    true_ll = sum(
        lg_1d_loglik(ys_by_unit[u], p[A], p[Q], p[R]) for u, p in true_units.items()
    )
    save_traces_plotnine(panel, plot, true_values={"logLik": true_ll}, mle_values=mle)


def test_panel_pfilter_accuracy():
    """Per-unit log-likelihoods match the exact Kalman values."""
    key = jax.random.key(1234)
    panel, ys_by_unit = lg_1d_panel(TRUE_UNITS, TRUE_UNITS, [A], T=T, key=key)

    panel.pfilter(J=5000, reps=30, key=key)
    logLiks = panel.results_history[-1].logLiks

    for unit, params in TRUE_UNITS.items():
        exact = lg_1d_loglik(ys_by_unit[unit], params[A], params[Q], params[R])
        est = logLiks.sel(unit=unit).mean().item()
        assert np.abs(est - exact) < 0.225, f"{unit}: est={est}, exact={exact}"


@pytest.mark.parametrize(
    "case, targets, plot",
    [
        (
            MIXED,
            {
                (A, None): 0.18,
                (Q, "unit1"): 0.12,
                (R, "unit1"): 0.09,
                (Q, "unit2"): 0.09,
                (R, "unit2"): 0.105,
            },
            "tests/plots/panel_mif_traces.png",
        ),
        (
            SHARED_ONLY,
            {(A, None): 0.18, (Q, None): 0.15, (R, None): 0.225},
            "tests/plots/panel_mif_shared_only_traces.png",
        ),
        (
            UNIT_ONLY,
            {
                (A, "unit1"): 0.18,
                (Q, "unit1"): 0.12,
                (R, "unit1"): 0.09,
                (A, "unit2"): 0.18,
                (Q, "unit2"): 0.09,
                (R, "unit2"): 0.105,
            },
            "tests/plots/panel_mif_unit_specific_only_traces.png",
        ),
    ],
    ids=["mixed", "shared_only", "unit_specific_only"],
)
def test_panel_mif_accuracy(case, targets, plot):
    """Panel mif converges toward the exact panel MLE under each sharing pattern."""
    shared_names, true_units, start_units = case
    key = jax.random.key(1234)
    panel, ys_by_unit = lg_1d_panel(true_units, start_units, shared_names, T=T, key=key)

    panel.theta = pp.PanelPomp.sample_params(
        param_bounds=theta_bounds(BOUNDS),
        units=list(true_units),
        n=5,
        key=key,
        shared_names=list(shared_names),
    )
    panel.mif(J=3000, M=100, rw_sd=rw_sd(), key=key)

    _check(panel, ys_by_unit, shared_names, targets, plot, true_units)


@pytest.mark.parametrize(
    "case, targets, plot",
    [
        (
            MIXED,
            {
                (A, None): 0.18,
                (Q, "unit1"): 0.105,
                (R, "unit1"): 0.18,
                (Q, "unit2"): 0.075,
                (R, "unit2"): 0.09,
            },
            "tests/plots/panel_train_traces.png",
        ),
        (
            SHARED_ONLY,
            {(A, None): 0.18, (Q, None): 0.105, (R, None): 0.18},
            "tests/plots/panel_train_shared_only_traces.png",
        ),
        (
            UNIT_ONLY,
            {
                (A, "unit1"): 0.18,
                (Q, "unit1"): 0.105,
                (R, "unit1"): 0.18,
                (A, "unit2"): 0.18,
                (Q, "unit2"): 0.075,
                (R, "unit2"): 0.09,
            },
            "tests/plots/panel_train_unit_specific_only_traces.png",
        ),
    ],
    ids=["mixed", "shared_only", "unit_specific_only"],
)
def test_panel_train_accuracy(case, targets, plot):
    """Panel train converges toward the exact panel MLE under each sharing pattern."""
    shared_names, true_units, start_units = case
    key = jax.random.key(1234)
    panel, ys_by_unit = lg_1d_panel(true_units, start_units, shared_names, T=T, key=key)

    panel.theta = pp.PanelPomp.sample_params(
        param_bounds=theta_bounds(BOUNDS),
        units=list(true_units),
        n=5,
        key=key,
        shared_names=list(shared_names),
    )
    eta = pp.LearningRate({n: 0.05 for n in ESTIMATED} | {n: 0.0 for n in FIXED})
    panel.train(
        J=1000,
        M=150,
        eta=eta.cosine_decay(0.05, M=150),
        optimizer=pp.Adam(scale=True, beta1=0.8),
        alpha=1.0,
        key=key,
    )

    _check(panel, ys_by_unit, shared_names, targets, plot, true_units)

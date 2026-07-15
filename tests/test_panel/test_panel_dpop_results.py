import pytest
import numpy as np
import pandas as pd
import jax
from copy import deepcopy

import pypomp as pp
from pypomp.core.results import PanelPompDpopTrainResult

_test_times = np.arange(1 / 52, 3 / 52, 1 / 52)


def _get_sir_panel_for_results():
    sir1 = pp.models.sir(seed=100, times=_test_times)
    sir2 = pp.models.sir(seed=200, times=_test_times)
    param_names = sir1.canonical_param_names
    theta1 = sir1.theta[0]
    theta2 = sir2.theta[0]

    unit_specific = pd.DataFrame(
        {
            "unit1": [theta1[p] for p in param_names],
            "unit2": [theta2[p] for p in param_names],
        },
        index=pd.Index(param_names),
    )
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])
    panel = pp.PanelPomp(
        Pomp_dict={"unit1": sir1, "unit2": sir2},
        theta=theta,
    )
    return panel


def _get_dpop_result(seed=0):
    panel = _get_sir_panel_for_results()
    panel.dpop_train(
        J=2,
        M=2,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer=pp.Adam(),
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(seed),
    )
    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    return res


def test_dpop_result_equality():
    res1 = _get_dpop_result(seed=0)
    res2 = deepcopy(res1)
    res3 = _get_dpop_result(seed=1)

    assert res1 == res2
    assert res1 != res3

    # Test mismatch on other attributes
    res_diff_opt = deepcopy(res1)
    res_diff_opt.optimizer = pp.SGD()
    assert res1 != res_diff_opt

    res_diff_j = deepcopy(res1)
    res_diff_j.J = 999
    assert res1 != res_diff_j

    res_diff_m = deepcopy(res1)
    res_diff_m.M = 999
    assert res1 != res_diff_m

    res_diff_eta = deepcopy(res1)
    res_diff_eta.eta = 0.5
    assert res1 != res_diff_eta

    res_diff_alpha = deepcopy(res1)
    res_diff_alpha.alpha = 0.5
    assert res1 != res_diff_alpha

    res_diff_alpha_cooling = deepcopy(res1)
    res_diff_alpha_cooling.alpha_cooling = 0.5
    assert res1 != res_diff_alpha_cooling

    res_diff_state = deepcopy(res1)
    res_diff_state.process_weight_state = "diff_state"
    assert res1 != res_diff_state

    res_diff_decay = deepcopy(res1)
    res_diff_decay.decay = 9.9
    assert res1 != res_diff_decay


def test_dpop_result_empty_traces():
    # If traces are empty, traces() should return an empty DataFrame
    res = PanelPompDpopTrainResult(
        method="dpop_train",
        execution_time=0.1,
        key=jax.random.key(0),
    )
    assert res.traces().empty
    assert res.CLL().empty
    assert res.ESS().empty


def test_dpop_result_print_summary(capsys):
    res = _get_dpop_result(seed=0)
    res.print_summary()
    out = capsys.readouterr().out
    assert "Method: dpop_train" in out
    assert "Optimizer:" in out
    assert "Number of particles" in out


def test_dpop_result_merge():
    res1 = _get_dpop_result(seed=0)
    res2 = _get_dpop_result(seed=1)

    # Happy path
    merged = PanelPompDpopTrainResult.merge(res1, res2)
    assert isinstance(merged, PanelPompDpopTrainResult)
    assert merged.shared_traces.shape[0] == 2
    assert merged.unit_traces.shape[0] == 2
    assert merged.logLiks.shape[0] == 2

    # Raises ValueError when no results provided
    with pytest.raises(ValueError, match="At least one"):
        PanelPompDpopTrainResult.merge()

    # Raises TypeError when mismatched type
    with pytest.raises(TypeError, match="All merged objects must be of type"):
        PanelPompDpopTrainResult.merge(res1, "not_a_result")  # type: ignore

    # Raises ValueError when mismatched parameters
    res_diff = deepcopy(res2)
    res_diff.J = 999
    with pytest.raises(ValueError, match="must have the same J"):
        PanelPompDpopTrainResult.merge(res1, res_diff)

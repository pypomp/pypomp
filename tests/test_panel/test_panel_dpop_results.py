import pytest
import numpy as np
import pandas as pd
import jax
from copy import deepcopy

import pypomp as pp
from pypomp.core.results import PanelPompDpopTrainResult

_test_times = np.arange(1 / 52, 3 / 52, 1 / 52)


def _build_sir_panel_for_results():
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
    return panel, theta


@pytest.fixture(scope="module")
def dpop_results_module():
    """Run dpop_train with seed=0 and seed=1 once per module; cache both results."""
    panel, theta = _build_sir_panel_for_results()

    def _run(seed):
        p = deepcopy(panel)
        p.theta = deepcopy(theta)
        p.results_history.clear()
        p.dpop_train(
            J=2,
            M=2,
            eta=0.01,
            theta=deepcopy(p.theta),
            chunk_size=1,
            optimizer=pp.Adam(),
            alpha=0.8,
            process_weight_state="logw",
            key=jax.random.key(seed),
        )
        res = p.results_history[-1]
        assert isinstance(res, PanelPompDpopTrainResult)
        return res

    res0 = _run(seed=0)
    res1 = _run(seed=1)
    return res0, res1


def test_dpop_result_equality(dpop_results_module):
    res0, res1 = dpop_results_module
    res1_copy = deepcopy(res0)
    res_different = res1

    assert res0 == res1_copy
    assert res0 != res_different

    # Test mismatch on other attributes
    res_diff_opt = deepcopy(res0)
    res_diff_opt.optimizer = pp.SGD()
    assert res0 != res_diff_opt

    res_diff_j = deepcopy(res0)
    res_diff_j.J = 999
    assert res0 != res_diff_j

    res_diff_m = deepcopy(res0)
    res_diff_m.M = 999
    assert res0 != res_diff_m

    res_diff_eta = deepcopy(res0)
    res_diff_eta.eta = 0.5
    assert res0 != res_diff_eta

    res_diff_alpha = deepcopy(res0)
    res_diff_alpha.alpha = 0.5
    assert res0 != res_diff_alpha

    res_diff_alpha_cooling = deepcopy(res0)
    res_diff_alpha_cooling.alpha_cooling = 0.5
    assert res0 != res_diff_alpha_cooling

    res_diff_state = deepcopy(res0)
    res_diff_state.process_weight_state = "diff_state"
    assert res0 != res_diff_state

    res_diff_decay = deepcopy(res0)
    res_diff_decay.decay = 9.9
    assert res0 != res_diff_decay


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


def test_dpop_result_print_summary(dpop_results_module):
    res0, _ = dpop_results_module
    res0.print_summary()


def test_dpop_result_merge(dpop_results_module):
    res0, res1 = dpop_results_module

    # Happy path
    merged = PanelPompDpopTrainResult.merge(res0, res1)
    assert isinstance(merged, PanelPompDpopTrainResult)
    assert merged.shared_traces.shape[0] == 2
    assert merged.unit_traces.shape[0] == 2
    assert merged.logLiks.shape[0] == 2

    # Raises ValueError when no results provided
    with pytest.raises(ValueError, match="At least one"):
        PanelPompDpopTrainResult.merge()

    # Raises TypeError when mismatched type
    with pytest.raises(TypeError, match="All merged objects must be of type"):
        PanelPompDpopTrainResult.merge(res0, "not_a_result")  # type: ignore

    # Raises ValueError when mismatched parameters
    res_diff = deepcopy(res1)
    res_diff.J = 999
    with pytest.raises(ValueError, match="must have the same J"):
        PanelPompDpopTrainResult.merge(res0, res_diff)

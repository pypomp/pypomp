from copy import deepcopy

import jax
import numpy as np
import pytest

import pypomp as pp
from pypomp.core.results import Result
from tests.helpers.models import sir_panel

_test_times = np.arange(1 / 52, 3 / 52, 1 / 52)


def _build_sir_panel_for_results():
    panel = sir_panel(sharing="none", times=_test_times)
    return panel, panel.theta


@pytest.fixture(scope="module")
def dpop_results_module():
    """Run dpop_train with seed=0 and seed=1 once per module; cache both results."""
    panel, theta = _build_sir_panel_for_results()

    def _run(seed):
        p = deepcopy(panel)
        p.theta = deepcopy(theta)
        p.results_history.clear()
        p._dpop_train(
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
        assert isinstance(res, Result)
        assert res.method == "dpop_train"
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

    # Test mismatch on config entries
    for key, val in [
        ("optimizer", pp.SGD()),
        ("J", 999),
        ("M", 999),
        ("eta", 0.5),
        ("alpha", 0.5),
        ("alpha_cooling", 0.5),
        ("process_weight_state", "diff_state"),
    ]:
        res_diff = deepcopy(res0)
        res_diff.config[key] = val
        assert res0 != res_diff


def test_dpop_result_empty_traces():
    # If traces are empty, traces() should return an empty DataFrame
    res = Result(
        method="dpop_train",
        kind="trace",
        panel=True,
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
    merged = Result.merge(res0, res1)
    assert isinstance(merged, Result)
    assert merged.shared_traces.shape[0] == 2
    assert merged.unit_traces.shape[0] == 2
    assert merged.logLiks.shape[0] == 2

    # Raises ValueError when no results provided
    with pytest.raises(ValueError, match="At least one"):
        Result.merge()

    # Raises TypeError when mismatched type
    with pytest.raises(TypeError, match="All merged objects must be of type"):
        Result.merge(res0, "not_a_result")  # type: ignore

    # Raises ValueError when mismatched parameters
    res_diff = deepcopy(res1)
    res_diff.config["J"] = 999
    with pytest.raises(ValueError, match="must have the same J"):
        Result.merge(res0, res_diff)

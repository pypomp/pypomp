from copy import deepcopy

import jax
import pandas as pd
import pytest
import xarray as xr

import pypomp as pp
from pypomp.core.results import PanelPompTrainResult


def _build_lg_panel():
    lg1 = pp.models.LG()
    lg2 = pp.models.LG()
    shared_names = ["A1", "C1"]
    unit_specific_names = [
        n for n in lg1.canonical_param_names if n not in shared_names
    ]
    p1, p2 = lg1.theta[0], lg2.theta[0]
    shared_df = pd.DataFrame(
        {"shared": [(p1[n] + p2[n]) / 2 for n in shared_names]},
        index=pd.Index(shared_names),
    )
    unit_specific_df = pd.DataFrame(
        {
            "unit1": [p1[n] for n in unit_specific_names],
            "unit2": [p2[n] for n in unit_specific_names],
        },
        index=pd.Index(unit_specific_names),
    )
    theta = pp.PanelParameters(
        theta=[{"shared": shared_df, "unit_specific": unit_specific_df}]
    )
    return pp.PanelPomp(Pomp_dict={"unit1": lg1, "unit2": lg2}, theta=theta)


def _train_and_get_result(seed: int, optimizer: str = "Adam"):
    panel = _build_lg_panel()
    panel.train(
        J=2,
        M=2,
        eta=0.01,
        theta=deepcopy(panel.theta),
        optimizer=optimizer,
        key=jax.random.key(seed),
    )
    res = panel.results_history[-1]
    assert isinstance(res, PanelPompTrainResult)
    return res


@pytest.fixture(scope="module")
def two_compatible_results():
    return _train_and_get_result(seed=0), _train_and_get_result(seed=1)


def test_merge_happy_path(two_compatible_results):
    r1, r2 = two_compatible_results
    merged = PanelPompTrainResult.merge(r1, r2)
    assert isinstance(merged, PanelPompTrainResult)
    assert merged.optimizer == r1.optimizer
    assert merged.J == r1.J
    assert merged.M == r1.M
    # Concatenation along theta_idx should add the replicate counts.
    assert isinstance(merged.shared_traces, xr.DataArray)
    assert isinstance(merged.unit_traces, xr.DataArray)
    assert isinstance(merged.logLiks, xr.DataArray)
    assert merged.shared_traces.shape[0] == (
        r1.shared_traces.shape[0] + r2.shared_traces.shape[0]
    )
    assert merged.unit_traces.shape[0] == (
        r1.unit_traces.shape[0] + r2.unit_traces.shape[0]
    )


def test_merge_empty_args_raises():
    with pytest.raises(ValueError, match="At least one"):
        PanelPompTrainResult.merge()


def test_merge_wrong_type_raises(two_compatible_results):
    r1, _ = two_compatible_results
    with pytest.raises(TypeError, match="must be of type PanelPompTrainResult"):
        PanelPompTrainResult.merge(r1, "not_a_result")


def test_merge_mismatched_optimizer_raises():
    r1 = _train_and_get_result(seed=0, optimizer="Adam")
    r2 = _train_and_get_result(seed=1, optimizer="FullMatrixAdam")
    with pytest.raises(ValueError, match="same optimizer"):
        PanelPompTrainResult.merge(r1, r2)


def test_train_result_equality(two_compatible_results):
    r1, _ = two_compatible_results
    assert r1 == deepcopy(r1)


def test_train_result_traces_returns_long_dataframe(two_compatible_results):
    r1, _ = two_compatible_results
    df = r1.traces()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Long format: each row is one (theta_idx, iteration, unit) tuple.
    for col in ["theta_idx", "iteration", "unit"]:
        assert col in df.columns


def test_train_result_print_summary(two_compatible_results, capsys):
    r1, _ = two_compatible_results
    r1.print_summary()
    out = capsys.readouterr().out
    # Spot-check the summary text mentions key fields.
    assert "Method" in out
    assert "Optimizer" in out
    assert "Number of particles" in out
    assert "Number of iterations" in out

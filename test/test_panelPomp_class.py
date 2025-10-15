import pandas as pd
import xarray as xr
import jax
import pypomp as pp
import pytest


@pytest.fixture(scope="function")
def measles_panel_setup():
    AK_mles = pp.UKMeasles.AK_mles()
    london_theta = AK_mles["London"].to_dict()
    hastings_theta = AK_mles["Hastings"].to_dict()
    london = pp.UKMeasles.Pomp(
        unit=["London"],
        theta=london_theta,
    )
    hastings = pp.UKMeasles.Pomp(
        unit=["Hastings"],
        theta=hastings_theta,
    )
    unit_specific = AK_mles[["London", "Hastings"]]
    assert isinstance(unit_specific, pd.DataFrame)
    panel = pp.PanelPomp(
        Pomp_dict={"London": london, "Hastings": hastings},
        shared=None,
        unit_specific=unit_specific,
    )
    key = jax.random.key(0)
    return panel, key


def test_get_unit_parameters(measles_panel_setup):
    panel, key = measles_panel_setup
    params = panel.get_unit_parameters(unit="London")
    assert isinstance(params, list)
    assert isinstance(params[0], dict)
    assert len(params) == panel._get_theta_list_len(panel.shared, panel.unit_specific)


def test_simulate(measles_panel_setup):
    panel, key = measles_panel_setup
    X_sim_order = ["unit", "replicate", "sim", "time"] + [
        f"state_{i}" for i in range(0, 6)
    ]
    Y_sim_order = ["unit", "replicate", "sim", "time", "obs_0"]

    X_sims, Y_sims = panel.simulate(key=key)

    assert isinstance(X_sims, pd.DataFrame)
    assert isinstance(Y_sims, pd.DataFrame)
    assert list(X_sims.columns) == X_sim_order
    assert list(Y_sims.columns) == Y_sim_order


def test_pfilter(measles_panel_setup):
    panel, key = measles_panel_setup
    panel.pfilter(J=2, key=key)
    assert isinstance(panel.results_history[-1]["logLiks"], xr.DataArray)
    assert panel.results_history[-1]["logLiks"].dims == ("theta", "unit", "replicate")
    assert panel.results_history[-1]["logLiks"].shape == (1, 2, 1)
    assert panel.results_history[-1]["shared"] is None
    assert panel.results_history[-1]["unit_specific"] is panel.unit_specific
    assert panel.results_history[-1]["J"] == 2
    assert panel.results_history[-1]["reps"] == 1
    assert panel.results_history[-1]["thresh"] == 0
    assert panel.results_history[-1]["key"] == key

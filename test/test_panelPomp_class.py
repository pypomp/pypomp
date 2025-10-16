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


@pytest.fixture(scope="function")
def measles_panel_setup2(measles_panel_setup):
    panel, key = measles_panel_setup
    panel.unit_specific = panel.unit_specific * 2
    return panel, key


def test_get_unit_parameters(measles_panel_setup2):
    panel, key = measles_panel_setup2
    params = panel.get_unit_parameters(unit="London")
    assert isinstance(params, list)
    assert isinstance(params[0], dict)
    assert len(params) == panel._get_theta_list_len(panel.shared, panel.unit_specific)


def test_simulate(measles_panel_setup2):
    panel, key = measles_panel_setup2
    X_sim_order = ["unit", "replicate", "sim", "time"] + [
        f"state_{i}" for i in range(0, 6)
    ]
    Y_sim_order = ["unit", "replicate", "sim", "time", "obs_0"]

    X_sims, Y_sims = panel.simulate(key=key)

    assert isinstance(X_sims, pd.DataFrame)
    assert isinstance(Y_sims, pd.DataFrame)
    assert list(X_sims.columns) == X_sim_order
    assert list(Y_sims.columns) == Y_sim_order


def test_pfilter(measles_panel_setup2):
    panel, key = measles_panel_setup2
    panel.pfilter(J=2, key=key)
    assert isinstance(panel.results_history[-1]["logLiks"], xr.DataArray)
    assert panel.results_history[-1]["logLiks"].dims == ("theta", "unit", "replicate")
    assert panel.results_history[-1]["logLiks"].shape == (2, 2, 1)
    assert panel.results_history[-1]["shared"] is None
    assert panel.results_history[-1]["unit_specific"] is panel.unit_specific
    assert panel.results_history[-1]["J"] == 2
    assert panel.results_history[-1]["reps"] == 1
    assert panel.results_history[-1]["thresh"] == 0
    assert panel.results_history[-1]["key"] == key


def test_mif(measles_panel_setup2):
    panel, key = measles_panel_setup2
    J = 2
    sigmas = 0.02
    sigmas_init = 0.1
    M = 2
    a = 0.5
    panel.mif(J=J, key=key, sigmas=sigmas, sigmas_init=sigmas_init, M=M, a=a)
    # Check that the last results are present and correct for 'mif'
    result = panel.results_history[-1]
    assert result["method"] == "mif"
    assert "shared_traces" in result
    assert "unit_traces" in result
    assert "unit_logliks" in result
    assert result["shared"] is None
    assert result["unit_specific"] is panel.unit_specific
    assert result["J"] == J
    assert result["M"] == M
    assert result["a"] == a
    assert result["sigmas"] == sigmas
    assert result["sigmas_init"] == sigmas_init
    assert isinstance(result["shared_traces"], xr.DataArray)
    assert isinstance(result["unit_traces"], xr.DataArray)
    assert isinstance(result["unit_logliks"], xr.DataArray)
    # additional shape checks
    assert "iteration" in result["shared_traces"].dims
    assert "variable" in result["shared_traces"].dims
    assert "iteration" in result["unit_traces"].dims
    assert "variable" in result["unit_traces"].dims
    assert "unit" in result["unit_traces"].dims
    assert result["shared_traces"].shape[1] == 3  # M+1 iterations where M=2, so 3
    assert result["unit_traces"].shape[1] == 3  # M+1 iterations where M=2, so 3
    assert set(result["unit_logliks"].coords["unit"].values) == set(
        panel.unit_objects.keys()
    )

import pandas as pd
import xarray as xr
import jax
import pypomp as pp
import pytest
import numpy as np
import pickle


@pytest.fixture(scope="module")
def measles_panel_setup_module():
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
    shared = panel.shared
    unit_specific = panel.unit_specific
    fresh_key = panel.fresh_key
    return panel, key, shared, unit_specific, fresh_key


@pytest.fixture(scope="function")
def measles_panel_setup(measles_panel_setup_module):
    panel, key, shared, unit_specific, fresh_key = measles_panel_setup_module
    panel.results_history.clear()
    panel.shared = shared
    panel.unit_specific = unit_specific
    panel.fresh_key = fresh_key
    return panel, key


@pytest.fixture(scope="function")
def measles_panel_setup2(measles_panel_setup):
    panel, key = measles_panel_setup
    panel.unit_specific = panel.unit_specific * 2
    return panel, key


@pytest.fixture(scope="module")
def measles_panel_mp_module(measles_panel_setup_module):
    panel, key, shared, unit_specific, fresh_key = measles_panel_setup_module
    J = 2
    M = 2
    a = 0.5
    sigmas = 0.02
    sigmas_init = 0.1
    panel.mif(J=J, key=key, sigmas=sigmas, sigmas_init=sigmas_init, M=M, a=a)
    panel.pfilter(J=J)
    results_history = panel.results_history
    fresh_key = panel.fresh_key
    return (
        panel,
        key,
        J,
        M,
        a,
        sigmas,
        sigmas_init,
        shared,
        unit_specific,
        fresh_key,
        results_history,
    )


@pytest.fixture(scope="function")
def measles_panel_mp(measles_panel_mp_module):
    (
        panel,
        key,
        J,
        M,
        a,
        sigmas,
        sigmas_init,
        shared,
        unit_specific,
        fresh_key,
        results_history,
    ) = measles_panel_mp_module
    panel.results_history = results_history
    panel.shared = shared
    panel.unit_specific = unit_specific
    panel.fresh_key = fresh_key
    return panel, key, J, M, a, sigmas, sigmas_init


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
    result = panel.results_history[-1]

    assert result["method"] == "mif"
    assert "shared_traces" in result
    assert "unit_traces" in result
    assert "logLiks" in result
    assert result["shared"] is None
    assert result["J"] == J
    assert result["M"] == M
    assert result["a"] == a
    assert result["sigmas"] == sigmas
    assert result["sigmas_init"] == sigmas_init
    assert isinstance(result["shared_traces"], xr.DataArray)
    assert isinstance(result["unit_traces"], xr.DataArray)
    assert isinstance(result["logLiks"], xr.DataArray)
    assert "iteration" in result["shared_traces"].dims
    assert "variable" in result["shared_traces"].dims
    assert "iteration" in result["unit_traces"].dims
    assert "variable" in result["unit_traces"].dims
    assert "unit" in result["unit_traces"].dims
    assert result["shared_traces"].shape[1] == M + 1
    assert result["unit_traces"].shape[1] == M + 1
    assert set(result["logLiks"].coords["unit"].values) == set(
        ["shared"] + list(panel.unit_objects.keys())
    )


def test_results(measles_panel_mp):
    panel, *_ = measles_panel_mp
    results0 = panel.results(0)
    results1 = panel.results(1)
    # Check expected columns for results0 (from mif, index=0)
    expected_cols = {
        "replicate",
        "unit",
        "shared logLik",
        "unit logLik",
    }
    # Add dynamic parameter columns (shared/unit) if present.
    results0_cols = set(results0.columns)
    assert expected_cols.issubset(results0_cols)
    # It should have one row for each replicate/unit combination
    n_units = len(panel.unit_objects)
    n_reps = results0["replicate"].nunique()
    assert len(results0) == n_units * n_reps

    # Check expected columns for results1 (from pfilter, index=1)
    results1_cols = set(results1.columns)
    assert expected_cols.issubset(results1_cols)
    n_units1 = len(panel.unit_objects)
    n_reps1 = results1["replicate"].nunique()
    assert len(results1) == n_units1 * n_reps1


def test_fresh_key(measles_panel_setup2):
    panel, key = measles_panel_setup2
    J = 2
    panel.pfilter(J=J, key=key)
    assert not np.array_equal(
        jax.random.key_data(panel.fresh_key), jax.random.key_data(key)
    )
    assert np.array_equal(
        jax.random.key_data(panel.results_history[0]["key"]),
        jax.random.key_data(key),
    )


def test_traces(measles_panel_mp):
    panel, *_ = measles_panel_mp
    traces = panel.traces()
    assert isinstance(traces, pd.DataFrame)
    assert "replicate" in traces.columns
    assert "unit" in traces.columns
    assert "iteration" in traces.columns
    assert "method" in traces.columns
    assert "logLik" in traces.columns


def test_time(measles_panel_mp):
    panel, *_ = measles_panel_mp
    time_df = panel.time()
    assert isinstance(time_df, pd.DataFrame)
    assert "method" in time_df.columns
    assert "time" in time_df.columns
    assert time_df.index.name == "history_index"


def test_pickle_panelpomp(measles_panel_mp):
    panel, *_ = measles_panel_mp

    pickled = pickle.dumps(panel)
    unpickled_panel = pickle.loads(pickled)

    # Check that the basic attributes are preserved
    assert isinstance(unpickled_panel, pp.PanelPomp)
    assert list(unpickled_panel.unit_objects.keys()) == list(panel.unit_objects.keys())
    assert unpickled_panel.shared is None  # This panel has no shared parameters
    assert isinstance(unpickled_panel.unit_specific, list)
    assert list(unpickled_panel.unit_specific[0].columns) == list(
        panel.unit_specific[0].columns
    )

    assert len(unpickled_panel.results_history) == len(panel.results_history)
    for orig, unpickled in zip(panel.results_history, unpickled_panel.results_history):
        assert orig.keys() == unpickled.keys()
        # Compare values for common keys; skip non-comparable objects
        for k in orig.keys():
            v1 = orig[k]
            v2 = unpickled[k]
            if isinstance(v1, np.ndarray):
                assert np.allclose(v1, v2)
            elif isinstance(v1, pd.DataFrame):
                pd.testing.assert_frame_equal(v1, v2)
            elif isinstance(v1, xr.DataArray):
                xr.testing.assert_equal(v1, v2)
            elif isinstance(v1, (float, int, str, type(None))):
                assert v1 == v2
            elif k == "key":
                assert np.array_equal(jax.random.key_data(v1), jax.random.key_data(v2))
            else:
                pass

    # Check that the unit objects are properly reconstructed
    for unit_name in panel.unit_objects.keys():
        original_unit = panel.unit_objects[unit_name]
        unpickled_unit = unpickled_panel.unit_objects[unit_name]

        assert isinstance(unpickled_unit, pp.Pomp)

        assert original_unit.ys.shape == unpickled_unit.ys.shape
        assert list(original_unit.ys.columns) == list(unpickled_unit.ys.columns)

        assert hasattr(unpickled_unit, "rinit")
        assert hasattr(unpickled_unit, "rproc")
        assert hasattr(unpickled_unit, "dmeas")

        assert callable(unpickled_unit.rinit.struct_per)
        assert callable(unpickled_unit.rproc.struct_per_interp)
        if unpickled_unit.dmeas is not None:
            assert callable(unpickled_unit.dmeas.struct_per)

    # check that the unpickled panel can be used for filtering
    unpickled_panel.pfilter(J=2)

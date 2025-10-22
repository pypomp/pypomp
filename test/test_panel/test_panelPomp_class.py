import pandas as pd
import xarray as xr
import jax
import pypomp as pp
import numpy as np
import pickle


def test_get_unit_parameters(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    params = panel.get_unit_parameters(unit="London")
    assert isinstance(params, list)
    assert isinstance(params[0], dict)
    assert len(params) == panel._get_theta_list_len(panel.shared, panel.unit_specific)


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


def test_fresh_key(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
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

        # Compare ys values, handling NaN values properly
        orig_ys_values = original_unit.ys.values
        unpickled_ys_values = unpickled_unit.ys.values
        assert orig_ys_values.shape == unpickled_ys_values.shape
        # Use numpy's array_equal which handles NaN values correctly
        assert np.array_equal(orig_ys_values, unpickled_ys_values, equal_nan=True)
        assert original_unit.theta == unpickled_unit.theta
        assert unpickled_unit.covars is not None
        assert (
            original_unit.covars.values.tolist()
            == unpickled_unit.covars.values.tolist()
        )
        assert original_unit.rinit == unpickled_unit.rinit
        assert original_unit.rproc.original_func == unpickled_unit.rproc.original_func
        assert original_unit.dmeas == unpickled_unit.dmeas
        assert original_unit.rproc.dt == unpickled_unit.rproc.dt
        assert original_unit.results_history == unpickled_unit.results_history
        assert (
            original_unit.traces().values.tolist()
            == unpickled_unit.traces().values.tolist()
        )

    # check that the unpickled panel can be used for filtering
    unpickled_panel.pfilter(J=2)


def test_sample_params(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    param_bounds = {
        "R0": [10.0, 60.0],
        "sigma": [25.0, 100.0],
        "gamma": [25.0, 320.0],
        "iota": [0.004, 3.0],
        "rho": [0.1, 0.9],
        "sigmaSE": [0.04, 0.1],
        "psi": [0.05, 3.0],
        "cohort": [0.1, 0.7],
        "amplitude": [0.1, 0.6],
        "S_0": [0.01, 0.07],
        "E_0": [0.000004, 0.0001],
        "I_0": [0.000003, 0.001],
        "R_0": [0.9, 0.99],
    }
    shared_names = ["gamma", "cohort"]
    shared_param_sets, unit_specific_param_sets = panel.sample_params(
        param_bounds=param_bounds,
        units=list(panel.unit_objects.keys()),
        n=2,
        key=key,
        shared_names=shared_names,
    )
    assert isinstance(shared_param_sets, list)
    assert isinstance(unit_specific_param_sets, list)
    assert len(shared_param_sets) == 2
    assert len(unit_specific_param_sets) == 2

    # Check that shared_param_sets DataFrames have correct index and column in correct order
    for shared_df in shared_param_sets:
        # Index should be shared_names, in order
        assert list(shared_df.index) == shared_names
        # Only one column which is exactly ["shared"]
        assert list(shared_df.columns) == ["shared"]

    # For unit_specific_param_sets, check columns and index are correct
    units = list(panel.unit_objects.keys())
    unit_specific_names = [
        name for name in param_bounds if name not in set(shared_names)
    ]
    for unit_df in unit_specific_param_sets:
        # Columns (units) are in the correct order
        assert list(unit_df.columns) == units
        # Index should be unit_specific_names, in order
        assert list(unit_df.index) == unit_specific_names

    # Also check that each value is within the specified bounds
    for shared_df in shared_param_sets:
        for name in shared_names:
            val = shared_df.loc[name, "shared"]
            lower, upper = param_bounds[name]
            assert lower <= val <= upper

    for unit_df in unit_specific_param_sets:
        for param_name in unit_specific_names:
            lower, upper = param_bounds[param_name]
            for unit in units:
                val = unit_df.loc[param_name, unit]
                assert lower <= val <= upper

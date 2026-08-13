import jax
import numpy as np
import pandas as pd
import xarray as xr

import pypomp.core.parameters as pp


def test_panel_parameters_dataset_init():
    # Construct an xr.Dataset manually and pass it to PanelParameters
    shared_da = xr.DataArray(
        [[10.0, 20.0], [11.0, 21.0]],
        dims=["theta_idx", "parameter"],
        coords={"theta_idx": [0, 1], "parameter": ["s1", "s2"]},
    )
    unit_specific_da = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]],
        dims=["theta_idx", "unit", "parameter"],
        coords={
            "theta_idx": [0, 1],
            "unit": ["u1", "u2"],
            "parameter": ["u1_param", "u2_param"],
        },
    )
    ds = xr.Dataset(
        data_vars={
            "shared": shared_da,
            "unit_specific": unit_specific_da,
        }
    )
    ds.attrs["shared_names"] = ["s1", "s2"]
    ds.attrs["unit_specific_names"] = ["u1_param", "u2_param"]

    params = pp.PanelParameters(ds)
    assert params.get_shared_param_names() == ["s1", "s2"]
    assert params.get_unit_param_names() == ["u1_param", "u2_param"]
    assert params.get_unit_names() == ["u1", "u2"]
    assert len(params) == 2


def test_panel_parameters_dict_init():
    # Construct using standard dict of DataFrames
    shared_df = pd.DataFrame({"shared": [10.0, 20.0]}, index=pd.Index(["s1", "s2"]))
    unit_specific_df = pd.DataFrame(
        {"unit1": [1.0, 2.0], "unit2": [3.0, 4.0]},
        index=pd.Index(["u1_param", "u2_param"]),
    )

    params = pp.PanelParameters(
        theta=[{"shared": shared_df, "unit_specific": unit_specific_df}]
    )

    assert params.get_shared_param_names() == ["s1", "s2"]
    assert params.get_unit_param_names() == ["u1_param", "u2_param"]
    assert params.get_unit_names() == ["unit1", "unit2"]

    # Verify internal dataset storage structure
    assert isinstance(params._data, xr.Dataset)
    assert "shared" in params._data
    assert "unit_specific" in params._data
    assert params._data["shared"].dims == ("theta_idx", "parameter")
    assert params._data["unit_specific"].dims == ("theta_idx", "unit", "parameter")


def test_dataframe_to_array_canonical(lg_panel_setup_some_shared):
    """`_dataframe_to_array_canonical` reorders a DataFrame column into the
    given canonical parameter order."""
    panel, _, _ = lg_panel_setup_some_shared
    df = pd.DataFrame({"col": [3.0, 1.0, 2.0]}, index=pd.Index(["b", "a", "c"]))

    arr = panel._dataframe_to_array_canonical(df, ["a", "b", "c"], "col")

    assert isinstance(arr, jax.Array)
    np.testing.assert_allclose(np.asarray(arr), [1.0, 3.0, 2.0])


def test_sample_params_no_shared_names(lg_panel_setup_some_shared):
    """Omitting shared_names means every parameter is unit-specific, which
    exercises the empty shared-array fallback in sample_params."""
    panel, _, key = lg_panel_setup_some_shared
    param_bounds = {n: (0.1, 1.0) for n in panel.canonical_param_names}
    units = list(panel.unit_objects.keys())

    param_sets = panel.sample_params(
        param_bounds=param_bounds, units=units, n=2, key=key
    )

    assert isinstance(param_sets, pp.PanelParameters)
    assert len(param_sets) == 2
    for param_set in param_sets:
        unit_df = param_set["unit_specific"]
        assert list(unit_df.index) == list(param_bounds.keys())
        assert list(unit_df.columns) == units
        for col in units:
            for name in param_bounds:
                val = unit_df.loc[name, col]
                lower, upper = param_bounds[name]
                assert lower <= val <= upper


def test_sample_params_all_shared(lg_panel_setup_some_shared):
    """When every parameter name is listed as shared, the unit-specific
    sampling path hits its empty-array fallback."""
    panel, _, key = lg_panel_setup_some_shared
    param_bounds = {n: (0.1, 1.0) for n in panel.canonical_param_names}
    units = list(panel.unit_objects.keys())
    shared_names = list(param_bounds.keys())

    param_sets = panel.sample_params(
        param_bounds=param_bounds,
        units=units,
        n=2,
        key=key,
        shared_names=shared_names,
    )

    assert isinstance(param_sets, pp.PanelParameters)
    for param_set in param_sets:
        shared_df = param_set["shared"]
        assert list(shared_df.index) == shared_names
        assert list(shared_df.columns) == ["shared"]
        for name in shared_names:
            val = shared_df.loc[name, "shared"]
            lower, upper = param_bounds[name]
            assert lower <= val <= upper

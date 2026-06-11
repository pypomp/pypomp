import xarray as xr
import pandas as pd
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

import xarray as xr


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

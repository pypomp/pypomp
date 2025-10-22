import xarray as xr


def test_pfilter_basic(measles_panel_setup_some_shared):
    """Test basic pfilter functionality with some shared parameters."""
    panel, rw_sd, key = measles_panel_setup_some_shared
    panel.pfilter(J=2, key=key)

    # Check results structure
    result = panel.results_history[-1]
    assert isinstance(result["logLiks"], xr.DataArray)
    assert result["logLiks"].dims == ("theta", "unit", "replicate")
    assert result["logLiks"].shape == (2, 2, 1)
    assert result["shared"] is panel.shared
    assert result["unit_specific"] is panel.unit_specific
    assert result["J"] == 2
    assert result["reps"] == 1
    assert result["thresh"] == 0
    assert result["key"] == key
    assert "execution_time" in result


def test_pfilter_unit_specific_only(measles_panel_setup_specific_only):
    """Test pfilter with unit-specific parameters only."""
    panel, rw_sd, key = measles_panel_setup_specific_only
    panel.pfilter(J=2, key=key)

    result = panel.results_history[-1]
    assert isinstance(result["logLiks"], xr.DataArray)
    assert result["logLiks"].dims == ("theta", "unit", "replicate")
    assert result["logLiks"].shape == (2, 2, 1)
    assert result["shared"] is None
    assert result["unit_specific"] is panel.unit_specific
    assert result["J"] == 2
    assert result["reps"] == 1
    assert result["thresh"] == 0
    assert result["key"] == key
    assert "execution_time" in result

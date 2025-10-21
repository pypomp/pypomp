import xarray as xr


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

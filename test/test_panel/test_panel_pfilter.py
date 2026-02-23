import xarray as xr
import numpy as np
from copy import deepcopy


def check_pfilter_result(result, theta_orig, J=2, reps=1, thresh=0, key=None):
    """Helper to verify common pfilter result attributes."""
    n_theta = theta_orig.num_replicates()
    n_units = len(result.logLiks.coords["unit"])
    assert isinstance(result.logLiks, xr.DataArray)
    assert result.logLiks.dims == ("theta", "unit", "replicate")
    assert result.logLiks.shape == (n_theta, n_units, reps)
    assert result.theta == theta_orig
    assert result.J == J
    assert result.reps == reps
    assert result.thresh == thresh
    if key is not None:
        assert result.key == key
    assert hasattr(result, "execution_time")


def test_pfilter_basic(measles_panel_setup_some_shared):
    """Test basic pfilter functionality with some shared parameters."""
    panel, rw_sd, key = measles_panel_setup_some_shared
    theta_orig = deepcopy(panel.theta)
    J = 2
    panel.pfilter(J=J, key=key)

    check_pfilter_result(panel.results_history[-1], theta_orig, J=J, key=key)


def test_pfilter_unit_specific_only(measles_panel_setup_specific_only):
    """Test pfilter with unit-specific parameters only."""
    panel, rw_sd, key = measles_panel_setup_specific_only
    theta_orig = deepcopy(panel.theta)
    J = 2
    panel.pfilter(J=J, key=key)

    check_pfilter_result(panel.results_history[-1], theta_orig, J=J, key=key)


def test_pfilter_diagnostics(measles_panel_setup_some_shared):
    """Test that CLL, ESS, filter_mean, and prediction_mean work correctly."""
    panel, rw_sd, key = measles_panel_setup_some_shared
    n_units = len(panel.unit_objects)
    n_theta = panel.theta.num_replicates()
    reps = 2
    J = 3

    first_unit = list(panel.unit_objects.values())[0]
    n_time = len(first_unit.ys)
    n_state = len(first_unit.statenames)

    for CLL, ESS, filter_mean, prediction_mean in [
        (False, False, False, False),
        (False, False, True, True),
        (True, True, True, True),
    ]:
        panel.results_history.clear()
        panel.pfilter(
            J=J,
            key=key,
            reps=reps,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
        )
        result = panel.results_history[-1]

        assert result.logLiks.shape == (n_theta, n_units, reps)
        assert result.logLiks.dims == ("theta", "unit", "replicate")

        if CLL:
            assert result.CLL is not None
            assert result.CLL.shape == (n_theta, n_units, reps, n_time)
            assert result.CLL.dims == ("theta", "unit", "replicate", "time")
            assert np.all(np.isfinite(result.CLL.data))
        else:
            assert result.CLL is None

        if ESS:
            assert result.ESS is not None
            assert result.ESS.shape == (n_theta, n_units, reps, n_time)
            assert result.ESS.dims == ("theta", "unit", "replicate", "time")
            assert np.all(np.isfinite(result.ESS.data))
            # ESS should be between 0 and J (allow small tolerance for floating point precision)
            assert np.all(result.ESS.data >= 0)
            assert np.all(result.ESS.data <= J + 1e-5)
        else:
            assert result.ESS is None

        if filter_mean:
            assert result.filter_mean is not None
            assert result.filter_mean.shape == (n_theta, n_units, reps, n_time, n_state)
            assert result.filter_mean.dims == (
                "theta",
                "unit",
                "replicate",
                "time",
                "state",
            )
        else:
            assert result.filter_mean is None

        if prediction_mean:
            assert result.prediction_mean is not None
            assert result.prediction_mean.shape == (
                n_theta,
                n_units,
                reps,
                n_time,
                n_state,
            )
            assert result.prediction_mean.dims == (
                "theta",
                "unit",
                "replicate",
                "time",
                "state",
            )
        else:
            assert result.prediction_mean is None

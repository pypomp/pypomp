import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Attempt to import statsmodels for tests
try:
    import statsmodels  # noqa: F401

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from pypomp import benchmarks
from pypomp import Pomp
from pypomp.panel.estimation_mixin import PanelEstimationMixin


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    return pd.DataFrame(
        {"y1": np.random.poisson(5, 50), "y2": np.random.poisson(10, 50)}
    )


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_arma_function(dummy_data):
    # Test univariate
    llf1 = benchmarks.arma(dummy_data[["y1"]], order=(1, 0, 0))
    assert isinstance(llf1, float)

    # Test multivariate
    llf_both = benchmarks.arma(dummy_data, order=(1, 0, 0))
    llf2 = benchmarks.arma(dummy_data[["y2"]], order=(1, 0, 0))

    # Should be the sum of individual likelihoods
    assert np.isclose(llf_both, llf1 + llf2)

    # Test log_ys
    # Implementation now uses log(y + 1)
    llf_log = benchmarks.arma(dummy_data, order=(1, 0, 0), log_ys=True)
    llf_manual_log = benchmarks.arma(
        np.log(dummy_data + 1), order=(1, 0, 0), log_ys=False
    )
    assert np.isclose(llf_log, llf_manual_log)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_negbin_function(dummy_data):
    # Test iid
    llf1 = benchmarks.negbin(dummy_data[["y1"]], autoregressive=False)
    assert isinstance(llf1, float)

    llf_both = benchmarks.negbin(dummy_data, autoregressive=False)
    llf2 = benchmarks.negbin(dummy_data[["y2"]], autoregressive=False)

    assert np.isclose(llf_both, llf1 + llf2)

    # Test AR(1)
    llf_ar1 = benchmarks.negbin(dummy_data[["y1"]], autoregressive=True)
    assert isinstance(llf_ar1, float)
    # Typically AR(1) likelihood should be different from iid
    assert not np.isclose(llf_ar1, llf1)

    llf_both_ar = benchmarks.negbin(dummy_data, autoregressive=True)
    llf2_ar = benchmarks.negbin(dummy_data[["y2"]], autoregressive=True)
    assert np.isclose(llf_both_ar, llf_ar1 + llf2_ar)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_pomp_benchmark_methods(dummy_data):
    pomp = MagicMock(spec=Pomp)
    pomp.ys = dummy_data

    # We monkeypatch the Pomp benchmark method directly by calling the class method
    # Or more robustly, we create a mock that has the mixin logic, or test on a real Pomp if easy.
    # Since Pomp is complex, we'll just test the unbound method on a dummy object
    pomp.arma = Pomp.arma.__get__(pomp)
    pomp.negbin = Pomp.negbin.__get__(pomp)

    llf_arma = pomp.arma(order=(1, 0, 0))
    assert isinstance(llf_arma, float)

    llf_nb = pomp.negbin()
    assert isinstance(llf_nb, float)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_panel_benchmark_methods(dummy_data):
    unit1 = MagicMock()
    unit1.ys = dummy_data[["y1"]]

    unit2 = MagicMock()
    unit2.ys = dummy_data[["y2"]]

    panel = MagicMock(spec=PanelEstimationMixin)
    panel.unit_objects = {"u1": unit1, "u2": unit2}

    panel.arma = PanelEstimationMixin.arma.__get__(panel)
    panel.negbin = PanelEstimationMixin.negbin.__get__(panel)

    df_arma = panel.arma(order=(1, 0, 0))
    assert isinstance(df_arma, pd.DataFrame)
    assert "unit" in df_arma.columns
    assert "logLik" in df_arma.columns
    assert len(df_arma) == 3  # [[TOTAL]], u1, u2

    # Check that [[TOTAL]] is the first row
    assert df_arma.iloc[0]["unit"] == "[[TOTAL]]"

    total_llf_arma = df_arma.iloc[0]["logLik"]
    expected_total_arma = benchmarks.arma(dummy_data, order=(1, 0, 0))
    assert np.isclose(total_llf_arma, expected_total_arma)

    df_nb = panel.negbin()
    assert isinstance(df_nb, pd.DataFrame)
    assert len(df_nb) == 3
    assert df_nb.iloc[0]["unit"] == "[[TOTAL]]"
    total_llf_nb = df_nb.iloc[0]["logLik"]
    expected_total_nb = benchmarks.negbin(dummy_data)
    assert np.isclose(total_llf_nb, expected_total_nb)


def test_missing_statsmodels_raises(monkeypatch, dummy_data):
    import importlib.util

    # Mock find_spec to return None (simulate missing statsmodels)
    def mock_find_spec(name):
        return None

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    with pytest.raises(ImportError, match="statsmodels"):
        benchmarks.arma(dummy_data)

    with pytest.raises(ImportError, match="statsmodels"):
        benchmarks.negbin(dummy_data)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_benchmark_warning_suppression(dummy_data):
    from unittest.mock import patch, MagicMock
    import warnings

    # Patch ARIMA class in the statsmodels module
    with patch("statsmodels.tsa.arima.model.ARIMA") as mock_arima_cls:
        mock_model = mock_arima_cls.return_value
        mock_res = MagicMock()
        mock_res.llf = 1.0
        mock_model.fit.return_value = mock_res

        def side_effect(*args, **kwargs):
            warnings.warn("Fake statsmodels warning", RuntimeWarning)
            return mock_res

        mock_model.fit.side_effect = side_effect

        # Use -1 to ensure only 1 warning is captured in our check
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            benchmarks.arma(dummy_data[["y1"]], suppress_warnings=True)

            assert mock_model.fit.called
            summary_warnings = [
                warn
                for warn in w_list
                if "arma: 1 warnings were produced by statsmodels"
                in str(warn.message)
            ]
            assert len(summary_warnings) == 1

    # Patch NegativeBinomial class in statsmodels.api
    with patch("statsmodels.api.NegativeBinomial") as mock_nb_cls:
        mock_model = mock_nb_cls.return_value
        mock_res = MagicMock()
        mock_res.llf = 1.0
        mock_model.fit.return_value = mock_res

        def side_effect_nb(*args, **kwargs):
            warnings.warn("Fake NB warning", RuntimeWarning)
            return mock_res

        mock_model.fit.side_effect = side_effect_nb

        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            benchmarks.negbin(dummy_data[["y1"]], suppress_warnings=True)

            assert mock_model.fit.called
            summary_warnings = [
                warn
                for warn in w_list
                if "negbin: 1 warnings were produced by statsmodels"
                in str(warn.message)
            ]
            assert len(summary_warnings) == 1


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_panel_benchmark_warning_suppression(dummy_data):
    from unittest.mock import patch, MagicMock
    import warnings

    unit1 = MagicMock()
    unit1.ys = dummy_data[["y1"]]
    unit2 = MagicMock()
    unit2.ys = dummy_data[["y2"]]

    panel = MagicMock(spec=PanelEstimationMixin)
    panel.unit_objects = {"u1": unit1, "u2": unit2}
    panel.arma = PanelEstimationMixin.arma.__get__(panel)
    panel.negbin = PanelEstimationMixin.negbin.__get__(panel)

    # Patch ARIMA class in the statsmodels module
    with patch("statsmodels.tsa.arima.model.ARIMA") as mock_arima_cls:
        mock_model = mock_arima_cls.return_value
        mock_res = MagicMock()
        mock_res.llf = 1.0
        mock_model.fit.return_value = mock_res

        def side_effect(*args, **kwargs):
            warnings.warn("Fake statsmodels warning", RuntimeWarning)
            return mock_res

        mock_model.fit.side_effect = side_effect

        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            panel.arma(suppress_warnings=True)

            # Each unit (u1, u2) produced a warning.
            # Total 2 warnings captured and summarized into ONE warning by PanelPomp
            summary_pattern = (
                "arma: 2 warnings were produced by statsmodels across units"
            )
            summary_warnings = [
                warn for warn in w_list if summary_pattern in str(warn.message)
            ]
            assert len(summary_warnings) == 1

    # Patch NegativeBinomial class in statsmodels.api
    with patch("statsmodels.api.NegativeBinomial") as mock_nb_cls:
        mock_model = mock_nb_cls.return_value
        mock_res = MagicMock()
        mock_res.llf = 1.0
        mock_model.fit.return_value = mock_res

        def side_effect_nb(*args, **kwargs):
            warnings.warn("Fake NB warning", RuntimeWarning)
            return mock_res

        mock_model.fit.side_effect = side_effect_nb

        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            panel.negbin(suppress_warnings=True)

            summary_pattern = (
                "negbin: 2 warnings were produced by statsmodels across units"
            )
            summary_warnings = [
                warn for warn in w_list if summary_pattern in str(warn.message)
            ]
            assert len(summary_warnings) == 1

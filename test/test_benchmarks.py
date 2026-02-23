import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Attempt to import statsmodels for tests
try:
    import statsmodels

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from pypomp.benchmarks import arma_benchmark, negbin_benchmark
from pypomp.pomp_class import Pomp
from pypomp.panelPomp.estimation_mixin import PanelEstimationMixin


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    return pd.DataFrame(
        {"y1": np.random.poisson(5, 50), "y2": np.random.poisson(10, 50)}
    )


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_arma_benchmark_function(dummy_data):
    # Test univariate
    llf1 = arma_benchmark(dummy_data[["y1"]], order=(1, 0, 0))
    assert isinstance(llf1, float)

    # Test multivariate
    llf_both = arma_benchmark(dummy_data, order=(1, 0, 0))
    llf2 = arma_benchmark(dummy_data[["y2"]], order=(1, 0, 0))

    # Should be the sum of individual likelihoods
    assert np.isclose(llf_both, llf1 + llf2)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_negbin_benchmark_function(dummy_data):
    llf1 = negbin_benchmark(dummy_data[["y1"]])
    assert isinstance(llf1, float)

    llf_both = negbin_benchmark(dummy_data)
    llf2 = negbin_benchmark(dummy_data[["y2"]])

    assert np.isclose(llf_both, llf1 + llf2)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_pomp_benchmark_methods(dummy_data):
    pomp = MagicMock(spec=Pomp)
    pomp.ys = dummy_data

    # We monkeypatch the Pomp benchmark method directly by calling the class method
    # Or more robustly, we create a mock that has the mixin logic, or test on a real Pomp if easy.
    # Since Pomp is complex, we'll just test the unbound method on a dummy object
    pomp.arma_benchmark = Pomp.arma_benchmark.__get__(pomp)
    pomp.negbin_benchmark = Pomp.negbin_benchmark.__get__(pomp)

    llf_arma = pomp.arma_benchmark(order=(1, 0, 0))
    assert isinstance(llf_arma, float)

    llf_nb = pomp.negbin_benchmark()
    assert isinstance(llf_nb, float)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
def test_panel_benchmark_methods(dummy_data):
    unit1 = MagicMock()
    unit1.ys = dummy_data[["y1"]]

    unit2 = MagicMock()
    unit2.ys = dummy_data[["y2"]]

    panel = MagicMock(spec=PanelEstimationMixin)
    panel.unit_objects = {"u1": unit1, "u2": unit2}

    panel.arma_benchmark = PanelEstimationMixin.arma_benchmark.__get__(panel)
    panel.negbin_benchmark = PanelEstimationMixin.negbin_benchmark.__get__(panel)

    llf_arma = panel.arma_benchmark(order=(1, 0, 0))
    assert np.isclose(llf_arma, arma_benchmark(dummy_data, order=(1, 0, 0)))

    llf_nb = panel.negbin_benchmark()
    assert np.isclose(llf_nb, negbin_benchmark(dummy_data))


def test_missing_statsmodels_raises(monkeypatch, dummy_data):
    import importlib.util

    # Mock find_spec to return None (simulate missing statsmodels)
    def mock_find_spec(name):
        return None

    monkeypatch.setattr(importlib.util, "find_spec", mock_find_spec)

    with pytest.raises(ImportError, match="statsmodels"):
        arma_benchmark(dummy_data)

    with pytest.raises(ImportError, match="statsmodels"):
        negbin_benchmark(dummy_data)

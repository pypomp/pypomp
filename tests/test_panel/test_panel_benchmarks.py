"""The panel arma/negbin benchmark wrappers.

These live beside the other panel tests because they need the measles panel
fixture from this package conftest.
"""

import numpy as np
import pandas as pd
import pytest


def test_panel_arma_suppress_warnings_toggle(measles_panel_setup_some_shared):
    """The suppress_warnings=True and suppress_warnings=False code paths both
    call benchmarks.arma(..., suppress_warnings=False) under the hood, so
    they should produce identical results and only differ in warning
    handling."""
    pytest.importorskip("statsmodels")
    panel, _, _ = measles_panel_setup_some_shared

    df_suppressed = panel.arma(order=(1, 0, 0), suppress_warnings=True)
    df_raw = panel.arma(order=(1, 0, 0), suppress_warnings=False)

    for df in (df_suppressed, df_raw):
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["unit", "logLik"]
        assert df.iloc[0]["unit"] == "[[TOTAL]]"
        unit_names = set(panel.unit_objects.keys())
        assert set(df["unit"]) == unit_names | {"[[TOTAL]]"}
        total = df.iloc[0]["logLik"]
        per_unit_sum = df.iloc[1:]["logLik"].sum()
        assert np.isclose(total, per_unit_sum)

    pd.testing.assert_frame_equal(
        df_suppressed.reset_index(drop=True), df_raw.reset_index(drop=True)
    )


def test_panel_negbin_suppress_warnings_toggle(measles_panel_setup_some_shared):
    """Same as above but for negbin: both branches should be numerically
    identical since suppress_warnings only changes warning handling."""
    pytest.importorskip("statsmodels")
    panel, _, _ = measles_panel_setup_some_shared

    df_suppressed = panel.negbin(suppress_warnings=True)
    df_raw = panel.negbin(suppress_warnings=False)

    for df in (df_suppressed, df_raw):
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["unit", "logLik"]
        assert df.iloc[0]["unit"] == "[[TOTAL]]"
        total = df.iloc[0]["logLik"]
        per_unit_sum = df.iloc[1:]["logLik"].sum()
        assert np.isclose(total, per_unit_sum)

    pd.testing.assert_frame_equal(
        df_suppressed.reset_index(drop=True), df_raw.reset_index(drop=True)
    )

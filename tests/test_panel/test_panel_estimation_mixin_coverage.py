"""Targeted coverage for miscellaneous branches in PanelEstimationMixin that
don't fit naturally into the other test_panel_*.py files (a helper method,
edge cases of sample_params, and the arma/negbin benchmark wrappers)."""

import jax
import numpy as np
import pandas as pd
import pytest

import pypomp as pp


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

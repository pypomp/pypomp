"""Tests for PanelParameters in pypomp.core.parameters.panel."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pypomp.core.parameters as pp
from pypomp.core.par_trans import ParTrans
from pypomp.core.parameters.panel import _standardize_panel_theta


def test_standardize_panel_theta_validation():
    # theta=None init
    ds, s_names, u_names = _standardize_panel_theta(None)
    assert len(s_names) == 0
    assert len(u_names) == 0

    # non-dict/non-list type
    with pytest.raises(
        TypeError, match="unsupported operand type|object is not iterable"
    ):
        pp.PanelParameters(cast(Any, 123))

    # missing keys in dict
    with pytest.raises(
        ValueError, match="must have exactly the keys 'shared' and 'unit_specific'"
    ):
        _standardize_panel_theta(cast(Any, {"shared": None}))

    # values not None or DataFrame
    with pytest.raises(TypeError, match="must be None or pd.DataFrames"):
        _standardize_panel_theta(cast(Any, {"shared": 123, "unit_specific": None}))

    # consistency checks across replicates (shared parameters)
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    with pytest.raises(
        ValueError, match="Some, but not all, shared parameters are None"
    ):
        _standardize_panel_theta(
            cast(
                list[dict[str, pd.DataFrame | None]],
                [
                    {"shared": shared_df, "unit_specific": None},
                    {"shared": None, "unit_specific": None},
                ],
            )
        )

    # consistency checks across replicates (unit-specific parameters)
    unit_df = pd.DataFrame({"u1": [1.0]}, index=["u_param"])
    with pytest.raises(
        ValueError, match="Some, but not all, unit-specific parameters are None"
    ):
        _standardize_panel_theta(
            cast(
                list[dict[str, pd.DataFrame | None]],
                [
                    {"shared": None, "unit_specific": unit_df},
                    {"shared": None, "unit_specific": None},
                ],
            )
        )

    # Shared DataFrame column count != 1
    shared_df_bad = pd.DataFrame({"s1": [1.0, 2.0], "s2": [3.0, 4.0]})
    with pytest.raises(
        ValueError, match="Shared parameters must have exactly one column"
    ):
        _standardize_panel_theta({"shared": shared_df_bad, "unit_specific": None})

    # Parameter name overlap
    shared_overlap = pd.DataFrame({"shared": [1.0]}, index=["param1"])
    unit_overlap = pd.DataFrame({"u1": [2.0]}, index=["param1"])
    with pytest.raises(
        ValueError, match="Parameter name\\(s\\) found in both shared and unit-specific"
    ):
        _standardize_panel_theta(
            {"shared": shared_overlap, "unit_specific": unit_overlap}
        )

    # Mismatched shared index across replicates
    shared_df1 = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    shared_df2 = pd.DataFrame({"shared": [2.0]}, index=["s2"])
    with pytest.raises(
        ValueError, match="Shared parameter index mismatch at replicate 1"
    ):
        _standardize_panel_theta(
            cast(
                list[dict[str, pd.DataFrame | None]],
                [
                    {"shared": shared_df1, "unit_specific": None},
                    {"shared": shared_df2, "unit_specific": None},
                ],
            )
        )

    # Mismatched unit index across replicates
    unit_df1 = pd.DataFrame({"u1": [1.0]}, index=["up1"])
    unit_df2 = pd.DataFrame({"u1": [2.0]}, index=["up2"])
    with pytest.raises(
        ValueError, match="Unit parameter index mismatch at replicate 1"
    ):
        _standardize_panel_theta(
            cast(
                list[dict[str, pd.DataFrame | None]],
                [
                    {"shared": None, "unit_specific": unit_df1},
                    {"shared": None, "unit_specific": unit_df2},
                ],
            )
        )

    # Mismatched unit columns across replicates
    unit_df_col1 = pd.DataFrame({"u1": [1.0]}, index=["up1"])
    unit_df_col2 = pd.DataFrame({"u2": [2.0]}, index=["up1"])
    with pytest.raises(ValueError, match="Unit columns mismatch at replicate 1"):
        _standardize_panel_theta(
            cast(
                list[dict[str, pd.DataFrame | None]],
                [
                    {"shared": None, "unit_specific": unit_df_col1},
                    {"shared": None, "unit_specific": unit_df_col2},
                ],
            )
        )


def test_panel_parameters_init_xr_dataset():
    # Construct xr.Dataset without attrs
    shared_da = xr.DataArray(
        [[1.0]],
        dims=["theta_idx", "parameter"],
        coords={"theta_idx": [0], "parameter": ["s1"]},
    )
    ds = xr.Dataset(data_vars={"shared": shared_da})
    panel = pp.PanelParameters(ds)
    assert panel.get_shared_param_names() == ["s1"]
    assert panel.get_unit_param_names() == []

    # Construct using existing PanelParameters
    panel_copy = pp.PanelParameters(panel)
    assert panel_copy == panel

    # Construct using existing PanelParameters and override logLik_unit
    panel_copy2 = pp.PanelParameters(panel, logLik_unit=np.array([4.0]))
    assert np.allclose(panel_copy2.logLik_unit, [[4.0]])

    # Set params using xr.Dataset
    panel.set_params(ds)
    assert panel.get_shared_param_names() == ["s1"]

    # Set params using None error
    with pytest.raises(ValueError, match="theta cannot be None"):
        panel.set_params(cast(Any, None))


def test_panel_parameters_log_lik_unit_format():
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    unit_df = pd.DataFrame({"u1": [2.0], "u2": [3.0]}, index=["up1"])

    # n_reps = 1, 1D logLik_unit input (should reshape to (1, n_units))
    panel = pp.PanelParameters(
        theta=cast(Any, {"shared": shared_df, "unit_specific": unit_df}),
        logLik_unit=np.array([1.5, 2.5]),
    )
    assert panel.logLik_unit.shape == (1, 2)
    assert np.allclose(panel.logLik_unit, [[1.5, 2.5]])
    assert np.allclose(panel.logLik, [4.0])

    # Shape mismatch error (use 2D array to trigger since 1D of size 1 is reshaped to (1,1))
    with pytest.raises(ValueError, match="logLik_unit shape mismatch"):
        pp.PanelParameters(
            theta=cast(Any, {"shared": shared_df, "unit_specific": unit_df}),
            logLik_unit=np.array([[1.5]]),
        )

    # logLik setter raises AttributeError
    with pytest.raises(AttributeError, match="Cannot set logLik directly"):
        panel.logLik = np.array([1.0])

    # logLik_unit setter
    panel.logLik_unit = np.array([[2.0, 3.0]])
    assert np.allclose(panel.logLik_unit, [[2.0, 3.0]])
    assert np.allclose(panel.logLik, [5.0])

    # Empty logLik unit check (n_units = 0, logLik_unit is empty)
    panel_shared_only = pp.PanelParameters(
        theta={"shared": shared_df, "unit_specific": None}, logLik_unit=np.array([])
    )
    assert panel_shared_only.logLik_unit.shape == (1, 0)


def test_panel_parameters_to_jax_array_edge_cases():
    # reps = 0
    panel_empty = pp.PanelParameters(None)
    assert panel_empty.to_jax_array().shape == (0, 0, 0)

    # unit_names is None and no unit specific parameters error
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    panel_shared_only = pp.PanelParameters(
        theta={"shared": shared_df, "unit_specific": None}
    )
    with pytest.raises(
        ValueError, match="unit_names required when no unit_specific parameters exist"
    ):
        panel_shared_only.to_jax_array()

    # unknown parameter name check
    with pytest.raises(KeyError, match="Parameter 'nonexistent' not found"):
        panel_shared_only.to_jax_array(param_names=["nonexistent"], unit_names=["u1"])

    # unknown unit name check
    unit_df = pd.DataFrame({"u1": [2.0]}, index=["up1"])
    panel = pp.PanelParameters(theta={"shared": None, "unit_specific": unit_df})
    with pytest.raises(KeyError, match="Unit mismatch for parameter"):
        panel.to_jax_array(param_names=["up1"], unit_names=["u2"])


def test_panel_parameters_mix_and_match():
    # mixed_and_matched on reps=0 does nothing (returns copy)
    panel_empty_orig = pp.PanelParameters(None)
    panel_empty = panel_empty_orig.mixed_and_matched()
    assert len(panel_empty_orig) == 0
    assert len(panel_empty) == 0

    # normal mixed_and_matched sorting check
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    unit_df = pd.DataFrame({"u1": [2.0], "u2": [3.0]}, index=["up1"])
    panel_orig = pp.PanelParameters(
        theta=cast(Any, [{"shared": shared_df, "unit_specific": unit_df}] * 3),
        logLik_unit=np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]]),
    )

    panel = panel_orig.mixed_and_matched()
    # original remains unmodified:
    assert panel_orig.logLik_unit[0, 0] == 1.0
    expected_u1_ll = [3.0, 2.0, 1.0]
    expected_u2_ll = [5.0, 4.0, 2.0]
    np.testing.assert_allclose(panel.logLik_unit[:, 0], expected_u1_ll)
    np.testing.assert_allclose(panel.logLik_unit[:, 1], expected_u2_ll)


def test_panel_parameters_eq_logLik_names_mismatch():
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    unit_df = pd.DataFrame({"u1": [2.0]}, index=["up1"])
    p1 = pp.PanelParameters(theta={"shared": shared_df, "unit_specific": unit_df})

    # Shared names mismatch
    shared_df2 = pd.DataFrame({"shared": [1.0]}, index=["s2"])
    p2 = pp.PanelParameters(theta={"shared": shared_df2, "unit_specific": unit_df})
    assert p1 != p2
    assert p1._eq_logLik(p2) is False

    # Unit specific names mismatch
    unit_df2 = pd.DataFrame({"u1": [2.0]}, index=["up2"])
    p3 = pp.PanelParameters(theta={"shared": shared_df, "unit_specific": unit_df2})
    assert p1 != p3
    assert p1._eq_logLik(p3) is False


def test_panel_parameters_utility_and_magic():
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    unit_df = pd.DataFrame({"u1": [2.0]}, index=["up1"])
    panel = pp.PanelParameters(theta={"shared": shared_df, "unit_specific": unit_df})

    # subset calling with int
    sub = panel.subset(0)
    assert len(sub) == 1

    # list call / iteration / params extraction
    assert isinstance(list(panel), list)
    assert isinstance(panel.params(), xr.Dataset)
    assert isinstance(panel.params(as_list=True), list)
    assert isinstance(panel.params(as_list=False), xr.Dataset)

    # _getitem_int / indexing (with proper Pandas DataFrame comparisons)
    d1 = panel[0]
    d2 = list(panel)[0]
    pd.testing.assert_frame_equal(d1["shared"], d2["shared"])
    pd.testing.assert_frame_equal(d1["unit_specific"], d2["unit_specific"])

    # multiplication with zero units (replicated_logLik empty path)
    panel_shared_only = pp.PanelParameters(
        theta={"shared": shared_df, "unit_specific": None}
    )
    res_mul = panel_shared_only * 2
    assert len(res_mul) == 2

    # transform logic
    def to_est_fn(theta: Any) -> Any:
        return dict(theta)

    def from_est_fn(theta: Any) -> Any:
        return dict(theta)

    p_trans = ParTrans(to_est=to_est_fn, from_est=from_est_fn)
    panel = panel.transformed(p_trans)

    # set_params using dict value to trigger 556-558 coverage
    panel.set_params({"shared": shared_df, "unit_specific": unit_df})

    # list conversion with reps=0 to trigger 570 coverage
    panel_empty = pp.PanelParameters(None)
    assert len(panel_empty._to_list()) == 0

    # _to_list with shared=None and unit_specific=None to trigger 590 and 602 coverage
    panel_specific_only = pp.PanelParameters(
        theta={"shared": None, "unit_specific": unit_df}
    )
    assert list(panel_specific_only)[0]["shared"] is None

    panel_shared_only_2 = pp.PanelParameters(
        theta={"shared": shared_df, "unit_specific": None}
    )
    assert list(panel_shared_only_2)[0]["unit_specific"] is None


def test_panel_parameters_merge_validation():
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    unit_df = pd.DataFrame({"u1": [2.0]}, index=["up1"])
    p1 = pp.PanelParameters(
        theta={"shared": shared_df, "unit_specific": unit_df},
        logLik_unit=np.array([[1.0]]),
    )

    # Empty merge error
    with pytest.raises(
        ValueError, match="At least one PanelParameters object must be provided."
    ):
        pp.PanelParameters.merge()

    # Non-PanelParameters type error
    with pytest.raises(
        TypeError, match="All merged objects must be of type PanelParameters."
    ):
        pp.PanelParameters.merge(p1, cast(Any, "not_panel"))

    # Shared names mismatch
    shared_df2 = pd.DataFrame({"shared": [1.0]}, index=["s2"])
    p2 = pp.PanelParameters(theta={"shared": shared_df2, "unit_specific": unit_df})
    with pytest.raises(ValueError, match="same canonical shared parameter names"):
        pp.PanelParameters.merge(p1, p2)

    # Unit specific names mismatch
    unit_df2 = pd.DataFrame({"u1": [2.0]}, index=["up2"])
    p3 = pp.PanelParameters(theta={"shared": shared_df, "unit_specific": unit_df2})
    with pytest.raises(ValueError, match="same canonical unit parameter names"):
        pp.PanelParameters.merge(p1, p3)

    # Scale mismatch
    p4 = pp.PanelParameters(
        theta={"shared": shared_df, "unit_specific": unit_df}, estimation_scale=True
    )
    with pytest.raises(ValueError, match="same estimation scale"):
        pp.PanelParameters.merge(p1, p4)

    # Unit names mismatch
    unit_df3 = pd.DataFrame({"u2": [2.0]}, index=["up1"])
    p5 = pp.PanelParameters(theta={"shared": shared_df, "unit_specific": unit_df3})
    with pytest.raises(ValueError, match="same unit names"):
        pp.PanelParameters.merge(p1, p5)

    # Normal merge
    p6 = pp.PanelParameters(
        theta={"shared": shared_df, "unit_specific": unit_df},
        logLik_unit=np.array([[2.0]]),
    )
    merged = pp.PanelParameters.merge(p1, p6)
    assert len(merged) == 2
    np.testing.assert_allclose(merged.logLik_unit, [[1.0], [2.0]])

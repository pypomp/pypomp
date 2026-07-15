import copy
from typing import Any, cast
import pytest
import numpy as np
import pandas as pd
import xarray as xr

import pypomp.core.parameters as pp
from pypomp.core.par_trans import ParTrans
from pypomp.core.parameters.pomp import _standardize_pomp_theta
from pypomp.core.parameters.panel import _standardize_panel_theta


# =====================================================================
# 1. Base Class (base.py) Test Cases
# =====================================================================


def test_base_parameter_set_methods():
    # Setup standard PompParameters for testing base methods
    pomp = pp.PompParameters({"a": 1.0, "b": 2.0}, logLik=np.array(1.5))

    # num_params and num_replicates
    assert pomp.num_params() == 2
    assert pomp.num_replicates() == 1

    # __iter__
    assert list(pomp) == [{"a": 1.0, "b": 2.0}]

    # __copy__ & __deepcopy__
    pomp_copy = copy.copy(pomp)
    assert pomp_copy == pomp
    assert pomp_copy._data is not pomp._data

    pomp_deepcopy = copy.deepcopy(pomp)
    assert pomp_deepcopy == pomp
    assert pomp_deepcopy._data is not pomp._data

    # __mul__ and __rmul__ validation
    assert 2 * pomp == pomp * 2
    assert len(pomp * 3) == 3

    with pytest.raises(TypeError):
        _ = pomp * cast(int, 2.5)
    with pytest.raises(ValueError, match="non-negative"):
        _ = pomp * -1
    with pytest.raises(ValueError, match="empty ParameterSet"):
        _ = pomp * 0

    # __repr__ & __str__
    assert "PompParameters" in repr(pomp)
    assert "PompParameters" in str(pomp)

    # __eq__ mismatch cases
    # Different types
    assert pomp != "not a parameter set"
    # Different estimation scales
    pomp_est = pp.PompParameters({"a": 1.0, "b": 2.0}, estimation_scale=True)
    assert pomp != pomp_est
    # Different param names
    pomp_diff_names = pp.PompParameters({"a": 1.0, "c": 2.0})
    assert pomp != pomp_diff_names
    # Different values
    pomp_diff_vals = pp.PompParameters({"a": 1.0, "b": 3.0})
    assert pomp != pomp_diff_vals

    # __getitem__ variations
    pomp_multi = pomp * 3
    # Slice
    assert isinstance(pomp_multi[0:2], pp.PompParameters)
    assert len(pomp_multi[0:2]) == 2
    # List
    assert isinstance(pomp_multi[[0, 2]], pp.PompParameters)
    assert len(pomp_multi[[0, 2]]) == 2
    # Numpy array
    assert isinstance(pomp_multi[cast(Any, np.array([1, 2]))], pp.PompParameters)
    assert len(pomp_multi[cast(Any, np.array([1, 2]))]) == 2
    # Single integer index
    assert pomp_multi[1] == {"a": 1.0, "b": 2.0}

    # Call base abstract methods directly via super / ParameterSet to cover their pass statements
    assert pp.ParameterSet.to_jax_array(pomp) is None  # type: ignore
    pp.ParameterSet.set_params(pomp, None)  # type: ignore
    pp.ParameterSet.logLik.__get__(pomp)  # type: ignore
    pp.ParameterSet._to_list(pomp)  # type: ignore
    pp.ParameterSet.subset(pomp, 0)  # type: ignore
    pp.ParameterSet._replicated_logLik(pomp, 1)  # type: ignore
    pp.ParameterSet._slice_logLik(pomp, np.array([0]))  # type: ignore
    pp.ParameterSet._eq_logLik(pomp, pomp)  # type: ignore
    pp.ParameterSet._getitem_int(pomp, 0)  # type: ignore
    pp.ParameterSet._transform_and_load(pomp, None, [], "to_est")  # type: ignore


def test_base_parameter_set_prune():
    # PompParameters empty reps pruned error
    pomp_empty = pp.PompParameters(None)
    with pytest.raises(ValueError, match="No parameter sets available to prune."):
        pomp_empty.pruned(1)

    # PompParameters n < 1 error
    pomp = pp.PompParameters({"a": 1.0}, logLik=np.array(1.5))
    with pytest.raises(ValueError, match="n must be at least 1."):
        pomp.pruned(0)

    # PompParameters all nan logLik error
    pomp_nan_lik = pp.PompParameters({"a": 1.0})
    with pytest.raises(ValueError, match="No valid log-likelihoods available to prune"):
        pomp_nan_lik.pruned(1)

    # Normal pruned for PompParameters (non-mutating)
    pomp_multi_orig = pp.PompParameters(
        [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}], logLik=np.array([1.0, 3.0, 2.0])
    )
    # n=2, refill=True
    pomp_multi = pomp_multi_orig.pruned(n=2, refill=True)
    assert len(pomp_multi_orig) == 3
    assert list(pomp_multi_orig.logLik) == [1.0, 3.0, 2.0]
    assert len(pomp_multi) == 3
    # The top two elements are 2.0 and 3.0. With refill, they repeat to fill 3 elements: [2.0, 3.0, 2.0]
    assert list(pomp_multi.logLik) == [3.0, 2.0, 3.0]

    # n=2, refill=False
    pomp_multi2_orig = pp.PompParameters(
        [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}], logLik=np.array([1.0, 3.0, 2.0])
    )
    pomp_multi2 = pomp_multi2_orig.pruned(n=2, refill=False)
    assert len(pomp_multi2_orig) == 3
    assert len(pomp_multi2) == 2
    assert list(pomp_multi2.logLik) == [3.0, 2.0]

    # PanelParameters pruned with nan logLik (should fallback to zeros and not raise error)
    # Give it a unit-specific parameter so n_units > 0 and logLik is [nan, nan, nan]
    unit_df = pd.DataFrame({"u1": [1.0]}, index=pd.Index(["up1"]))
    panel_orig = pp.PanelParameters(
        theta=[{"shared": None, "unit_specific": unit_df}] * 3
    )
    # logLik_unit is all NaN initially. Pruning should set logLik to zero and proceed.
    panel = panel_orig.pruned(n=2, refill=False)
    assert len(panel_orig) == 3
    assert len(panel) == 2
    assert np.all(np.isnan(panel.logLik))


def test_base_parameter_set_transform():
    # Test simple transform round-trip
    def to_est_fn(theta: Any) -> Any:
        t = dict(theta)
        if "a" in t:
            t["a"] = np.log(t["a"])
        return t

    def from_est_fn(theta: Any) -> Any:
        t = dict(theta)
        if "a" in t:
            t["a"] = np.exp(t["a"])
        return t

    p_trans = ParTrans(to_est=to_est_fn, from_est=from_est_fn)

    # pomp natural scale
    pomp = pp.PompParameters({"a": 2.0, "b": 5.0})
    assert pomp.estimation_scale is False

    assert isinstance(pomp.params(), xr.DataArray)
    assert isinstance(pomp.params(as_list=False), xr.DataArray)
    assert isinstance(pomp.params(as_list=True), list)

    # Auto transformed (direction is None) -> will transform from natural to estimation scale
    pomp_transformed = pomp.transformed(p_trans)
    assert pomp.estimation_scale is False  # original remains unmodified
    assert pomp_transformed.estimation_scale is True
    assert np.allclose(pomp_transformed[0]["a"], np.log(2.0))
    assert np.allclose(pomp_transformed[0]["b"], 5.0)

    # Transform back using explicit direction
    pomp_back = pomp_transformed.transformed(p_trans, direction="from_est")
    assert pomp_transformed.estimation_scale is True  # remains unmodified
    assert pomp_back.estimation_scale is False
    assert np.allclose(pomp_back[0]["a"], 2.0)
    assert np.allclose(pomp_back[0]["b"], 5.0)

    # Transform to_est again explicitly
    pomp_to_est = pomp_back.transformed(p_trans, direction="to_est")
    assert pomp_to_est.estimation_scale is True
    assert np.allclose(pomp_to_est[0]["a"], np.log(2.0))


# =====================================================================
# 2. PompParameters (pomp.py) Test Cases
# =====================================================================


def test_standardize_pomp_theta_validation():
    # Call _standardize_pomp_theta directly to cover early exits and errors
    da = xr.DataArray([[[1.0]]], dims=["theta_idx", "unit", "parameter"])
    assert _standardize_pomp_theta(da) is da

    with pytest.raises(ValueError, match="theta cannot be None"):
        _standardize_pomp_theta(None)

    # theta=None error in set_params
    with pytest.raises(ValueError, match="theta cannot be None"):
        pp.PompParameters(None).set_params(cast(Any, None))

    # invalid type
    with pytest.raises(TypeError, match="theta must be a Mapping, Sequence"):
        pp.PompParameters(cast(Any, 123))

    # empty list
    with pytest.raises(ValueError, match="theta cannot be empty"):
        pp.PompParameters([])

    # sequence of non-mappings (raises TypeError)
    with pytest.raises(TypeError):
        pp.PompParameters(cast(Any, [{"a": 1}, 123]))

    # dict with bool
    with pytest.raises(TypeError, match="is not a float: got bool"):
        pp.PompParameters(cast(Any, {"a": True}))

    # dict with non-numeric (e.g. str)
    with pytest.raises(TypeError, match="is not a float: got str"):
        pp.PompParameters(cast(Any, {"a": "hello"}))

    # inconsistent keys
    with pytest.raises(ValueError, match="different keys than the first set"):
        pp.PompParameters([{"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 3.0}])


def test_pomp_parameters_init_dataarray():
    # 1D DataArray with unnamed dimension (should auto-rename to parameter)
    da_1d_unnamed = xr.DataArray([1.0, 2.0])
    p = pp.PompParameters(da_1d_unnamed)
    assert "parameter" in p._data.dims

    # 2D DataArray without parameter dimension error
    da_2d_bad = xr.DataArray([[1.0, 2.0]], dims=["theta_idx", "not_parameter"])
    with pytest.raises(
        ValueError, match="2D DataArray must have 'parameter' dimension"
    ):
        pp.PompParameters(da_2d_bad)

    # 2D DataArray without theta_idx (should rename other dimension)
    da_2d_rename = xr.DataArray([[1.0, 2.0]], dims=["other", "parameter"])
    p_rename = pp.PompParameters(da_2d_rename)
    assert p_rename._data.dims == ("theta_idx", "unit", "parameter")

    # 3D DataArray check
    da_3d = xr.DataArray([[[1.0], [2.0]]], dims=["theta_idx", "unit", "parameter"])
    p_3d = pp.PompParameters(da_3d)
    assert p_3d.num_replicates() == 1

    # 3D DataArray with different coordinates transpose path
    da_3d_other = xr.DataArray(
        np.ones((1, 1, 1)), dims=["parameter", "unit", "theta_idx"]
    )
    p_3d_other = pp.PompParameters(da_3d_other)
    assert p_3d_other._data.dims == ("theta_idx", "unit", "parameter")

    # 3D DataArray with bad dimensions (line 183 coverage) raises KeyError on sizes
    da_3d_bad_dims = xr.DataArray(np.ones((1, 1, 1)), dims=["x", "y", "z"])
    with pytest.raises(KeyError):
        pp.PompParameters(da_3d_bad_dims)

    # 4D DataArray error
    da_4d = xr.DataArray(np.ones((1, 1, 1, 1)))
    with pytest.raises(ValueError, match="DataArray must be 1D, 2D, or 3D"):
        pp.PompParameters(da_4d)


def test_pomp_parameters_log_lik_format():
    # logLik scalar broadcasting
    p = pp.PompParameters([{"a": 1.0}, {"a": 2.0}], logLik=np.array(5.0))
    assert np.allclose(p.logLik, [5.0, 5.0])

    # logLik length mismatch
    with pytest.raises(ValueError, match="Length of logLik"):
        pp.PompParameters([{"a": 1.0}, {"a": 2.0}], logLik=np.array([1.0]))

    # logLik setter
    p.logLik = np.array(10.0)
    assert np.allclose(p.logLik, [10.0, 10.0])


def test_pomp_parameters_to_jax_array():
    p = pp.PompParameters({"a": 1.0, "b": 2.0})
    # missing parameter
    with pytest.raises(KeyError, match="expected by model but missing"):
        p.to_jax_array(["a", "c"])


def test_pomp_parameters_subset_and_copy():
    p = pp.PompParameters([{"a": 1.0}, {"a": 2.0}])
    # Call subset with int explicitly
    sub = p.subset(1)
    assert len(sub) == 1
    assert sub[0] == {"a": 2.0}

    # Copy constructors with logLik
    p_copy1 = pp.PompParameters(p)
    assert p_copy1 == p
    p_copy2 = pp.PompParameters(p, logLik=np.array([5.0, 6.0]))
    assert np.allclose(p_copy2.logLik, [5.0, 6.0])

    # set_params test
    p.set_params({"a": 3.0})
    assert p[0] == {"a": 3.0}


def test_pomp_parameters_merge():
    p1 = pp.PompParameters({"a": 1.0, "b": 2.0}, logLik=np.array(1.0))
    p2 = pp.PompParameters({"a": 3.0, "b": 4.0}, logLik=np.array(2.0))

    # empty merge error
    with pytest.raises(
        ValueError, match="At least one PompParameters object must be provided."
    ):
        pp.PompParameters.merge()

    # invalid type in merge
    with pytest.raises(
        TypeError, match="All merged objects must be of type PompParameters."
    ):
        pp.PompParameters.merge(p1, cast(Any, "not_pomp"))

    # parameter names mismatch in merge
    p_diff_names = pp.PompParameters({"a": 1.0, "c": 3.0})
    with pytest.raises(
        ValueError, match="must have the same canonical parameter names"
    ):
        pp.PompParameters.merge(p1, p_diff_names)

    # scale mismatch in merge
    p_diff_scale = pp.PompParameters({"a": 1.0, "b": 2.0}, estimation_scale=True)
    with pytest.raises(ValueError, match="must have the same estimation scale"):
        pp.PompParameters.merge(p1, p_diff_scale)

    # normal merge
    merged = pp.PompParameters.merge(p1, p2)
    assert len(merged) == 2
    assert list(merged.logLik) == [1.0, 2.0]
    assert merged[0] == {"a": 1.0, "b": 2.0}
    assert merged[1] == {"a": 3.0, "b": 4.0}


# =====================================================================
# 3. PanelParameters (panel.py) Test Cases
# =====================================================================


def test_standardize_panel_theta_validation():
    # theta=None init
    ds, s_names, u_names = _standardize_panel_theta(None)
    assert len(s_names) == 0
    assert len(u_names) == 0

    # non-dict/non-list type
    with pytest.raises(TypeError):
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

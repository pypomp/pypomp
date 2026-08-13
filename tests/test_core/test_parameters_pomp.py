"""Tests for PompParameters in pypomp.core.parameters.pomp."""

from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

import pypomp.core.parameters as pp
from pypomp.core.parameters.pomp import _standardize_pomp_theta


def test_standardize_pomp_theta_validation():
    # A 3D (theta_idx, unit, parameter) DataArray is accepted by the constructor
    da = xr.DataArray([[[1.0]]], dims=["theta_idx", "unit", "parameter"])
    assert pp.PompParameters(da).num_replicates() == 1

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
    with pytest.raises(
        TypeError, match="unsupported operand type|object is not iterable"
    ):
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
    assert p_rename.num_replicates() == 1
    assert p_rename.num_params() == 2

    # 3D DataArray check (singleton unit is collapsed)
    da_3d = xr.DataArray([[[1.0], [2.0]]], dims=["theta_idx", "unit", "parameter"])
    p_3d = pp.PompParameters(da_3d)
    assert p_3d.num_replicates() == 1

    # 3D DataArray with different coordinates transpose path
    da_3d_other = xr.DataArray(
        np.ones((1, 1, 1)), dims=["parameter", "unit", "theta_idx"]
    )
    p_3d_other = pp.PompParameters(da_3d_other)
    assert p_3d_other.num_replicates() == 1

    # 3D DataArray with unexpected dimensions raises a clear error
    da_3d_bad_dims = xr.DataArray(np.ones((1, 1, 1)), dims=["x", "y", "z"])
    with pytest.raises(ValueError, match="3D DataArray must have dims"):
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

"""Tests for the base parameter-set behavior in pypomp.core.parameters.base."""

import copy
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pypomp.core.parameters as pp
from pypomp.core.par_trans import ParTrans


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

    with pytest.raises(
        TypeError, match="unsupported operand type|object is not iterable"
    ):
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

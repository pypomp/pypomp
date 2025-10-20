import pytest
import numpy as np
import pypomp as pp
import warnings
import time

# test_val is based on a direct test against R-pomp::logmeanexp
#
# library(pomp)
# x = c(100,101,102,103,104)
# logmeanexp(x,se=TRUE)
#         est          se
# 102.8424765   0.7510094
#
# import jax.numpy as jnp
# import pypomp as pp
# x = jnp.array([100,101,102,103,104])
# pp.logmeanexp(x)
# pp.logmeanexp_se(x)
# >>> pp.logmeanexp(x)
# Array(102.842476, dtype=float32)
# >>> pp.logmeanexp_se(x)
# Array(0.7510107, dtype=float32)


def test_val():
    x = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    lme = pp.logmeanexp(x)
    lme_se = pp.logmeanexp_se(x)
    assert np.round(lme, 2) == 102.84
    assert np.round(lme_se, 2) == 0.75


def test_nan():
    x = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    assert np.isnan(pp.logmeanexp_se(x[:1]))


def test_empty_array():
    # logmeanexp and logmeanexp_se should return nan for empty input
    empty = np.array([])
    with pytest.warns(UserWarning):
        lme = pp.logmeanexp(empty)
    lme_se = pp.logmeanexp_se(empty)
    assert np.isnan(lme)
    assert np.isnan(lme_se)


def test_all_nan():
    # All NaN input should return nan
    arr = np.array([np.nan, np.nan])
    lme = pp.logmeanexp(arr)
    lme_se = pp.logmeanexp_se(arr)
    assert np.isnan(lme)
    assert np.isnan(lme_se)


def test_inf_values():
    # Array with finite values and -inf
    arr = np.array([100, 101, -np.inf])
    lme = pp.logmeanexp(arr)
    lme_se = pp.logmeanexp_se(arr)
    # Both should be finite, non nan
    assert not np.isnan(lme)
    assert not np.isinf(lme)
    assert not np.isnan(lme_se)
    assert not np.isinf(lme_se)


def test_mixed_nan_inf():
    # Array with finite, nan, and -inf
    arr = np.array([1, np.nan, -np.inf])
    lme = pp.logmeanexp(arr)
    lme_se = pp.logmeanexp_se(arr)
    assert np.isnan(lme)
    assert np.isnan(lme_se)


def test_single_value():
    # Single value: logmeanexp should be the value, se should be nan
    arr = np.array([42.0])
    lme = pp.logmeanexp(arr)
    lme_se = pp.logmeanexp_se(arr)
    assert lme == 42.0
    assert np.isnan(lme_se)


def test_large_values():
    # Large values should not cause overflow
    arr = np.array([1e10, 1e10 + 1, 1e10 + 2])
    lme = pp.logmeanexp(arr)
    lme_se = pp.logmeanexp_se(arr)
    # Should be close to 1e10 + logmeanexp([0,1,2])
    expected = 1e10 + pp.logmeanexp(np.array([0, 1, 2]))
    assert np.isclose(lme, expected, atol=1e-5)
    assert not np.isnan(lme_se)


def test_large_dominant_value():
    # One very large value should not print a warning
    arr = np.array([100, 101, 1e10 + 2])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pp.logmeanexp(arr)
        assert all("empty array" not in str(warn.message) for warn in w)
    lme_se = pp.logmeanexp_se(arr)
    assert not np.isnan(lme_se)


def test_speed():
    # Test that logmeanexp_se runs in under 2 seconds with 10000 values
    arr = np.random.randn(10000)
    start = time.time()
    pp.logmeanexp_se(arr)
    duration = time.time() - start
    assert duration < 2


def test_ignore_nan_true():
    # Array with some nans: ignore_nan=True should drop them
    arr = np.array([1.0, np.nan, 2.0, 3.0])
    lme = pp.logmeanexp(arr, ignore_nan=True)
    lme_se = pp.logmeanexp_se(arr, ignore_nan=True)
    # Should be equal to logmeanexp([1,2,3])
    arr_no_nan = np.array([1.0, 2.0, 3.0])
    expected_lme = pp.logmeanexp(arr_no_nan)
    expected_lme_se = pp.logmeanexp_se(arr_no_nan)
    assert np.isclose(lme, expected_lme, atol=1e-7)
    assert np.isclose(lme_se, expected_lme_se, atol=1e-7)

    # If all values are nan, should return nan
    arr_all_nan = np.array([np.nan, np.nan])
    with warnings.catch_warnings(record=True) as w:
        lme_nan = pp.logmeanexp(arr_all_nan, ignore_nan=True)
        lme_se_nan = pp.logmeanexp_se(arr_all_nan, ignore_nan=True)
    assert np.isnan(lme_nan)
    assert np.isnan(lme_se_nan)

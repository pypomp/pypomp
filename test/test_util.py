import unittest
import numpy as np
import pypomp as pp
import pytest
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


class TestUtil(unittest.TestCase):
    def setUp(self):
        self.logmeanexp = pp.logmeanexp
        self.logmeanexp_se = pp.logmeanexp_se
        self.x = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

    def test_val(self):
        lme = self.logmeanexp(self.x)
        lme_se = self.logmeanexp_se(self.x)
        self.assertEqual(np.round(lme, 2), 102.84)
        self.assertEqual(np.round(lme_se, 2), 0.75)

    def test_nan(self):
        self.assertTrue(np.isnan(self.logmeanexp_se(self.x[:1])))

    def test_empty_array(self):
        # logmeanexp and logmeanexp_se should return nan for empty input
        empty = np.array([])
        with pytest.warns(UserWarning):
            lme = self.logmeanexp(empty)
        lme_se = self.logmeanexp_se(empty)
        self.assertTrue(np.isnan(lme))
        self.assertTrue(np.isnan(lme_se))

    def test_all_nan(self):
        # All NaN input should return nan
        arr = np.array([np.nan, np.nan])
        lme = self.logmeanexp(arr)
        lme_se = self.logmeanexp_se(arr)
        self.assertTrue(np.isnan(lme))
        self.assertTrue(np.isnan(lme_se))

    def test_inf_values(self):
        # Array with finite values and -inf
        arr = np.array([100, 101, -np.inf])
        lme = self.logmeanexp(arr)
        lme_se = self.logmeanexp_se(arr)
        # Both should be finite, non nan
        self.assertFalse(np.isnan(lme))
        self.assertFalse(np.isinf(lme))
        self.assertFalse(np.isnan(lme_se))
        self.assertFalse(np.isinf(lme_se))

    def test_mixed_nan_inf(self):
        # Array with finite, nan, and -inf
        arr = np.array([1, np.nan, -np.inf])
        lme = self.logmeanexp(arr)
        lme_se = self.logmeanexp_se(arr)
        self.assertTrue(np.isnan(lme))
        self.assertTrue(np.isnan(lme_se))

    def test_single_value(self):
        # Single value: logmeanexp should be the value, se should be nan
        arr = np.array([42.0])
        lme = self.logmeanexp(arr)
        lme_se = self.logmeanexp_se(arr)
        self.assertEqual(lme, 42.0)
        self.assertTrue(np.isnan(lme_se))

    def test_large_values(self):
        # Large values should not cause overflow
        arr = np.array([1e10, 1e10 + 1, 1e10 + 2])
        lme = self.logmeanexp(arr)
        lme_se = self.logmeanexp_se(arr)
        # Should be close to 1e10 + logmeanexp([0,1,2])
        expected = 1e10 + self.logmeanexp(np.array([0, 1, 2]))
        self.assertAlmostEqual(lme, expected, places=5)
        self.assertFalse(np.isnan(lme_se))

    def test_large_dominant_value(self):
        # One very large value should not print a warning
        arr = np.array([100, 101, 1e10 + 2])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.logmeanexp(arr)
            self.assertTrue(all("empty array" not in str(warn.message) for warn in w))
        lme_se = self.logmeanexp_se(arr)
        self.assertFalse(np.isnan(lme_se))

    def test_speed(self):
        # Test that logmeanexp_se runs in under 1 second with 10000 values
        arr = np.random.randn(10000)
        start = time.time()
        self.logmeanexp_se(arr)
        duration = time.time() - start
        self.assertTrue(duration < 1)

    def test_ignore_nan_true(self):
        # Array with some nans: ignore_nan=True should drop them
        arr = np.array([1.0, np.nan, 2.0, 3.0])
        lme = self.logmeanexp(arr, ignore_nan=True)
        lme_se = self.logmeanexp_se(arr, ignore_nan=True)
        # Should be equal to logmeanexp([1,2,3])
        arr_no_nan = np.array([1.0, 2.0, 3.0])
        expected_lme = self.logmeanexp(arr_no_nan)
        expected_lme_se = self.logmeanexp_se(arr_no_nan)
        self.assertAlmostEqual(lme, expected_lme, places=7)
        self.assertAlmostEqual(lme_se, expected_lme_se, places=7)

        # If all values are nan, should return nan
        arr_all_nan = np.array([np.nan, np.nan])
        with warnings.catch_warnings(record=True) as w:
            lme_nan = self.logmeanexp(arr_all_nan, ignore_nan=True)
            lme_se_nan = self.logmeanexp_se(arr_all_nan, ignore_nan=True)
        self.assertTrue(np.isnan(lme_nan))
        self.assertTrue(np.isnan(lme_se_nan))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import *
from pypomp.pfilter_pf import pfilter_pf

LG_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = LG()

class TestPfilterPf_LG(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(111)
        self.J = 5
        self.ys = ys
        self.theta = theta
        self.covars = covars
        
        self.rinit = rinit
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures

    def test_internal_basic(self):
        val1 = pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                          ys=self.ys, covars=self.covars, thresh=10, key=self.key)
        val2 = pfilter_pf(rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                          ys=self.ys, key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)

    def test_class_basic(self):
        val1 = pfilter_pf(LG_obj, self.J, thresh=10, key=self.key)
        val2 = pfilter_pf(LG_obj, key=self.key)
        val3 = pfilter_pf(LG_obj, self.J, self.rinit, self.rprocess, self.dmeasure, theta=[], ys=[], key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)
        self.assertEqual(val3.shape, ())
        self.assertTrue(jnp.isfinite(val3.item()))
        self.assertEqual(val3.dtype, jnp.float32)

    def test_invalid_input(self):
        with self.assertRaises(ValueError) as text:
            pfilter_pf(key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, thresh=-1, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, theta=self.theta, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, ys=self.ys, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

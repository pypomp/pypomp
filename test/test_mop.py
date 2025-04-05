import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import *
from pypomp.mop import mop

LG_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = LG()

class TestMop_LG(unittest.TestCase):
    def setUp(self):
        self.J = 5
        self.ys = ys
        self.theta = theta
        self.covars = covars
        self.sigmas = 0.02
        self.key = jax.random.PRNGKey(111)

        self.rinit = rinit
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures

    def test_internal_basic(self):
        val1 = mop(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                   ys=self.ys, alpha=0.97, key=self.key)
        val2 = mop(rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta, ys=self.ys, key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)

    def test_class_basic(self):
        val1 = mop(LG_obj, self.J, alpha=0.97, key=self.key)
        val2 = mop(LG_obj, key=self.key)
        val3 = mop(LG_obj, self.J, self.rinit, self.rprocess, self.dmeasure, theta=[], ys=[], key=self.key)
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
            mop(key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J, alpha=0.97, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J, theta=self.theta, ys=self.ys, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, ys=self.ys, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

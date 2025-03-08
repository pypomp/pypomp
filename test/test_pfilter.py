import os
import jax
import sys
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.pfilter import pfilter
from pypomp.internal_functions import _pfilter_internal

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "pypomp")))
from LG import LG

LG_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = LG()

class TestPfilter_LG(unittest.TestCase):
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
        val1 = pfilter(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                       ys=self.ys, covars=self.covars, thresh=10, key=self.key)
        val2 = pfilter(rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta, ys=self.ys)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)

    def test_class_basic(self):
        val1 = pfilter(LG_obj, self.J, thresh=10, key=self.key)
        val2 = pfilter(LG_obj)
        val3 = pfilter(LG_obj, self.J, self.rinit, self.rprocess, self.dmeasure, theta=[], ys=[])
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
            pfilter()

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter(J=self.J)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter(J=self.J, thresh=-1, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter(J=self.J, theta=self.theta, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

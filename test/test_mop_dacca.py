import os
import csv
import jax
import sys
import unittest
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.mop import mop

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "pypomp")))
from dacca import dacca

dacca_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = dacca()

class TestMop_Dacca(unittest.TestCase):
    def setUp(self):
        self.J = 3
        self.key = jax.random.PRNGKey(111)
        dacca_obj, 
        ys, 
        theta, 
        covars, 
        rinit, 
        rprocess, 
        dmeasure, 
        rprocesses, 
        dmeasures = dacca()
        self.dacca_obj = dacca_obj
        self.ys = ys
        self.theta = theta
        self.covars = covars
        self.rinit = rinit
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures
        
    def test_internal_basic(self):
        val1 = mop(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                   ys=self.ys, covars=self.covars, alpha = 0.95, key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
    
    def test_class_basic(self):
        val = mop(dacca_obj, self.J, alpha = 0.95, key=self.key)
        self.assertEqual(val.shape, ())
        self.assertTrue(jnp.isfinite(val.item()))
        self.assertEqual(val.dtype, jnp.float32)

    def test_invalid_input(self):
        
        with self.assertRaises(ValueError) as text:
            mop()

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")
        
        # without inputting 'covars'
        with self.assertRaises(TypeError) as text:
            mop(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                    ys=self.ys, alpha = 0.95, key=self.key)

        self.assertEqual(str(text.exception), "'NoneType' object is not subscriptable") 


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
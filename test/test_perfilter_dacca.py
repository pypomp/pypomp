import os
import csv
import jax
import sys
import unittest
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.perfilter import perfilter

#current_dir = os.getcwd()
#sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "pypomp")))
sys.path.insert(0, 'pypomp')
from dacca import dacca

dacca_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = dacca()

class TestPerfilter_Dacca(unittest.TestCase):
    def setUp(self):
        self.J = 3
        self.key = jax.random.PRNGKey(111)
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
        val1, theta1 = perfilter(J=self.J, rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                 theta=self.theta, ys=self.ys, sigmas=0.02, covars=self.covars, thresh=-1, key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(theta1.shape, (self.J, 21))

    def test_class_basic(self):
        val, theta_new = perfilter(self.dacca_obj, self.J, sigmas=0.02, thresh=-1, key=self.key)
        self.assertEqual(val.shape, ())
        self.assertTrue(jnp.isfinite(val.item()))
        self.assertEqual(val.dtype, jnp.float32)
        self.assertEqual(theta_new.shape, (self.J, 21))
        
    def test_invalid_input(self):
        
        with self.assertRaises(ValueError) as text:
            perfilter()

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")
        
        # without inputting 'covars'
        with self.assertRaises(TypeError) as text:
            perfilter(J=self.J, rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                      theta=self.theta, ys=self.ys, sigmas=0.02, thresh=-1, key=self.key)

        self.assertEqual(str(text.exception), "'NoneType' object is not subscriptable") 


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
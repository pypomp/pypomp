import os
import csv
import jax
import sys
import unittest
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.fit import fit
from pypomp.internal_functions import _fit_internal

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "pypomp")))
from dacca import dacca

dacca_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = dacca()

class TestFit_Dacca(unittest.TestCase):
    def setUp(self):
        self.J = 3
        self.sigmas = 0.02
        self.sggmas_init = 1e-20
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
    
    def test_internal_basic_mif2(self):
        mif_loglik1, mif_theta1 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                      dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                      ys=self.ys, sigmas=0.02, sigmas_init=1e-20, covars=self.covars, M=2, a=0.9,
                                      thresh_mif=-1, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))
    
    def test_class_basic_mif2(self):
        mif_loglik1, mif_theta1 = fit(self.dacca_obj, J=self.J, Jh=3, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9,
                                      thresh_mif=-1, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

    def test_invalid_mif2(self):

        # missing covars argument
        with self.assertRaises(TypeError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20,
                ys=self.ys, M=2, a=0.9, thresh_mif=-1, mode="IF2")
            
        self.assertEqual(str(text.exception), "'NoneType' object is not subscriptable") 

    def test_internal_basic_GD(self):
        ## ls = False
        # method = SGD
        GD_loglik1, GD_theta1 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="SGD", alpha=0.97,
                                    covars=self.covars, scale=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

    def test_class_GD_basic(self):

        ## ls = True
        GD_loglik4, GD_theta4 = fit(dacca_obj, J=self.J, Jh=3, itns=2, method="BFGS", scale=True, ls=True, mode="GD")
        self.assertEqual(GD_loglik4.shape, (3,))
        self.assertEqual(GD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta4.dtype, jnp.float32))

    def test_invalid_GD_input(self):

        # missing covars argument
        with self.assertRaises(TypeError) as text:
            fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="SGD", alpha=0.97,
                scale=True, mode="GD")
            
        self.assertEqual(str(text.exception), "'NoneType' object is not subscriptable") 
        
    def test_internal_IFAD_basic(self):
        IFAD_loglik1, IFAD_theta1 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD",
                                        covars=self.covars, itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (3,))
        self.assertEqual(IFAD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

    def test_class_IFAD_basic(self):

        IFAD_loglik3, IFAD_theta3 = fit(dacca_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2,
                                        a=0.9, method="WeightedNewton", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik3.shape, (2,))
        self.assertEqual(IFAD_theta3.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta3.dtype, jnp.float32))

    def test_invalid_IFAD_input(self):

         # missing covars argument
        with self.assertRaises(TypeError) as text:
            fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD",
                itns=2, ls=True, alpha=0.97, mode="IFAD")
            
        self.assertEqual(str(text.exception), "'NoneType' object is not subscriptable") 
        



if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
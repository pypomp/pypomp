import os
import jax
import sys
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.fit import fit
from pypomp.internal_functions import _fit_internal

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "pypomp")))
from LG import LG


LG_obj, ys, theta, covars, rinit, rprocess, dmeasure, rprocesses, dmeasures = LG()

class TestFit_LG(unittest.TestCase):
    def setUp(self):
        self.ys = ys
        self.covars = None
        self.sigmas = 0.02
        self.J = 5
        self.theta = theta
        self.key = jax.random.PRNGKey(111)

        self.rinit = rinit
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures

    def test_internal_mif_basic(self):
        mif_loglik1, mif_theta1 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                      dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                      ys=self.ys, sigmas=0.02, sigmas_init=1e-20, covars=None, M=2, a=0.9,
                                      thresh_mif=-1, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

    def test_class_mif_basic(self):

        mif_loglik1, mif_theta1 = fit(LG_obj, J=self.J, Jh=10, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9,
                                      thresh_mif=-1, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = fit(LG_obj, sigmas=0.02, sigmas_init=1e-20, mode="IF2")
        self.assertEqual(mif_loglik2.shape, (11,))
        self.assertEqual(mif_theta2.shape, (11, 100,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

    def test_invalid_mif_input(self):

        with self.assertRaises(ValueError) as text:
            fit(mode="IF")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, thresh_mif=-1, mode="IF")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, sigmas_init=1e-20, M=2, a=0.9, thresh_mif=-1, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, sigmas=0.02, M=2, a=0.9, thresh_mif=-1, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, M=2, a=0.9, thresh_mif=-1, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=0.02, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas=0.02, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=0.02, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas=0.02, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

    def test_internal_GD_basic(self):
        ## ls = False
        # method = SGD
        GD_loglik1, GD_theta1 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="SGD", alpha=0.97,
                                    scale=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

        # method = Newton
        GD_loglik3, GD_theta3 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="Newton", alpha=0.97,
                                    scale=True, mode="GD")
        self.assertEqual(GD_loglik3.shape, (3,))
        self.assertEqual(GD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta3.dtype, jnp.float32))

        # refined Newton
        GD_loglik5, GD_theta5 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="WeightedNewton",
                                    alpha=0.97, scale=True, mode="GD")
        self.assertEqual(GD_loglik5.shape, (3,))
        self.assertEqual(GD_theta5.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik5.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta5.dtype, jnp.float32))

        # BFGS
        GD_loglik7, GD_theta7 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="BFGS", alpha=0.97,
                                    scale=True, mode="GD")
        self.assertEqual(GD_loglik7.shape, (3,))
        self.assertEqual(GD_theta7.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik7.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta7.dtype, jnp.float32))

    def test_class_GD_basic(self):

        ## ls = True
        GD_loglik1, GD_theta1 = fit(LG_obj, J=self.J, Jh=3, itns=2, method="SGD", alpha=0.97, scale=True, ls=True,
                                    mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

        GD_loglik2, GD_theta2 = fit(LG_obj, J=self.J, Jh=3, itns=2, method="Newton", alpha=0.97, scale=True, ls=True,
                                    mode="GD")
        self.assertEqual(GD_loglik2.shape, (3,))
        self.assertEqual(GD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta2.dtype, jnp.float32))

        GD_loglik3, GD_theta3 = fit(LG_obj, J=self.J, Jh=3, itns=2, method="WeightedNewton", alpha=0.97, scale=True,
                                    ls=True, mode="GD")
        self.assertEqual(GD_loglik3.shape, (3,))
        self.assertEqual(GD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta3.dtype, jnp.float32))

        GD_loglik4, GD_theta4 = fit(LG_obj, J=self.J, Jh=3, itns=2, method="BFGS", scale=True, ls=True, mode="GD")
        self.assertEqual(GD_loglik4.shape, (3,))
        self.assertEqual(GD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta4.dtype, jnp.float32))

    def test_invalid_GD_input(self):

        with self.assertRaises(ValueError) as text:
            fit(mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(mode="GD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures, theta=self.theta, ys=self.ys,
                mode="GD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, Jh=10, itns=2, alpha=0.97, scale=True, mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, Jh=10, itns=2, alpha=0.97, scale=True, mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, itns=2, alpha=0.97, scale=True, mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

    def test_internal_IFAD_basic(self):
        IFAD_loglik1, IFAD_theta1 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD",
                                        itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (3,))
        self.assertEqual(IFAD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

        IFAD_loglik2, IFAD_theta2 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="Newton",
                                        itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik2.shape, (3,))
        self.assertEqual(IFAD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta2.dtype, jnp.float32))

        IFAD_loglik3, IFAD_theta3 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9,
                                        method="WeightedNewton", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik3.shape, (3,))
        self.assertEqual(IFAD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta3.dtype, jnp.float32))

        IFAD_loglik4, IFAD_theta4 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="BFGS",
                                        itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik4.shape, (3,))
        self.assertEqual(IFAD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta4.dtype, jnp.float32))

    def test_class_IFAD_basic(self):

        IFAD_loglik1, IFAD_theta1 = fit(LG_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2, 
                                        a=0.9, method="SGD", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (2,))
        self.assertEqual(IFAD_theta1.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

        IFAD_loglik2, IFAD_theta2 = fit(LG_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2, 
                                        a=0.9, method="Newton", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik2.shape, (2,))
        self.assertEqual(IFAD_theta2.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta2.dtype, jnp.float32))

        IFAD_loglik3, IFAD_theta3 = fit(LG_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2,
                                        a=0.9, method="WeightedNewton", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik3.shape, (2,))
        self.assertEqual(IFAD_theta3.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta3.dtype, jnp.float32))

        IFAD_loglik4, IFAD_theta4 = fit(LG_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2, 
                                        a=0.9, method="BFGS", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik4.shape, (2,))
        self.assertEqual(IFAD_theta4.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta4.dtype, jnp.float32))

    def test_invalid_IFAD_input(self):

        with self.assertRaises(ValueError) as text:
            fit(mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(mode="AD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, alpha=0.97,
                mode="AD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, sigmas=self.sigmas, M=2, a=0.9, method="SGD", itns=2, alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(LG_obj, J=self.J, M=2, a=0.9, method="SGD", itns=2, alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, ls=True,
                alpha=0.97, mode="AD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, ls=True,
                alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas_init=1e-20, M=2, a=0.9, method="Newton", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=self.sigmas, M=2, a=0.9, method="WeightedNewton", itns=2, ls=True, alpha=0.97,
                mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, M=2, a=0.9, method="BFGS", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, M=2, a=0.9, method="SGD", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocesses=self.rprocesses,
                dmeasures=self.dmeasures,
                ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="Newton", itns=2, ls=True,
                alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

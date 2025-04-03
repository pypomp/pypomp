import os
import jax
import sys
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import *

#current_dir = os.getcwd()
#sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "pypomp")))
#from LG import LG_internal
sys.path.insert(0, 'pypomp')
from LG import LG

LG_obj, ys, theta, covars, rinit, rproc, dmeas, rprocess, dmeasure, rprocesses, dmeasures = LG_internal()

def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return A, C, Q, R


def transform_thetas(A, C, Q, R):
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])
    

class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        self.J = 5
        self.ys = ys
        self.theta = theta
        self.covars = covars
        self.sigmas = 0.02
        self.key = jax.random.PRNGKey(111)

        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures

    def test_basic_initialization(self): 
        self.assertEqual(LG_obj.covars, self.covars)
        obj_ys = LG_obj.ys
        self.assertTrue(jnp.array_equal(obj_ys, self.ys))
        self.assertTrue(jnp.array_equal(LG_obj.theta, self.theta))

    def test_invalid_initialization(self):
        # missing parameters
        with self.assertRaises(TypeError):
            Pomp(None, self.rproc, self.dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, None, self.dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, None, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, self.theta, self.covars)

    def test_mop_valid(self):
        mop_obj = LG_obj.mop(self.J, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj.shape, ())
        self.assertTrue(jnp.isfinite(mop_obj.item()))
        self.assertEqual(mop_obj.dtype, jnp.float32)

        mop_obj_edge = LG_obj.mop(1, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj_edge.shape, ())
        self.assertTrue(jnp.isfinite(mop_obj_edge.item()))
        self.assertEqual(mop_obj_edge.dtype, jnp.float32)

        # test mean
        mop_obj_mean = LG_obj.mop_mean(self.J, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj_mean.shape, ())
        self.assertTrue(jnp.isfinite(mop_obj_mean.item()))
        self.assertEqual(mop_obj_mean.dtype, jnp.float32)

    def test_mop_invalid(self):
        # missing values
        with self.assertRaises(TypeError):
            LG_obj.mop(alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.mop(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            LG_obj.mop(0, alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.mop(-1, alpha=0.97, key=self.key)
        with self.assertRaises(ValueError):
            LG_obj.mop(jnp.array([10, 20]), alpha=0.97, key=self.key)

        value = LG_obj.mop(self.J, alpha=jnp.inf, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertFalse(jnp.isfinite(value.item()))

        # undefined argument
        with self.assertRaises(TypeError):
            LG_obj.mop(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            LG_obj.mop(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.mop(self.J, self.rinit, self.rprocess, self.dmeasure, alpha=0.97, key=self.key)

        ### mop_mean
        with self.assertRaises(TypeError):
            LG_obj.mop_mean(alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.mop_mean(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            LG_obj.mop_mean(0, alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.mop_mean(-1, alpha=0.97, key=self.key)
        with self.assertRaises(ValueError):
            LG_obj.mop_mean(jnp.array([10, 20]), alpha=0.97, key=self.key)

        value_mean = LG_obj.mop_mean(self.J, alpha=jnp.inf, key=self.key)
        self.assertEqual(value_mean.dtype, jnp.float32)
        self.assertEqual(value_mean.shape, ())
        self.assertFalse(jnp.isfinite(value_mean.item()))

        # undefined argument
        with self.assertRaises(TypeError):
            LG_obj.mop_mean(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            LG_obj.mop_mean(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                            alpha=0.97, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.mop_mean(self.J, self.rinit, self.rprocess, self.dmeasure, alpha=0.97, key=self.key)

    # pfilter
    def test_pfilter_valid(self):
        pfilter_obj = LG_obj.pfilter(self.J, thresh=-1, key=self.key)
        self.assertEqual(pfilter_obj.shape, ())
        self.assertTrue(jnp.isfinite(pfilter_obj.item()))
        self.assertEqual(pfilter_obj.dtype, jnp.float32)

        pfilter_obj_edge = LG_obj.pfilter(1, thresh=10, key=self.key)
        self.assertEqual(pfilter_obj_edge.shape, ())
        self.assertTrue(jnp.isfinite(pfilter_obj_edge.item()))
        self.assertEqual(pfilter_obj_edge.dtype, jnp.float32)

        pfilter_obj_mean = LG_obj.pfilter_mean(self.J, thresh=-1, key=self.key)
        self.assertEqual(pfilter_obj_mean.shape, ())
        self.assertTrue(jnp.isfinite(pfilter_obj_mean.item()))
        self.assertEqual(pfilter_obj_mean.dtype, jnp.float32)

    def test_pfilter_invalid(self):
        # missing values
        with self.assertRaises(TypeError):
            LG_obj.pfilter(thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.pfilter(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            LG_obj.pfilter(0, thresh=100, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.pfilter(-1, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            LG_obj.pfilter(jnp.array([10, 20]), key=self.key)
        # undefined argument
        with self.assertRaises(TypeError):
            LG_obj.pfilter(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            LG_obj.pfilter(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                             key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.pfilter(self.J, self.rinit, self.rprocess, self.dmeasure, key=self.key)

        ### pfilter_mean
        with self.assertRaises(TypeError):
            LG_obj.pfilter_mean(thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.pfilter_mean(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            LG_obj.pfilter_mean(0, thresh=100, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.pfilter_mean(-1, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            LG_obj.pfilter_mean(jnp.array([10, 20]), key=self.key)
        # undefined argument
        with self.assertRaises(TypeError):
            LG_obj.pfilter_mean(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            LG_obj.pfilter_mean(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                  key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.pfilter_mean(self.J, self.rinit, self.rprocess, self.dmeasure, key=self.key)

    def test_perfilter_valid(self):
        val1, theta1 = LG_obj.perfilter(self.J, self.sigmas, key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(theta1.shape, (self.J, 16))

        val2, theta2 = LG_obj.perfilter(1, 0, thresh=10, key=self.key)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)
        self.assertEqual(theta2.shape, (1, 16))

        val3, theta3 = LG_obj.perfilter_mean(self.J, self.sigmas, key=self.key)
        self.assertEqual(val3.shape, ())
        self.assertTrue(jnp.isfinite(val3.item()))
        self.assertEqual(val3.dtype, jnp.float32)
        self.assertEqual(theta3.shape, (self.J, 16))

        val4, theta4 = LG_obj.perfilter_mean(1, 0, thresh=10, key=self.key)
        self.assertEqual(val4.shape, ())
        self.assertTrue(jnp.isfinite(val4.item()))
        self.assertEqual(val4.dtype, jnp.float32)
        self.assertEqual(theta4.shape, (1, 16))

    def test_perfilter_invalid(self):
        # missing values
        with self.assertRaises(TypeError):
            LG_obj.perfilter(sigmas=self.sigmas, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.perfilter(key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.perfilter(self.J, key=self.key)

        with self.assertRaises(IndexError):
            LG_obj.perfilter(0, self.sigmas, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            LG_obj.perfilter(-1, self.sigmas, thresh=100, key=self.key)

        # undefined arg
        with self.assertRaises(TypeError):
            LG_obj.perfilter(self.J, self.sigmas, alpha=0.97, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.perfilter(self.theta, self.ys, self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.perfilter(self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                             key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.perfilter_mean(sigmas=self.sigmas, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.perfilter_mean(key=self.key)

        with self.assertRaises(IndexError):
            LG_obj.perfilter_mean(0, self.sigmas, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            LG_obj.perfilter_mean(-1, self.sigmas, thresh=100, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.perfilter_mean(self.J, self.sigmas, alpha=0.97, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.perfilter_mean(self.theta, self.ys, self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess,
                                  dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.perfilter_mean(self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess,
                                  dmeasure=self.dmeasure, key=self.key)

    def test_pfilter_pf_valid(self):
        pfilter_obj = LG_obj.pfilter_pf(self.J, thresh=-1, key=self.key)
        self.assertEqual(pfilter_obj.shape, ())
        self.assertTrue(jnp.isfinite(pfilter_obj.item()))
        self.assertEqual(pfilter_obj.dtype, jnp.float32)

        pfilter_obj_edge = LG_obj.pfilter_pf(1, thresh=10, key=self.key)
        self.assertEqual(pfilter_obj_edge.shape, ())
        self.assertTrue(jnp.isfinite(pfilter_obj_edge.item()))
        self.assertEqual(pfilter_obj_edge.dtype, jnp.float32)

    def test_pfilter_pf_invalid(self):
        # missing values
        with self.assertRaises(TypeError):
            LG_obj.pfilter_pf(thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.pfilter_pf(key=self.key)
        # inapprpropriate input
        with self.assertRaises(TypeError):
            LG_obj.pfilter_pf(0, thresh=100, key=self.key)
        with self.assertRaises(TypeError):
            LG_obj.pfilter_pf(-1, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            LG_obj.pfilter_pf(jnp.array([10, 20]), key=self.key)
        # undefined argument
        with self.assertRaises(TypeError):
            LG_obj.pfilter_pf(self.J, a=0.97, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.pfilter_pf(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                              dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            LG_obj.pfilter_pf(self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, key=self.key)

    def test_fit_mif_valid(self):
       
        mif_loglik1, mif_theta1 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, J=self.J, mode="IF2",
                                            monitor=False)
        self.assertEqual(mif_loglik2.shape, (0,))
        self.assertEqual(mif_theta2.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

        # M = 0
        mif_loglik3, mif_theta3 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=0, a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik3.shape, (1,))
        self.assertEqual(mif_theta3.shape, (1, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta3.dtype, jnp.float32))

        # M = -1
        mif_loglik4, mif_theta4 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=-1, a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik4.shape, (1,))
        self.assertEqual(mif_theta4.shape, (1, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta4.dtype, jnp.float32))

        mif_loglik5, mif_theta5 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, mode="IF2")
        self.assertEqual(mif_loglik5.shape, (11,))
        self.assertEqual(mif_theta5.shape, (11, 100,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik5.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta5.dtype, jnp.float32))

    def test_fit_mif_invalid(self):
        # missing args
        with self.assertRaises(TypeError):
            LG_obj.fit(mode="IF2")
        with self.assertRaises(TypeError):
            LG_obj.fit(mode="IF")
        with self.assertRaises(TypeError):
            LG_obj.fit(sigmas=0.02, M=1, mode="IF2")
        with self.assertRaises(TypeError):
            LG_obj.fit(sigmas_init=1e-20, a=0.9, mode="IF2")

        # useless input
        with self.assertRaises(TypeError):
            LG_obj.fit(rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, rprocesses=self.rprocesses,
                       dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, mode="IF2")

    def test_fit_GD_valid(self):

        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        
        for method in methods:
            with self.subTest(method=method):
                GD_loglik1, GD_theta1 = LG_obj.fit(J=3, Jh=3, method=method, itns=1, alpha=0.97, scale=True, mode="GD")
                self.assertEqual(GD_loglik1.shape, (2,))
                self.assertEqual(GD_theta1.shape, (2,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))
                
                if method in ["WeightedNewton", "BFGS"]:
                    GD_loglik2, GD_theta2 = LG_obj.fit(J=3, Jh=3, method=method, itns=1, alpha=0.97, scale=True, ls=True, mode="GD")
                    self.assertEqual(GD_loglik2.shape, (2,))
                    self.assertEqual(GD_theta2.shape, (2,) + self.theta.shape)
                    self.assertTrue(jnp.issubdtype(GD_loglik2.dtype, jnp.float32))
                    self.assertTrue(jnp.issubdtype(GD_theta2.dtype, jnp.float32))

    def test_fit_GD_invalid(self):
    
        with self.assertRaises(TypeError):
            LG_obj.fit(mode="SGD")
        with self.assertRaises(TypeError):
            LG_obj.fit(J=0, mode="GD")
        with self.assertRaises(TypeError):
            LG_obj.fit(J=-1, mode="GD")
        with self.assertRaises(TypeError):
            LG_obj.fit(Jh=0, mode="GD")
        with self.assertRaises(TypeError):
            LG_obj.fit(Jh=-1, mode="GD")

        # useless input
        with self.assertRaises(TypeError):
            LG_obj.fit(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=10, method="BFGS",
                       itns=2, alpha=0.97, scale=True, mode="GD")

    def test_fit_IFAD_valid(self):
        #pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)
        IFAD_loglik1, IFAD_theta1 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="SGD", itns=2,
                                               alpha=0.97, scale=True, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (3,))
        self.assertEqual(IFAD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

        IFAD_loglik2, IFAD_theta2 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="Newton",
                                               itns=2, alpha=0.97, scale=True, ls=True, mode="IFAD")
        self.assertEqual(IFAD_loglik2.shape, (3,))
        self.assertEqual(IFAD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta2.dtype, jnp.float32))

        IFAD_loglik3, IFAD_theta3 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10,
                                               method="WeightedNewton", itns=2, alpha=0.97, scale=True, mode="IFAD")
        self.assertEqual(IFAD_loglik3.shape, (3,))
        self.assertEqual(IFAD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta3.dtype, jnp.float32))

        IFAD_loglik4, IFAD_theta4 = LG_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="BFGS",
                                               itns=2, alpha=0.97, scale=True, mode="IFAD")
        self.assertEqual(IFAD_loglik4.shape, (3,))
        self.assertEqual(IFAD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta4.dtype, jnp.float32))

    def test_fit_IFAD_invalid(self):
        # pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)
        # missing
        with self.assertRaises(TypeError):
            LG_obj.fit()
        with self.assertRaises(TypeError):
            LG_obj.fit(mode="ADIF")
        with self.assertRaises(TypeError):
            LG_obj.fit(mode="IFAD")
        with self.assertRaises(TypeError):
            LG_obj.fit(sigmas=self.sigmas, mode="IFAD")

        # useless input
        with self.assertRaises(TypeError):
            LG_obj.fit(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                       sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="SGD", itns=2, alpha=0.97, scale=True,
                       mode="IFAD")

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

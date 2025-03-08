import os
import jax
import sys
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.internal_functions import _fit_internal

curr_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(curr_dir, "..", "pypomp")))
from LG import LG_internal

def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return A, C, Q, R


def transform_thetas(A, C, Q, R):
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])

LG_obj, ys, theta, covars, rinit, rproc, dmeas, rprocess, dmeasure, rprocesses, dmeasures = LG_internal()

class TestFitInternal_LG(unittest.TestCase):
    def setUp(self):
        self.J = 5
        self.ys = ys
        self.theta = theta
        self.covars = covars
        self.key = jax.random.PRNGKey(111)

        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures 
        
    def test_basic_mif(self):
        mif_loglik1, mif_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2,
                                               a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2,
                                               a=0.9, J=self.J, mode="IF2", monitor=False)
        self.assertEqual(mif_loglik2.shape, (0,))
        self.assertEqual(mif_theta2.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

        mif_loglik3, mif_theta3 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20,
                                               thresh_mif=-1, mode="IF2")
        self.assertEqual(mif_loglik3.shape, (11,))
        self.assertEqual(mif_theta3.shape, (11, 100,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta3.dtype, jnp.float32))

        mif_loglik4, mif_theta4 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20,
                                               thresh_mif=-1, mode="IF2", monitor=False)
        self.assertEqual(mif_loglik4.shape, (0,))
        self.assertEqual(mif_theta4.shape, (11, 100,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta4.dtype, jnp.float32))

    def test_edge_mif_J(self):
        mif_loglik1, mif_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2,
                                               a=0.9, J=1, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, 1,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2,
                                               a=0.9, J=100, mode="IF2")
        self.assertEqual(mif_loglik2.shape, (3,))
        self.assertEqual(mif_theta2.shape, (3, 100,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

    def test_edge_mif_sigmas(self):
        # sigmas = 0 and sigmas_init! = 0
        mif_loglik1, mif_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0, sigmas_init=1e-20, M=2, a=0.9,
                                               J=self.J, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))
        self.assertTrue(jnp.all(mif_loglik1 == mif_loglik1[0]))

        # sigmas_init = 0 and sigmas ！= 0
        mif_loglik2, mif_theta2 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=0, M=2, a=0.9,
                                               J=self.J, mode="IF2")
        self.assertEqual(mif_loglik2.shape, (3,))
        self.assertEqual(mif_theta2.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

        # sigmas = 0 and sigmas_init = 0
        mif_loglik3, mif_theta3 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0, sigmas_init=0, M=2, a=0.9,
                                               J=self.J, mode="IF2")
        self.assertEqual(mif_loglik3.shape, (3,))
        self.assertEqual(mif_theta3.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta3.dtype, jnp.float32))
        self.assertTrue(jnp.all(mif_loglik3 == mif_loglik3[0]))
        self.assertTrue(jnp.array_equal(mif_theta3[0], mif_theta3[1]))

    def test_edge_mif_ys(self):
        ys = self.ys[0, :]
        mif_loglik, mif_theta = _fit_internal(self.theta, ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses,
                                             self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, J=self.J,
                                             mode="IF2")
        self.assertEqual(mif_loglik.shape, (3,))
        self.assertEqual(mif_theta.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta.dtype, jnp.float32))

    def test_edge_mif_M(self):
        # M = 0
        mif_loglik, mif_theta = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                             self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=0,
                                             a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik.shape, (1,))
        self.assertEqual(mif_theta.shape, (1, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta.dtype, jnp.float32))

        # M = -1
        mif_loglik1, mif_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                               self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=-1,
                                               a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (1,))
        self.assertEqual(mif_theta1.shape, (1, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

    def test_mif_dmeasure_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        dmeasures = jax.vmap(custom_dmeas, (None, 0, 0))
        mif_loglik, mif_theta = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, dmeasure, self.rprocesses,
                                             dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, J=self.J,
                                             mode="IF2")
        self.assertEqual(mif_loglik.shape, (3,))
        self.assertEqual(mif_theta.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta.dtype, jnp.float32))
        self.assertTrue(jnp.all(jnp.isnan(mif_loglik)))

    def test_mif_zero_dmeasure(self):
        def zero_dmeasure(ys, particlesP, theta):
            return jnp.zeros((particlesP.shape[0],))

        def zero_dmeasures(ys, particlesP, theta):
            return jnp.zeros((particlesP.shape[0],))

        mif_loglik, mif_theta = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, zero_dmeasure,
                                             self.rprocesses, zero_dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2,
                                             a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik.shape, (3,))
        self.assertEqual(mif_theta.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta.dtype, jnp.float32))
        self.assertTrue(jnp.array_equal(mif_loglik, jnp.zeros(3)))

    def test_mif_rprocess_inf(self):
        def custom_rproc(state, theta, key, covars=None):
            # override the state variable
            state = jnp.array([-jnp.inf, -jnp.inf])
            A, C, Q, R = get_thetas(theta)
            key, subkey = jax.random.split(key)
            return jax.random.multivariate_normal(key=subkey,
                                                  mean=A @ state, cov=Q)

        rprocess = jax.vmap(custom_rproc, (0, None, 0, None))
        rprocesses = jax.vmap(custom_rproc, (0, 0, 0, None))
        mif_loglik, mif_theta = _fit_internal(self.theta, self.ys, self.rinit, rprocess, self.dmeasure, rprocesses,
                                             self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, J=self.J,
                                             mode="IF2")
        self.assertEqual(mif_loglik.shape, (3,))
        self.assertEqual(mif_theta.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta.dtype, jnp.float32))
        self.assertTrue(jnp.all(jnp.isnan(mif_loglik)))

    def test_missing_mif(self):
        with self.assertRaises(TypeError):
            _fit_internal(mode="IF2")
        with self.assertRaises(TypeError):
            _fit_internal(mode="IF")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, mode="IF2")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, mode="IF2")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, mode="IF2")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         mode="IF2")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IF")

    def test_mif_wrongtype_J(self):
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                          sigmas=0.02, sigmas_init=1e-20, J=jnp.array([10, 20]), mode="IF2")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, J="pop", mode="IF2")

        def generate_J(n):
            return jnp.array(10, 20)

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, J=lambda n: generate_J(n), mode="IF2")

    def test_mif_wrongtype_theta(self):
        with self.assertRaises(AttributeError):
            theta = "theta"
            _fit_internal(theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IF2")
        with self.assertRaises(TypeError):
            theta = jnp.array(["theta"])
            _fit_internal(theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IF2")
        with self.assertRaises(IndexError):
            theta = jnp.array(5)
            _fit_internal(theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IF2")

    def test_mif_wrongtype_rinit(self):
        def onestep(theta, J, covars=None):
            raise RuntimeError("boink")

        rinit = onestep

        with self.assertRaises(RuntimeError) as cm:
            _fit_internal(self.theta, self.ys, rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IF2")

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, rinit="rinit", rprocess=self.rprocess, dmeausre=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20,
                         mode="IF2")

    def test_mif_wrongtype_rprocess(self):
        def onestep(state, theta, key, covars=None):
            raise RuntimeError("boink")

        rprocess = onestep
        rprocesses = onestep

        with self.assertRaises(RuntimeError) as cm:
            _fit_internal(self.theta, self.ys, self.rinit, rprocess, self.dmeasure, rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IF2")
        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, rprocess="rprocess", dmeausre=self.dmeasure,
                         rprocesses="rprocesses", dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, mode="IF2")

    def test_mif_wrongtype_dmeasure(self):
        def onestep(y, preds, theta):
            raise RuntimeError("boink")

        dmeasure = onestep
        dmeasures = onestep

        with self.assertRaises(RuntimeError) as cm:
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, dmeasure, self.rprocesses, dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IF2")

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, dmeasure="dmeasure",
                         rprocesses=self.rprocesses, dmeasures="dmeasures", sigmas=0.02, sigmas_init=1e-20, mode="IF2")

    def test_mif_invalid_J(self):
        with self.assertRaises(TypeError):
            J = 0
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, J=J,
                         mode="IF2")

        with self.assertRaises(ValueError):
            J = -1
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, J=J,
                         mode="IF2")

    def test_mif_invalid_ys(self):
        # ys = self.ys[0,:]
        y = jnp.full(self.ys.shape, jnp.inf)
        mif_loglik, mif_theta = _fit_internal(self.theta, y, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                                             rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02,
                                             sigmas_init=1e-20, mode="IF2")
        self.assertTrue(jnp.all(jnp.isnan(mif_loglik)))

    def test_mif_invalid_sigmas(self):
        sigmas = jnp.inf
        mif_loglik, mif_theta = _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess,
                                             dmeasure=self.dmeasure, rprocesses=self.rprocesses,
                                             dmeasures=self.dmeasures, sigmas=sigmas, sigmas_init=1e-20, mode="IF2")
        self.assertFalse(jnp.isnan(mif_loglik[0]))
        self.assertTrue(jnp.all(jnp.isnan(mif_loglik[1:])))
        sigmas_init = jnp.inf
        mif_loglik2, mif_theta2 = _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess,
                                               dmeasure=self.dmeasure, rprocesses=self.rprocesses,
                                               dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=sigmas_init,
                                               mode="IF2")
        self.assertTrue(jnp.all(jnp.isnan(mif_loglik2)))

    def test_new_arg(self):
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, b=0.97,
                         mode="IF2")

    def test_basic_GD(self):
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
    
        for method in methods:
            with self.subTest(method=method):
                GD_loglik, GD_theta = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                                    J=2, Jh=2, method=method, itns=2, alpha=0.97, scale=True, mode="GD")
                self.assertEqual(GD_loglik.shape, (3,))
                self.assertEqual(GD_theta.shape, (3,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(GD_loglik.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(GD_theta.dtype, jnp.float32))

                GD_loglik_ls, GD_theta_ls = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                                          J=2, Jh=2, method=method, itns=2, alpha=0.97, scale=True, 
                                                          ls=True, mode="GD")
                self.assertEqual(GD_loglik_ls.shape, (3,))
                self.assertEqual(GD_theta_ls.shape, (3,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(GD_loglik_ls.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(GD_theta_ls.dtype, jnp.float32))

    def test_edge_GD_J(self):
        # J, Jh = 1, 1
        GD_loglik1, GD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=1,
                                              Jh=1, method="SGD", itns=1, alpha=0.97, scale=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (2,))
        self.assertEqual(GD_theta1.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))
        

    def test_edge_GD_itns(self):
        GD_loglik1, GD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=3, Jh=3,
                                             method="Newton", itns=0, alpha=0.97, scale=True, ls=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (1,))
        self.assertEqual(GD_theta1.shape, (1,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

        GD_loglik2, GD_theta2 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=3, Jh=3,
                                             method="Newton", itns=-1, alpha=0.97, scale=True, ls=True, mode="GD")
        self.assertEqual(GD_loglik2.shape, (1,))
        self.assertEqual(GD_theta2.shape, (1,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta2.dtype, jnp.float32))

    def test_edge_GD_eta(self):
        GD_loglik1, GD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=3, Jh=3,
                                             method="BFGS", itns=1, alpha=0.97, scale=True, eta=0, mode="GD")
        self.assertEqual(GD_loglik1.shape, (2,))
        self.assertEqual(GD_theta1.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))
        self.assertTrue(jnp.array_equal(GD_theta1[0], GD_theta1[1]))

    def test_edge_GD_alpha(self):
        GD_loglik1, GD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=3, Jh=3,
                                             method="SGD", itns=1, alpha=1, scale=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (2,))
        self.assertEqual(GD_theta1.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

        GD_loglik2, GD_theta2 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=3, Jh=3,
                                             method="BFGS", itns=1, alpha=0, scale=True, mode="GD")
        self.assertEqual(GD_loglik2.shape, (2,))
        self.assertEqual(GD_theta2.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta2.dtype, jnp.float32))

    def test_edge_GD_thresh(self):
        GD_loglik1, GD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=3, Jh=3,
                                             method="BFGS", itns=2, alpha=0.97, scale=True, thresh_tr=-10000, mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

        GD_loglik2, GD_theta2 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=3, Jh=3,
                                             method="BFGS", itns=2, alpha=0.97, scale=True, thresh_tr=10000, mode="GD")
        self.assertEqual(GD_loglik2.shape, (3,))
        self.assertEqual(GD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta2.dtype, jnp.float32))

    def test_GD_dmeasure_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        GD_loglik1, GD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, dmeasure, J=3, Jh=3,
                                             method="WeightedNewton", itns=2, alpha=0.97, scale=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))
        self.assertTrue(jnp.all(jnp.isnan(GD_loglik1)))

    def test_GD_dmeasure_zero(self):
        def zero_dmeasure(ys, particlesP, theta):
            return jnp.zeros((particlesP.shape[0],))

        GD_loglik1, GD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, zero_dmeasure, J=3, Jh=3,
                                             method="BFGS", itns=2, alpha=0.97, scale=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))
        self.assertTrue(jnp.array_equal(GD_loglik1, jnp.zeros(3)))

    def test_GD_rprocess_inf(self):
        def custom_rproc(state, theta, key, covars=None):
            # override the state variable
            state = jnp.array([-jnp.inf, -jnp.inf])
            A, C, Q, R = get_thetas(theta)
            key, subkey = jax.random.split(key)
            return jax.random.multivariate_normal(key=subkey,
                                                  mean=A @ state, cov=Q)

        rprocess = jax.vmap(custom_rproc, (0, None, 0, None))
        GD_loglik, GD_theta = _fit_internal(self.theta, self.ys, self.rinit, rprocess, self.dmeasure, J=3, Jh=3,
                                           method="WeightedNewton", itns=1, alpha=0.97, scale=True, mode="GD")
        self.assertEqual(GD_loglik.shape, (2,))
        self.assertEqual(GD_theta.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta.dtype, jnp.float32))
        self.assertTrue(jnp.all(jnp.isnan(GD_loglik)))

    def test_missing_GD(self):
        with self.assertRaises(TypeError):
            _fit_internal(mode="GD")
        with self.assertRaises(TypeError):
            _fit_internal(mode="SGD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, mode="GD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, mode="GD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rprocess, self.dmeasure, mode="GD")

    def test_GD_wrongtype_J(self):
        with self.assertRaises(ValueError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=jnp.array([10, 20]), Jh=10,
                          method="SGD", itns=2, alpha=0.97, scale=True, mode="GD")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J="pop", Jh=10, method="SGD",
                         itns=2, alpha=0.97, scale=True, mode="GD")

        def generate_J(n):
            return jnp.array(10, 20)

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=lambda n: generate_J(n),
                         Jh=10, method="SGD", itns=2, alpha=0.97, scale=True, mode="GD")

        with self.assertRaises(ValueError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, Jh=jnp.array([10, 20]), J=10,
                          method="Newton", itns=2, alpha=0.97, scale=True, mode="GD")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, Jh="pop", J=10, method="Newton",
                         itns=2, alpha=0.97, scale=True, mode="GD")

        def generate_Jh(n):
            return jnp.array(10, 20)

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, Jh=lambda n: generate_Jh(n),
                         J=10, method="Newton", itns=2, alpha=0.97, scale=True, mode="GD")

    def test_GD_wrongtype_theta(self):
        with self.assertRaises(AttributeError):
            theta = "theta"
            _fit_internal(theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=10, method="SGD", itns=2,
                         alpha=0.97, scale=True, mode="GD")
        with self.assertRaises(TypeError):
            theta = jnp.array(["theta"])
            _fit_internal(theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=10, method="SGD", itns=2,
                         alpha=0.97, scale=True, mode="GD")
        with self.assertRaises(IndexError):
            theta = jnp.array(5)
            _fit_internal(theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=10, method="SGD", itns=2,
                         alpha=0.97, scale=True, mode="GD")

    def test_GD_wrongtype_rinit(self):
        def onestep(theta, J, covars=None):
            raise RuntimeError("boink")

        rinit = onestep

        with self.assertRaises(RuntimeError) as cm:
            _fit_internal(self.theta, self.ys, rinit, self.rprocess, self.dmeasure, J=10, Jh=10, method="Newton", itns=2,
                         alpha=0.97, scale=True, mode="GD")

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, rinit="rinit", rprocess=self.rprocess, dmeasure=self.dmeasure, J=10,
                         Jh=10, method="Newton", itns=2, alpha=0.97, scale=True, mode="GD")

    def test_GD_wrongtype_rprocess(self):
        def onestep(state, theta, key, covars=None):
            raise RuntimeError("boink")

        rprocess = onestep

        with self.assertRaises(RuntimeError) as cm:
            _fit_internal(self.theta, self.ys, self.rinit, rprocess, self.dmeasure, J=10, Jh=10, method="WeightedNewton",
                         itns=2, alpha=0.97, scale=True, mode="GD")
        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, rprocess="rprocess", dmeasure=self.dmeasure, J=10, Jh=10,
                         method="WeightedNewton", itns=2, alpha=0.97, scale=True, mode="GD")

    def test_GD_wrongtype_dmeasure(self):
        def onestep(y, preds, theta):
            raise RuntimeError("boink")

        dmeasure = onestep

        with self.assertRaises(RuntimeError) as cm:
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, dmeasure, J=10, Jh=10, method="BFGS", itns=2,
                         alpha=0.97, scale=True, mode="GD")

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, dmeasure="dmeasure", J=10, Jh=10,
                         method="BFGS", itns=2, alpha=0.97, scale=True, mode="GD")

    def test_GD_invalid_J(self):
        with self.assertRaises(TypeError):
            J = 0
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=J, Jh=10, method="Newton",
                         itns=2, alpha=0.97, scale=True, mode="GD")

        with self.assertRaises(TypeError):
            J = -1
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=J, Jh=10, method="Newton",
                         itns=2, alpha=0.97, scale=True, mode="GD")

        with self.assertRaises(TypeError):
            Jh = 0
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=Jh,
                         method="WeightedNewton", itns=2, alpha=0.97, scale=True, mode="GD")

        with self.assertRaises(TypeError):
            Jh = -1
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=Jh,
                         method="WeightedNewton", itns=2, alpha=0.97, scale=True, mode="GD")

    def test_GD_invalid_ys(self):
        # ys = self.ys[0,:]
        y = jnp.full(self.ys.shape, jnp.inf)
        GD_loglik, GD_theta = _fit_internal(self.theta, y, self.rinit, self.rprocess, self.dmeasure, J=2, Jh=2,
                                           method="BFGS", itns=2, alpha=0.97, scale=True, mode="GD")
        self.assertTrue(jnp.all(jnp.isnan(GD_loglik)))

    def test_new_GD_arg(self):
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=10, method="SGD",
                         itns=2, alpha=0.97, scale=True, b=0.5, mode="GD")

    def test_basic_IFAD(self):
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for method in methods:
            with self.subTest(method=method):
                IFAD_loglik, IFAD_theta = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                                        self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, 
                                                        M=2, J=1, Jh=1, method=method, itns=1, alpha=0.97, 
                                                        scale=True, mode="IFAD")
                self.assertEqual(IFAD_loglik.shape, (2,))
                self.assertEqual(IFAD_theta.shape, (2,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(IFAD_loglik.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(IFAD_theta.dtype, jnp.float32))



    def test_missing_IFAD(self):
        with self.assertRaises(TypeError):
            _fit_internal()
        with self.assertRaises(TypeError):
            _fit_internal(mode="ADIF")
        with self.assertRaises(TypeError):
            _fit_internal(mode="IFAD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, mode="IFAD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, mode="IFAD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rprocess, self.dmeasure, mode="IFAD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rprocess, self.dmeasure, sigmas=0.02, sigmas_init=1e-20, mode="IFAD")
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, mode="IFAD")

    def test_edge_IFAD_M(self):
        # M = 0
        IFAD_loglik1, IFAD_theta1 = _fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,
                                                 self.rprocesses, self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=0,
                                                 J=10, Jh=10, method="SGD", itns=2, alpha=0.97, scale=True, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (3,))
        self.assertEqual(IFAD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

    def test_IFAD_invalid_J(self):
        with self.assertRaises(TypeError):
            J = 0
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, J=J,
                         Jh=5, mode="IFAD")

        with self.assertRaises(ValueError):
            J = -1
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, J=J,
                         Jh=5, mode="IFAD")

        with self.assertRaises(TypeError):
            Jh = 0
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, J=3,
                         Jh=Jh, method="Newton", mode="IFAD")

        with self.assertRaises(TypeError):
            Jh = -1
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, J=3,
                         Jh=Jh, method="WeightedNewton", mode="IFAD")

    def test_IFAD_invalid_ys(self):
        ys = self.ys[0,:]
        y = jnp.full(self.ys.shape, jnp.inf)
        IFAD_loglik, IFAD_theta = _fit_internal(self.theta, y, self.rinit, rprocess=self.rprocess,
                                               dmeasure=self.dmeasure, rprocesses=self.rprocesses,
                                               dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, M=2, itns=1,
                                               J=3, Jh=3, method="SGD", mode="IFAD")
        self.assertTrue(jnp.all(jnp.isnan(IFAD_loglik)))

    def test_IFAD_invalid_sigmas(self):
        sigmas = jnp.inf
        sigmas_init = jnp.inf
        IFAD_loglik, IFAD_theta = _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess,
                                               dmeasure=self.dmeasure, rprocesses=self.rprocesses,
                                               dmeasures=self.dmeasures, sigmas=sigmas, sigmas_init=sigmas_init, M=2, 
                                               itns=1, J=3, Jh=3, mode="IFAD")
        self.assertTrue(jnp.all(jnp.isnan(IFAD_loglik)))


    def test_IFAD_arg(self):
        with self.assertRaises(TypeError):
            _fit_internal(self.theta, self.ys, self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         rprocesses=self.rprocesses, dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, b=0.97,
                         mode="IF2")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

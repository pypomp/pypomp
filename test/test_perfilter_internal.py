import os
import jax
import sys
import unittest
import jax.numpy as jnp

from jax import vmap
from pypomp.LG import *
from pypomp.internal_functions import _perfilter_internal
from pypomp.internal_functions import _perfilter_internal_mean

def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return jnp.array([A, C, Q, R])

get_perthetas = vmap(get_thetas, in_axes = 0)

def transform_thetas(A, C, Q, R):
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])

LG_obj, ys, theta, covars, rinit, rproc, dmeas, rprocess, dmeasure, rprocesses, dmeasures = LG_internal()

class TestPerfilterInternal_LG(unittest.TestCase):
    def setUp(self):
        self.J = 10
        self.ys = ys
        self.theta = theta
        self.sigmas = 0.02
        self.covars = None
        self.key = jax.random.PRNGKey(111)

        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures
        self.ndim = self.theta.ndim

    def test_basic_function(self):
        result1, theta1 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        self.assertEqual(theta1.shape, (self.J, 16))
        theta1_new = get_perthetas(theta1)
        self.assertEqual(theta1_new.shape, (self.J, 4, 2, 2))
        
        result2, theta2 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(theta2.shape, (self.J, 16))
        self.assertEqual(result1, result2)
        self.assertTrue(jnp.array_equal(theta1, theta2))
        theta2_new = get_perthetas(theta2)
        self.assertEqual(theta2_new.shape, (self.J, 4, 2, 2))

        result3, theta3 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=10, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        self.assertEqual(theta3.shape, (self.J, 16))
        theta3_new = get_perthetas(theta3)
        self.assertEqual(theta3_new.shape, (self.J, 4, 2, 2))

        result4, theta4 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=10, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        self.assertEqual(theta4.shape, (self.J, 16))
        theta4_new = get_perthetas(theta4)
        self.assertEqual(theta4_new.shape, (self.J, 4, 2, 2))

        self.assertEqual(result3, result4)
        self.assertTrue(jnp.array_equal(theta3, theta4))
        self.assertTrue(jnp.array_equal(theta3_new, theta4_new))

        result5, theta5 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result5.shape, ())
        self.assertTrue(jnp.isfinite(result5.item()))
        self.assertEqual(result5.dtype, jnp.float32)
        self.assertEqual(theta5.shape, (self.J, 16))
        theta5_new = get_perthetas(theta5)
        self.assertEqual(theta5_new.shape, (self.J, 4, 2, 2))

        result6, theta6 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result6.shape, ())
        self.assertTrue(jnp.isfinite(result6.item()))
        self.assertEqual(result6.dtype, jnp.float32)
        self.assertEqual(theta6.shape, (self.J, 16))
        theta6_new = get_perthetas(theta6)
        self.assertEqual(theta6_new.shape, (self.J, 4, 2, 2))

        self.assertEqual(result5, result6)
        self.assertTrue(jnp.array_equal(theta5, theta6))
        self.assertTrue(jnp.array_equal(theta5_new, theta6_new))

    def test_edge_J(self):
        result1, theta1 = _perfilter_internal(self.theta, self.ys, 1, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        self.assertEqual(theta1.shape, (1, 16))
        theta1_new = get_perthetas(theta1)
        self.assertEqual(theta1_new.shape, (1, 4, 2, 2))
        result2, theta2 = _perfilter_internal(self.theta, self.ys, 1, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(theta2.shape, (1, 16))
        theta2_new = get_perthetas(theta2)
        self.assertEqual(theta2_new.shape, (1, 4, 2, 2))
        result3, theta3 = _perfilter_internal(self.theta, self.ys, 100, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        self.assertEqual(theta3.shape, (100, 16))
        theta3_new = get_perthetas(theta3)
        self.assertEqual(theta3_new.shape, (100, 4, 2, 2))
        result4, theta4 = _perfilter_internal(self.theta, self.ys, 100, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        self.assertEqual(theta4.shape, (100, 16))
        theta4_new = get_perthetas(theta4)
        self.assertEqual(theta4_new.shape, (100, 4, 2, 2))

    def test_edge_thresh(self):
        result1, theta1 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=0, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        self.assertEqual(theta1.shape, (self.J, 16))
        result2, theta2 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-10, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(theta2.shape, (self.J, 16))
        self.assertEqual(result1, result2)

        result3, theta3 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=10000, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        self.assertEqual(theta3.shape, (self.J, 16))

        result4, theta4 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1000, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        self.assertEqual(theta4.shape, (self.J, 16))
        self.assertEqual(result1, result4)

    def test_edge_ys(self):
        # when len(ys) = 1
        ys = self.ys[0, :]
        result, theta = _perfilter_internal(self.theta, ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                           self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)
        self.assertTrue(theta.shape, (self.J, 16))

    def test_edge_sigmas(self):
        # when len(ys) = 1
        sigmas = 0
        result, theta = _perfilter_internal(self.theta, self.ys, self.J, sigmas, self.rinit, self.rprocesses,
                                           self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)
        self.assertTrue(theta.shape, (self.J, 16))

        sigmas = -0.001
        result3, theta3 = _perfilter_internal(self.theta, self.ys, self.J, sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        self.assertTrue(theta.shape, (self.J, 16))

    def test_dmeasures_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasures = jax.vmap(custom_dmeas, (None, 0, 0))
        result, theta = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                           dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(result.shape, ())
        self.assertFalse(jnp.isfinite(result.item()))
        self.assertEqual(theta.shape, (self.J, 16))

    def test_zero_dmeasures(self):
        def zero_dmeasures(y, particlesP, thetas):
            return jnp.zeros((particlesP.shape[0],))

        result, theta = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                           zero_dmeasures, self.ndim, covars=self.covars, key=self.key)
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 0.0)
        self.assertEqual(theta.shape, (self.J, 16))

    def test_rprocess_inf(self):
        def custom_rproc(state, theta, key, covars=None):
            # override the state variable
            state = jnp.array([-jnp.inf, -jnp.inf])
            A, C, Q, R = get_thetas(theta)
            key, subkey = jax.random.split(key)
            return jax.random.multivariate_normal(key=subkey,
                                                  mean=A @ state, cov=Q)

        rprocesses = jax.vmap(custom_rproc, (0, 0, 0, None))
        result, theta = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, rprocesses,
                                           self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertFalse(jnp.isnan(result).item())
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(result.shape, ())
        self.assertEqual(theta.shape, (self.J, 16))

    def test_invalid_ys(self):
        y = jnp.full(self.ys.shape, jnp.inf)
        value, theta = _perfilter_internal(self.theta, y, self.J, self.sigmas, rinit=self.rinit,
                                          rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim,
                                          covars=self.covars, key=self.key)
        self.assertFalse(jnp.isnan(value).item())
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertEqual(theta.shape, (self.J, 16))

    def test_missing(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.ys,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rprocesses,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.dmeasures,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit, self.rprocesses,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit, self.dmeasures,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rprocesses, self.dmeasures,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit, self.rprocesses, self.dmeasures,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rinit, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocesses, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rinit, self.dmeasures,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rprocesses, self.dmeasures, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.dmeasures, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rprocesses, self.dmeasures,key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.ndim, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.ndim, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.ndim, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.dmeasures, self.ndim, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rprocesses, self.dmeasures, self.ndim, key=self.key)

    def test_missing_theta(self):
        with self.assertRaises(AttributeError):
            _perfilter_internal(None, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(AttributeError):
            _perfilter_internal(None, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, key=self.key)

    def test_missing_ys(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, None, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, None, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, key=self.key)

    def test_missing_J(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, None, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, None, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, key=self.key)

    def test_missing_sigmas(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, None, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, None, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, key=self.key)

    def test_missing_rinit(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, None, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, None, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, key=self.key)

    def test_missing_rprocesses(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, None, self.dmeasures, self.ndim,
                               self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, None, self.dmeasures, self.ndim,
                               self.covars, key=self.key)

    def test_missing_dmeasures(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, None, self.ndim,
                               self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, None, self.ndim,
                               self.covars, key=self.key)

    # wrong type parameters
    def test_wrongtype_J(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, J=jnp.array(10, 20), sigmas=self.sigmas, rinit=self.rinit,
                                rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars,key=self.key)

        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, J="pop", sigmas=self.sigmas, rinit=self.rinit,
                               rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

        def generate_J(n):
            return jnp.array(10, 20)

        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, J=lambda n: generate_J(n), sigmas=self.sigmas, rinit=self.rinit,
                               rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

    def test_wrongtype_theta(self):
        with self.assertRaises(TypeError):
            theta = "theta"
            _perfilter_internal(theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)

        with self.assertRaises(TypeError):
            theta = jnp.array(["theta"])
            _perfilter_internal(theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)

        with self.assertRaises(IndexError):
            # zero-dimeansional array
            theta = jnp.array(5)
            _perfilter_internal(theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)

    def test_wrongtype_sigmas(self):
        with self.assertRaises(TypeError):
            sigmas = jnp.array(10, 20)
            _perfilter_internal(self.theta, self.ys, self.J, sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            sigmas = "sigmas"
            _perfilter_internal(self.theta, self.ys, self.J, sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)

    def test_wrongtype_rinit(self):
        def onestep(theta, J, covars=None):
            raise RuntimeError("boink")

        rinit = onestep

        with self.assertRaises(RuntimeError) as cm:
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit="rinit", rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

    def test_wrongtype_rprocesses(self):
        def onestep(state, theta, key, covars=None):
            raise RuntimeError("boink")

        rprocesses = onestep

        with self.assertRaises(RuntimeError) as cm:
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, rprocesses, self.dmeasures,
                               self.ndim, self.covars, key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses="rprocess",
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

        # wrongly use rprocess as rprocesses
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocess,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

    def test_wrongtype_dmeasures(self):
        def onestep(y, preds, theta):
            raise RuntimeError("boink")

        dmeasures = onestep

        with self.assertRaises(RuntimeError) as cm:
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, dmeasures,
                               self.ndim, self.covars, key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures="dmeasure", ndim=self.ndim, covars=self.covars, key=self.key)

    def test_wrongtype_thresh(self):
        thresh = "-0.0001"
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, thresh=thresh,
                               key=self.key)

        with self.assertRaises(TypeError):
            thresh = jnp.array([0.97, 0.97])
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, thresh=thresh,
                               key=self.key)

    # inappropriate values
    def test_invalid_J(self):
        with self.assertRaises(IndexError):
            J = 0
            _perfilter_internal(self.theta, self.ys, J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            J = -1
            _perfilter_internal(self.theta, self.ys, J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

    def test_invalid_thresh(self):
        thresh = jnp.inf
        value1, theta1 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit,
                                            rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim,
                                            covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value1.dtype, jnp.float32)
        self.assertEqual(value1.shape, ())
        self.assertTrue(jnp.isfinite(value1.item()))
        self.assertEqual(theta1.shape, (self.J, 16))

        thresh = -jnp.inf
        value2, theta2 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit,
                                            rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim,
                                            covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value2.dtype, jnp.float32)
        self.assertEqual(value2.shape, ())
        self.assertTrue(jnp.isfinite(value2.item()))
        self.assertEqual(theta2.shape, (self.J, 16))

    def test_new_arg(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, alpha=0.9, thresh=10,
                               key=self.key)

    def test_mean(self):
        result, theta = _perfilter_internal_mean(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                                self.dmeasures, self.ndim, self.covars, thresh=100, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(theta.shape, (self.J, 16))


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

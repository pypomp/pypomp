import os
import jax
import sys
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.internal_functions import _mop_internal
from pypomp.internal_functions import _mop_internal_mean

#curr_dir = os.getcwd()
#sys.path.append(os.path.abspath(os.path.join(curr_dir, "..", "pypomp")))
sys.path.insert(0, 'pypomp')
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

class TestMopInternal_LG(unittest.TestCase):
    def setUp(self):
        self.J = 10
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

    def test_basic(self):
        result1 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                alpha=0.97, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                alpha=0.97, key=self.key)

        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(result1, result2)

        result3 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=1, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)

        result4 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=1, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        self.assertEqual(result3, result4)

    def test_edge_J(self):
        result1 = _mop_internal(self.theta, self.ys, 1, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=0.97, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)

    def test_edge_alpha(self):
        result1 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=0, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=1, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)

    def test_small_alpha(self):
        alpha = 1e-10
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                              alpha=alpha, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)

    def test_edge_ys(self):
        # when len(ys) = 1
        ys = self.ys[0, :]
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                              alpha=0.97, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)

    def test_dmeasure_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, dmeasure, self.covars, alpha=0.97,
                              key=self.key)
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(result.shape, ())
        self.assertFalse(jnp.isfinite(result.item()))

    def test_zero_dmeasure(self):
        def zero_dmeasure(ys, particlesP, theta):
            return jnp.zeros((particlesP.shape[0],))

        result = _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                              dmeasure=zero_dmeasure, covars=self.covars, alpha=0.97, key=self.key)
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 0.0)

    def test_rprocess_inf(self):
        def custom_rproc(state, theta, key, covars=None):
            # override the state variable
            state = jnp.array([-jnp.inf, -jnp.inf])
            A, C, Q, R = get_thetas(theta)
            key, subkey = jax.random.split(key)
            return jax.random.multivariate_normal(key=subkey,
                                                  mean=A @ state, cov=Q)

        rprocess = jax.vmap(custom_rproc, (0, None, 0, None))
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, rprocess, self.dmeasure, self.covars, alpha=0.97,
                              key=self.key)
        self.assertTrue(jnp.isnan(result).item())

    # error handling - missing paramters - theta, ys, J, rinit, rprocess, dmeasure
    def test_missing(self):
        with self.assertRaises(TypeError):
            _mop_internal()
        with self.assertRaises(TypeError):
            _mop_internal(self.theta)
        with self.assertRaises(TypeError):
            _mop_internal(self.ys)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.rinit)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.rinit, self.rprocess)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.rinit, self.dmeasure)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.rprocess, self.dmeasure)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rinit)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rinit, self.dmeasure)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rprocess, self.dmeasure)

    def test_missing_theta(self):
        with self.assertRaises(TypeError):
            _mop_internal(None, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)
        with self.assertRaises(TypeError):
            _mop_internal(None, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars)

    def test_missing_ys(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, None, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, None, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars)

    def test_missing_J(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, None, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, None, self.rinit, self.rprocess, self.dmeasure, self.covars)

    def test_missing_rinit(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, None, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, None, self.rprocess, self.dmeasure, self.covars)

    def test_missing_rprocess(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rinit, None, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rinit, None, self.dmeasure, self.covars)

    def test_missing_dmeasure(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, None, self.covars, alpha=0.97,
                         key=self.key)

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, None, self.covars)

    # error handling - wrong paramter type - theta, ys, J, rinit, rprocess, dmeasure
    def test_wrongtype_J(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, J=jnp.array(10, 20), rinit=self.rinit, rprocess=self.rprocess,
                          dmeasure=self.dmeasure, covars=self.covars)

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, J="pop", rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars)

        def generate_J(n):
            return jnp.array(10, 20)

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, J=lambda n: generate_J(n), rinit=self.rinit, rprocess=self.rprocess,
                         dmeasure=self.dmeasure, covars=self.covars)

    def test_wrongtype_theta(self):
        with self.assertRaises(TypeError):
            theta = "theta"
            _mop_internal(theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)

    def test_wrongtype_rinit1(self):
        def onestep(theta, J, covars=None):
            raise RuntimeError("boink")

        rinit = onestep

        with self.assertRaises(RuntimeError) as cm:
            _mop_internal(self.theta, self.ys, self.J, rinit, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, rinit="rinit", rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars, alpha=0.97, key=self.key)

    def test_wrongtype_rprocess(self):
        def onestep(state, theta, key, covars=None):
            raise RuntimeError("boink")

        rprocess = onestep

        with self.assertRaises(RuntimeError) as cm:
            _mop_internal(self.theta, self.ys, self.J, self.rinit, rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess="rprocess", dmeasure=self.dmeasure,
                         covars=self.covars, alpha=0.97, key=self.key)

    def test_wrongtype_dmeasure(self):
        def onestep(y, preds, theta):
            raise RuntimeError("boink")

        dmeasure = onestep

        with self.assertRaises(RuntimeError) as cm:
            _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, dmeasure, self.covars, alpha=0.97,
                         key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure="dmeasure",
                         covars=self.covars, alpha=0.97, key=self.key)

    def test_wrongtype_alpha(self):
        alpha = "-0.0001"
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars, alpha=alpha, key=self.key)

        with self.assertRaises(TypeError):
            alpha = jnp.array([0.97, 0.97])
            _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars, alpha=alpha, key=self.key)

    # using inappropriate value

    def test_invalid_J(self):
        with self.assertRaises(TypeError):
            J = 0
            _mop_internal(self.theta, self.ys, J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars, alpha=0.97, key=self.key)

        with self.assertRaises(TypeError):
            J = -1
            _mop_internal(self.theta, self.ys, J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars, alpha=0.97, key=self.key)

    def test_invalid_alpha1(self):
        alpha = jnp.inf
        value = _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, alpha=alpha, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertFalse(jnp.isfinite(value.item()))

    def test_invalid_alpha2(self):
        alpha = -jnp.inf
        value = _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, alpha=alpha, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertFalse(jnp.isfinite(value.item()))

    def test_new_arg(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars, a=0.9, alpha=0.97, key=self.key)

    def test_mean(self):
        result = _mop_internal_mean(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                   dmeasure=self.dmeasure, covars=self.covars, alpha=0.97, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)

        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

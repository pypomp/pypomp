import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import *
from pypomp.internal_functions import _pfilter_pf_internal

def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return A, C, Q, R


def transform_thetas(A, C, Q, R):
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])


LG_obj, ys, theta, covars, rinit, rproc, dmeas, rprocess, dmeasure, rprocesses, dmeasures = LG_internal()

class TestPfilterPfInternal_LG(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(111)
        self.J = 5
        self.ys = ys
        self.theta = theta
        self.covars = covars

        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rprocess = rprocess
        self.dmeasure = dmeasure
        self.rprocesses = rprocesses
        self.dmeasures = dmeasures

    def test_basic(self):
        result1 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=100, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(result1, result2)

        result3 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=10, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        result4 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=10, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        self.assertEqual(result3, result4)

        result5 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=-1, key=self.key)
        self.assertEqual(result5.shape, ())
        self.assertTrue(jnp.isfinite(result5.item()))
        self.assertEqual(result5.dtype, jnp.float32)
        result6 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=-1, key=self.key)
        self.assertEqual(result6.shape, ())
        self.assertTrue(jnp.isfinite(result6.item()))
        self.assertEqual(result6.dtype, jnp.float32)
        self.assertEqual(result5, result6)

    def test_edge_J(self):
        result1 = _pfilter_pf_internal(self.theta, self.ys, 1, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                      key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _pfilter_pf_internal(self.theta, self.ys, 1, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                      thresh=-1, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        result3 = _pfilter_pf_internal(self.theta, self.ys, 100, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                      key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        result4 = _pfilter_pf_internal(self.theta, self.ys, 100, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                      thresh=-1, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)

    def test_edge_thresh(self):
        result1 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=0, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=-10, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(result1, result2)

        result3 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=10000, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        self.assertNotEqual(result1, result3)

        result4 = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                      self.covars, thresh=-1000, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        self.assertEqual(result1, result4)

    def test_edge_ys(self):
        # when len(ys) = 1
        ys = self.ys[0, :]
        result = _pfilter_pf_internal(self.theta, ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                     key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)

    def test_dmeasure_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        result = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, dmeasure, self.covars,
                                     key=self.key)
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(result.shape, ())
        self.assertFalse(jnp.isfinite(result.item()))

    def test_zero_dmeasure(self):
        def zero_dmeasure(ys, particlesP, theta):
            return jnp.zeros((particlesP.shape[0],))

        result = _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                     dmeasure=zero_dmeasure, covars=self.covars, key=self.key)
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
        result = _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, rprocess, self.dmeasure, self.covars,
                                     key=self.key)
        self.assertTrue(jnp.isnan(result).item())

    def test_missing(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.ys, key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.rinit,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.rprocess,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.dmeasure,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.rinit, self.rprocess,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.rinit, self.dmeasure,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.rprocess, self.dmeasure,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.dmeasure,key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rprocess, self.dmeasure,key=self.key)

    def test_missing_theta(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(None, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                                key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(None, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,key=self.key)

    def test_missing_ys(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, None, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, None, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,key=self.key)

    def test_missing_J(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, None, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                thresh=-1, key=self.key)

        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, None, self.rinit, self.rprocess, self.dmeasure, self.covars,key=self.key)

    def test_missing_rinit(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, None, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                                key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, None, self.rprocess, self.dmeasure, self.covars,key=self.key)

    def test_missing_rprocess(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, None, self.dmeasure, self.covars, thresh=-1,
                                key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, None, self.dmeasure, self.covars,key=self.key)

    def test_missing_dmeasure(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, None, self.covars, thresh=-1,
                                key=self.key)

        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, None, self.covars,key=self.key)

    # error handling - wrong paramter type - theta, ys, J, rinit, rprocess, dmeasure
    def test_wrongtype_J(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, J=jnp.array(10, 20), rinit=self.rinit, rprocess=self.rprocess,
                                 dmeasure=self.dmeasure, covars=self.covars,key=self.key)

        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, J="pop", rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars,key=self.key)

        def generate_J(n):
            return jnp.array(10, 20)

        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, J=lambda n: generate_J(n), rinit=self.rinit,
                                rprocess=self.rprocess, dmeasure=self.dmeasure, covars=self.covars,key=self.key)

    def test_wrongtype_theta(self):
        with self.assertRaises(TypeError):
            theta = "theta"
            _pfilter_pf_internal(theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            theta = jnp.array(["theta"])
            _pfilter_pf_internal(theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                thresh=-1, key=self.key)
        with self.assertRaises(IndexError):
            theta = jnp.array(5)
            _pfilter_pf_internal(theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                thresh=-1, key=self.key)

    def test_wrongtype_rinit(self):
        def onestep(theta, J, covars=None):
            raise RuntimeError("boink")

        rinit = onestep

        with self.assertRaises(RuntimeError) as cm:
            _pfilter_pf_internal(self.theta, self.ys, self.J, rinit, self.rprocess, self.dmeasure, self.covars,
                                key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, rinit="rinit", rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars, key=self.key)

    def test_wrongtype_rprocess(self):
        def onestep(state, theta, key, covars=None):
            raise RuntimeError("boink")

        rprocess = onestep

        with self.assertRaises(RuntimeError) as cm:
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, rprocess, self.dmeasure, self.covars,
                                key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess="rprocess",
                                dmeasure=self.dmeasure, covars=self.covars, key=self.key)

    def test_wrongtype_dmeasure(self):
        def onestep(y, preds, theta):
            raise RuntimeError("boink")

        dmeasure = onestep

        with self.assertRaises(RuntimeError) as cm:
            _pfilter_pf_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, dmeasure, self.covars,
                                key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure="dmeasure", covars=self.covars, key=self.key)

    def test_wrongtype_thresh(self):
        thresh = "-0.0001"
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)

        with self.assertRaises(TypeError):
            thresh = jnp.array([0.97, 0.97])
            _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)

    # using inappropriate value
    def test_invalid_J(self):
        with self.assertRaises(TypeError):
            J = 0
            _pfilter_pf_internal(self.theta, self.ys, J, rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            J = -1
            _pfilter_pf_internal(self.theta, self.ys, J, rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars, key=self.key)

    def test_invalid_ys(self):
        # ys = self.ys[0,:]
        y = jnp.full(self.ys.shape, jnp.inf)
        value = _pfilter_pf_internal(self.theta, y, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                    dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        self.assertTrue(jnp.isnan(value).item())

    def test_invalid_thresh1(self):
        thresh = jnp.inf
        value = _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                    dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertTrue(jnp.isfinite(value.item()))

    def test_invalid_thresh2(self):
        thresh = -jnp.inf
        value = _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                    dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertTrue(jnp.isfinite(value.item()))

    def test_new_arg(self):
        with self.assertRaises(TypeError):
            _pfilter_pf_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars, alpha=0.9, thresh=10, key=self.key)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

import jax
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.internal_functions import _pfilter_internal
from pypomp.internal_functions import _pfilter_internal_mean


def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return A, C, Q, R


def transform_thetas(A, C, Q, R):
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])


class TestFitInternal_LG(unittest.TestCase):
    def setUp(self):
        fixed = False
        self.key = jax.random.PRNGKey(111)
        self.J = 10
        angle = 0.2
        angle2 = angle if fixed else -0.5
        A = jnp.array([[jnp.cos(angle2), -jnp.sin(angle)],
                       [jnp.sin(angle), jnp.cos(angle2)]])
        C = jnp.eye(2)
        Q = jnp.array([[1, 1e-4],
                       [1e-4, 1]]) / 100
        R = jnp.array([[1, .1],
                       [.1, 1]]) / 10
        self.theta = transform_thetas(A, C, Q, R)
        x = jnp.ones(2)
        xs = []
        ys = []
        T = 4
        for i in tqdm(range(T)):
            self.key, subkey = jax.random.split(self.key)
            x = jax.random.multivariate_normal(key=subkey, mean=A @ x, cov=Q)
            self.key, subkey = jax.random.split(self.key)
            y = jax.random.multivariate_normal(key=subkey, mean=C @ x, cov=R)
            xs.append(x)
            ys.append(y)
        self.xs = jnp.array(xs)
        self.ys = jnp.array(ys)
        self.covars = None

        def custom_rinit(theta, J, covars=None):
            return jnp.ones((J, 2))

        def custom_rproc(state, theta, key, covars=None):
            A, C, Q, R = get_thetas(theta)
            key, subkey = jax.random.split(key)
            return jax.random.multivariate_normal(key=subkey,
                                                  mean=A @ state, cov=Q)

        def custom_dmeas(y, preds, theta):
            A, C, Q, R = get_thetas(theta)
            return jax.scipy.stats.multivariate_normal.logpdf(y, preds, R)

        self.rinit = custom_rinit
        self.rproc = custom_rproc
        self.dmeas = custom_dmeas
        self.rprocess = jax.vmap(custom_rproc, (0, None, 0, None))
        self.dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        self.rprocesses = jax.vmap(custom_rproc, (0, 0, 0, None))
        self.dmeasures = jax.vmap(custom_dmeas, (None, 0, 0))

    # _pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars = None, thresh = 100, key = None):
    def test_basic(self):
        result1 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=100, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        # test result1 and result2 are the same
        self.assertEqual(result1, result2)

        result3 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=10, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        result4 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=10, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        # test result3 and result4 are the same
        self.assertEqual(result3, result4)

        result5 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=-1, key=self.key)
        self.assertEqual(result5.shape, ())
        self.assertTrue(jnp.isfinite(result5.item()))
        self.assertEqual(result5.dtype, jnp.float32)
        result6 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=-1, key=self.key)
        self.assertEqual(result6.shape, ())
        self.assertTrue(jnp.isfinite(result6.item()))
        self.assertEqual(result6.dtype, jnp.float32)
        self.assertEqual(result5, result6)

    def test_edge_J(self):
        result1 = _pfilter_internal(self.theta, self.ys, 1, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _pfilter_internal(self.theta, self.ys, 1, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=-1, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        result3 = _pfilter_internal(self.theta, self.ys, 100, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        result4 = _pfilter_internal(self.theta, self.ys, 100, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=-1, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)

    def test_edge_thresh(self):
        result1 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=0, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=-10, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(result1, result2)

        result3 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=10000, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(jnp.isfinite(result3.item()))
        self.assertEqual(result3.dtype, jnp.float32)
        self.assertNotEqual(result1, result3)

        result4 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh=-1000, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(jnp.isfinite(result4.item()))
        self.assertEqual(result4.dtype, jnp.float32)
        self.assertEqual(result1, result4)

    def test_edge_ys(self):
        # when len(ys) = 1
        ys = self.ys[0, :]
        result = _pfilter_internal(self.theta, ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                  key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result.item()))
        self.assertEqual(result.dtype, jnp.float32)

    def test_dmeasure_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        result = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, dmeasure, self.covars,
                                  key=self.key)
        self.assertEqual(result.dtype, jnp.float32)
        self.assertEqual(result.shape, ())
        self.assertFalse(jnp.isfinite(result.item()))

    def test_zero_dmeasure(self):
        def zero_dmeasure(ys, particlesP, theta):
            return jnp.zeros((particlesP.shape[0],))

        result = _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
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
        result = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, rprocess, self.dmeasure, self.covars,
                                  key=self.key)
        self.assertTrue(jnp.isnan(result).item())

    def test_missing(self):
        with self.assertRaises(TypeError):
            _pfilter_internal()
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.ys)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.rinit)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.rprocess)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.dmeasure)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.rinit, self.rprocess)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.rinit, self.dmeasure)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.rprocess, self.dmeasure)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.dmeasure)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rprocess, self.dmeasure)

    def test_missing_theta(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(None, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                             key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_internal(None, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars)

    def test_missing_ys(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, None, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                             key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, None, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars)

    def test_missing_J(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, None, self.rinit, self.rprocess, self.dmeasure, self.covars,
                             thresh=-1, key=self.key)

        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, None, self.rinit, self.rprocess, self.dmeasure, self.covars)

    def test_missing_rinit(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, None, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                             key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, None, self.rprocess, self.dmeasure, self.covars)

    def test_missing_rprocess(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, None, self.dmeasure, self.covars, thresh=-1,
                             key=self.key)
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, None, self.dmeasure, self.covars)

    def test_missing_dmeasure(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, None, self.covars, thresh=-1,
                             key=self.key)

        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, None, self.covars)

    # error handling - wrong paramter type - theta, ys, J, rinit, rprocess, dmeasure, (alpha is optional, key is optional)
    def test_wrongtype_J(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, J=jnp.array(10, 20), rinit=self.rinit, rprocess=self.rprocess,
                              dmeasure=self.dmeasure, covars=self.covars)

        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, J="pop", rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars)

        def generate_J(n):
            return jnp.array(10, 20)

        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, J=lambda n: generate_J(n), rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars)

    def test_wrongtype_theta(self):
        with self.assertRaises(TypeError):
            theta = "theta"
            _pfilter_internal(theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                             key=self.key)
        with self.assertRaises(TypeError):
            theta = jnp.array(["theta"])
            _pfilter_internal(theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                             key=self.key)
        with self.assertRaises(IndexError):
            theta = jnp.array(5)
            _pfilter_internal(theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh=-1,
                             key=self.key)

    def test_wrongtype_rinit(self):
        def onestep(theta, J, covars=None):
            raise RuntimeError("boink")

        rinit = onestep

        with self.assertRaises(RuntimeError) as cm:
            _pfilter_internal(self.theta, self.ys, self.J, rinit, self.rprocess, self.dmeasure, self.covars,
                             key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, rinit="rinit", rprocess=self.rprocess, dmeasure=self.dmeasure,
                             covars=self.covars, key=self.key)

    def test_wrongtype_rprocess(self):
        def onestep(state, theta, key, covars=None):
            raise RuntimeError("boink")

        rprocess = onestep

        with self.assertRaises(RuntimeError) as cm:
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, rprocess, self.dmeasure, self.covars,
                             key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess="rprocess", dmeasure=self.dmeasure,
                             covars=self.covars, key=self.key)

    def test_wrongtype_dmeasure(self):
        def onestep(y, preds, theta):
            raise RuntimeError("boink")

        dmeasure = onestep

        with self.assertRaises(RuntimeError) as cm:
            _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, dmeasure, self.covars,
                             key=self.key)

        self.assertEqual(str(cm.exception), "boink")

        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure="dmeasure",
                             covars=self.covars, key=self.key)

    def test_wrongtype_thresh(self):
        thresh = "-0.0001"
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)

        with self.assertRaises(TypeError):
            thresh = jnp.array([0.97, 0.97])
            _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)

    # using inappropriate value

    def test_invalid_J(self):
        with self.assertRaises(TypeError):
            J = 0
            _pfilter_internal(self.theta, self.ys, J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                             covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            J = -1
            _pfilter_internal(self.theta, self.ys, J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                             covars=self.covars, key=self.key)

    def test_invalid_ys(self):
        # ys = self.ys[0,:]
        y = jnp.full(self.ys.shape, jnp.inf)
        value = _pfilter_internal(self.theta, y, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                 dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        self.assertTrue(jnp.isnan(value).item())

    def test_invalid_thresh1(self):
        thresh = jnp.inf
        value = _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                 dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertTrue(jnp.isfinite(value.item()))

    def test_invalid_thresh2(self):
        thresh = -jnp.inf
        value = _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                 dmeasure=self.dmeasure, covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertTrue(jnp.isfinite(value.item()))

    def test_new_arg(self):
        with self.assertRaises(TypeError):
            _pfilter_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, alpha=0.9, thresh=10, key=self.key)

    # test the pfilter and pfilter_mean
    def test_mean(self):
        result1 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(jnp.isfinite(result1.item()))
        self.assertEqual(result1.dtype, jnp.float32)
        result2 = _pfilter_internal_mean(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure,
                                        self.covars, thresh=100, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(jnp.isfinite(result2.item()))
        self.assertEqual(result2.dtype, jnp.float32)
        self.assertEqual(result1, result2 * len(self.ys))


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

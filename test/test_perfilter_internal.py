import jax
import unittest
import jax.numpy as np

from tqdm import tqdm
from pypomp.internal_functions import _pfilter_internal
from pypomp.internal_functions import _perfilter_internal
from pypomp.internal_functions import _perfilter_internal_mean

def get_thetas(theta):
    A = theta[0]
    C = theta[1]
    Q = theta[2]
    R = theta[3]
    return A, C, Q, R


def transform_thetas(theta):
    return np.array([A, C, Q, R])


class TestPerfilterInternal_LG(unittest.TestCase):
    def setUp(self):
        fixed = False
        self.key = jax.random.PRNGKey(111)
        self.J = 10
        self.sigmas = 0.01
        angle = 0.2
        angle2 = angle if fixed else -0.5
        A = np.array([[np.cos(angle2), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle2)]])
        C = np.eye(2)
        Q = np.array([[1, 1e-4],
                      [1e-4, 1]]) / 100
        R = np.array([[1, .1],
                      [.1, 1]]) / 10
        self.theta = np.array([A, C, Q, R])
        x = np.ones(2)
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
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.covars = None

        def custom_rinit(theta, J, covars=None):
            return np.ones((J, 2))

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
        self.ndim = self.theta.ndim

    # _perfilter_internal(theta, ys, J, sigmas, rinit, rprocesses, dmeasures, covars=None, a = 0.95, thresh=100, key=None):
    def test_basic_function(self):
        result1, theta1 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(np.isfinite(result1.item()))
        self.assertEqual(result1.dtype, np.float32)
        self.assertEqual(theta1.shape, (self.J, 4, 2, 2))
        result2, theta2 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(np.isfinite(result2.item()))
        self.assertEqual(result2.dtype, np.float32)
        self.assertEqual(theta2.shape, (self.J, 4, 2, 2))
        self.assertEqual(result1, result2)
        self.assertTrue(np.array_equal(theta1, theta2))

        result3, theta3 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=10, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(np.isfinite(result3.item()))
        self.assertEqual(result3.dtype, np.float32)
        self.assertEqual(theta3.shape, (self.J, 4, 2, 2))
        result4, theta4 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=10, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(np.isfinite(result4.item()))
        self.assertEqual(result4.dtype, np.float32)
        self.assertEqual(theta4.shape, (self.J, 4, 2, 2))
        self.assertEqual(result3, result4)
        self.assertTrue(np.array_equal(theta3, theta4))

        result5, theta5 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result5.shape, ())
        self.assertTrue(np.isfinite(result5.item()))
        self.assertEqual(result5.dtype, np.float32)
        self.assertEqual(theta5.shape, (self.J, 4, 2, 2))
        result6, theta6 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result6.shape, ())
        self.assertTrue(np.isfinite(result6.item()))
        self.assertEqual(result6.dtype, np.float32)
        self.assertEqual(theta6.shape, (self.J, 4, 2, 2))
        self.assertEqual(result5, result6)
        self.assertTrue(np.array_equal(theta5, theta6))

    def test_edge_J(self):
        result1, theta1 = _perfilter_internal(self.theta, self.ys, 1, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(np.isfinite(result1.item()))
        self.assertEqual(result1.dtype, np.float32)
        self.assertEqual(theta1.shape, (1, 4, 2, 2))
        result2, theta2 = _perfilter_internal(self.theta, self.ys, 1, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(np.isfinite(result2.item()))
        self.assertEqual(result2.dtype, np.float32)
        self.assertEqual(theta2.shape, (1, 4, 2, 2))
        result3, theta3 = _perfilter_internal(self.theta, self.ys, 100, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(np.isfinite(result3.item()))
        self.assertEqual(result3.dtype, np.float32)
        self.assertEqual(theta3.shape, (100, 4, 2, 2))
        result4, theta4 = _perfilter_internal(self.theta, self.ys, 100, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(np.isfinite(result4.item()))
        self.assertEqual(result4.dtype, np.float32)
        self.assertEqual(theta4.shape, (100, 4, 2, 2))

    def test_edge_thresh(self):
        result1, theta1 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=0, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(np.isfinite(result1.item()))
        self.assertEqual(result1.dtype, np.float32)
        self.assertEqual(theta1.shape, (self.J, 4, 2, 2))
        result2, theta2 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-10, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(np.isfinite(result2.item()))
        self.assertEqual(result2.dtype, np.float32)
        self.assertEqual(theta2.shape, (self.J, 4, 2, 2))
        self.assertEqual(result1, result2)

        result3, theta3 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=10000, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(np.isfinite(result3.item()))
        self.assertEqual(result3.dtype, np.float32)
        self.assertEqual(theta3.shape, (self.J, 4, 2, 2))

        result4, theta4 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, thresh=-1000, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(np.isfinite(result4.item()))
        self.assertEqual(result4.dtype, np.float32)
        self.assertEqual(theta4.shape, (self.J, 4, 2, 2))
        self.assertEqual(result1, result4)

    def test_edge_ys(self):
        # when len(ys) = 1
        ys = self.ys[0, :]
        result, theta = _perfilter_internal(self.theta, ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                           self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.isfinite(result.item()))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(theta.shape, (self.J, 4, 2, 2))

    def test_edge_sigmas(self):
        # when len(ys) = 1
        sigmas = 0
        result, theta = _perfilter_internal(self.theta, self.ys, self.J, sigmas, self.rinit, self.rprocesses,
                                           self.dmeasures, self.ndim, self.covars, key=self.key)
        result2 = _pfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.isfinite(result.item()))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(theta.shape, (self.J, 4, 2, 2))
        self.assertEqual(result, result2)

        sigmas = -0.001
        result3, theta3 = _perfilter_internal(self.theta, self.ys, self.J, sigmas, self.rinit, self.rprocesses,
                                             self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(np.isfinite(result3.item()))
        self.assertEqual(result3.dtype, np.float32)
        self.assertTrue(theta.shape, (self.J, 4, 2, 2))
        self.assertNotEqual(result3, result2)

    def test_dmeasures_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasures = jax.vmap(custom_dmeas, (None, 0, 0))
        result, theta = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                           dmeasures, self.ndim, self.covars, key=self.key)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, ())
        self.assertFalse(np.isfinite(result.item()))
        self.assertEqual(theta.shape, (self.J, 4, 2, 2))

    def test_zero_dmeasures(self):
        def zero_dmeasures(y, particlesP, thetas):
            return np.zeros((particlesP.shape[0],))

        result, theta = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                           zero_dmeasures, self.ndim, covars=self.covars, key=self.key)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 0.0)
        self.assertEqual(theta.shape, (self.J, 4, 2, 2))

    def test_rprocess_inf(self):
        def custom_rproc(state, theta, key, covars=None):
            # override the state variable
            state = np.array([-np.inf, -np.inf])
            A, C, Q, R = get_thetas(theta)
            key, subkey = jax.random.split(key)
            return jax.random.multivariate_normal(key=subkey,
                                                  mean=A @ state, cov=Q)

        rprocesses = jax.vmap(custom_rproc, (0, 0, 0, None))
        result, theta = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, rprocesses,
                                           self.dmeasures, self.ndim, self.covars, key=self.key)
        self.assertFalse(np.isnan(result).item())
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, ())
        self.assertEqual(theta.shape, (self.J, 4, 2, 2))

    def test_invalid_ys(self):
        y = np.full(self.ys.shape, np.inf)
        value, theta = _perfilter_internal(self.theta, y, self.J, self.sigmas, rinit=self.rinit,
                                          rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim,
                                          covars=self.covars, key=self.key)
        self.assertFalse(np.isnan(value).item())
        self.assertEqual(value.dtype, np.float32)
        self.assertEqual(value.shape, ())
        self.assertEqual(theta.shape, (self.J, 4, 2, 2))

    def test_missing(self):
        with self.assertRaises(TypeError):
            _perfilter_internal()
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.ys)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rprocesses)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit, self.rprocesses)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rprocesses, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.rinit, self.rprocesses, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rinit)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rinit, self.rprocesses)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rinit, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.rprocesses, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rprocesses, self.dmeasures)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.ndim)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.ndim)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.ndim)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.dmeasures, self.ndim)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rprocesses, self.dmeasures, self.ndim)

    def test_missing_theta(self):
        with self.assertRaises(AttributeError):
            _perfilter_internal(None, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(AttributeError):
            _perfilter_internal(None, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars)

    def test_missing_ys(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, None, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, None, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars)

    def test_missing_J(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, None, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, None, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars)

    def test_missing_sigmas(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, None, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, None, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars)

    def test_missing_rinit(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, None, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, None, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars)

    def test_missing_rprocesses(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, None, self.dmeasures, self.ndim,
                               self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, None, self.dmeasures, self.ndim,
                               self.covars)

    def test_missing_dmeasures(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, None, self.ndim,
                               self.covars, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, None, self.ndim,
                               self.covars)

    # wrong type parameters
    def test_wrongtype_J(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, J=np.array(10, 20), sigmas=self.sigmas, rinit=self.rinit,
                               rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars)

        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, J="pop", sigmas=self.sigmas, rinit=self.rinit,
                               rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars)

        def generate_J(n):
            return np.array(10, 20)

        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, J=lambda n: generate_J(n), sigmas=self.sigmas, rinit=self.rinit,
                               rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars)

    def test_wrongtype_theta(self):
        with self.assertRaises(TypeError):
            theta = "theta"
            _perfilter_internal(theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)

        with self.assertRaises(TypeError):
            theta = np.array(["theta"])
            _perfilter_internal(theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)

        with self.assertRaises(IndexError):
            # zero-dimeansional array
            theta = np.array(5)
            _perfilter_internal(theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses, self.dmeasures,
                               self.ndim, self.covars, thresh=-1, key=self.key)

    def test_wrongtype_sigmas(self):
        with self.assertRaises(TypeError):
            sigmas = np.array(10, 20)
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
            thresh = np.array([0.97, 0.97])
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, thresh=thresh,
                               key=self.key)

    # inappropriate values
    def test_invalid_J(self):
        with self.assertRaises(IndexError):
            J = 0
            _perfilter_internal(self.theta, self.ys, J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

        with self.assertRaises(ValueError):
            J = -1
            _perfilter_internal(self.theta, self.ys, J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, key=self.key)

    def test_invalid_thresh(self):
        thresh = np.inf
        value1, theta1 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit,
                                            rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim,
                                            covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value1.dtype, np.float32)
        self.assertEqual(value1.shape, ())
        self.assertTrue(np.isfinite(value1.item()))
        self.assertEqual(theta1.shape, (self.J, 4, 2, 2))

        thresh = -np.inf
        value2, theta2 = _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit,
                                            rprocesses=self.rprocesses, dmeasures=self.dmeasures, ndim=self.ndim,
                                            covars=self.covars, thresh=thresh, key=self.key)
        self.assertEqual(value2.dtype, np.float32)
        self.assertEqual(value2.shape, ())
        self.assertTrue(np.isfinite(value2.item()))
        self.assertEqual(theta2.shape, (self.J, 4, 2, 2))

    def test_new_arg(self):
        with self.assertRaises(TypeError):
            _perfilter_internal(self.theta, self.ys, self.J, self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                               dmeasures=self.dmeasures, ndim=self.ndim, covars=self.covars, alpha=0.9, thresh=10,
                               key=self.key)

    def test_mean(self):
        result, theta = _perfilter_internal_mean(self.theta, self.ys, self.J, self.sigmas, self.rinit, self.rprocesses,
                                                self.dmeasures, self.ndim, self.covars, thresh=100, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.isfinite(result.item()))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(theta.shape, (self.J, 4, 2, 2))


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

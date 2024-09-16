import jax
import unittest
import jax.numpy as np

from tqdm import tqdm
from pypomp.internal_functions import _mop_internal
from pypomp.internal_functions import _mop_internal_mean


def get_thetas(theta):
    A = theta[0]
    C = theta[1]
    Q = theta[2]
    R = theta[3]
    return A, C, Q, R


def transform_thetas(theta):
    return np.array([A, C, Q, R])


class TestMopInternal_LG(unittest.TestCase):
    def setUp(self):
        fixed = False
        self.key = jax.random.PRNGKey(111)
        self.J = 3
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

    def test_basic(self):
        result1 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=0.97, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(np.isfinite(result1.item()))
        self.assertEqual(result1.dtype, np.float32)
        result2 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=0.97, key=self.key)

        self.assertEqual(result2.shape, ())
        self.assertTrue(np.isfinite(result2.item()))
        self.assertEqual(result2.dtype, np.float32)
        self.assertEqual(result1, result2)

        result3 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=1, key=self.key)
        self.assertEqual(result3.shape, ())
        self.assertTrue(np.isfinite(result3.item()))
        self.assertEqual(result3.dtype, np.float32)

        result4 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=1, key=self.key)
        self.assertEqual(result4.shape, ())
        self.assertTrue(np.isfinite(result4.item()))
        self.assertEqual(result4.dtype, np.float32)
        self.assertEqual(result3, result4)

    def test_edge_J(self):
        result1 = _mop_internal(self.theta, self.ys, 1, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=0.97, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(np.isfinite(result1.item()))
        self.assertEqual(result1.dtype, np.float32)

    def test_edge_alpha(self):
        result1 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=0, key=self.key)
        self.assertEqual(result1.shape, ())
        self.assertTrue(np.isfinite(result1.item()))
        self.assertEqual(result1.dtype, np.float32)
        result2 = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                               alpha=1, key=self.key)
        self.assertEqual(result2.shape, ())
        self.assertTrue(np.isfinite(result2.item()))
        self.assertEqual(result2.dtype, np.float32)

    def test_small_alpha(self):
        alpha = 1e-10
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                              alpha=alpha, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.isfinite(result.item()))
        self.assertEqual(result.dtype, np.float32)

    def test_edge_ys(self):
        # when len(ys) = 1
        ys = self.ys[0, :]
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                              alpha=0.97, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.isfinite(result.item()))
        self.assertEqual(result.dtype, np.float32)

    def test_dmeasure_inf(self):
        # reset dmeasure to be the function that always reture -Inf, overide the self functions
        def custom_dmeas(y, preds, theta):
            return -float('inf')

        dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, self.rprocess, dmeasure, self.covars, alpha=0.97,
                              key=self.key)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, ())
        self.assertFalse(np.isfinite(result.item()))

    def test_zero_dmeasure(self):
        def zero_dmeasure(ys, particlesP, theta):
            return np.zeros((particlesP.shape[0],))

        result = _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                              dmeasure=zero_dmeasure, covars=self.covars, alpha=0.97, key=self.key)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 0.0)

    def test_rprocess_inf(self):
        def custom_rproc(state, theta, key, covars=None):
            # override the state variable
            state = np.array([-np.inf, -np.inf])
            A, C, Q, R = get_thetas(theta)
            key, subkey = jax.random.split(key)
            return jax.random.multivariate_normal(key=subkey,
                                                  mean=A @ state, cov=Q)

        rprocess = jax.vmap(custom_rproc, (0, None, 0, None))
        result = _mop_internal(self.theta, self.ys, self.J, self.rinit, rprocess, self.dmeasure, self.covars, alpha=0.97,
                              key=self.key)
        self.assertTrue(np.isnan(result).item())

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
            _mop_internal(self.theta, self.ys, J=np.array(10, 20), rinit=self.rinit, rprocess=self.rprocess,
                         dmeasure=self.dmeasure, covars=self.covars)

        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, J="pop", rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars)

        def generate_J(n):
            return np.array(10, 20)

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
            alpha = np.array([0.97, 0.97])
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
        alpha = np.inf
        value = _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, alpha=alpha, key=self.key)
        self.assertEqual(value.dtype, np.float32)
        self.assertEqual(value.shape, ())
        self.assertFalse(np.isfinite(value.item()))

    def test_invalid_alpha2(self):
        alpha = -np.inf
        value = _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                             dmeasure=self.dmeasure, covars=self.covars, alpha=alpha, key=self.key)
        self.assertEqual(value.dtype, np.float32)
        self.assertEqual(value.shape, ())
        self.assertFalse(np.isfinite(value.item()))

    def test_new_arg(self):
        with self.assertRaises(TypeError):
            _mop_internal(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                         covars=self.covars, a=0.9, alpha=0.97, key=self.key)

    def test_mean(self):
        result = _mop_internal_mean(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                   dmeasure=self.dmeasure, covars=self.covars, alpha=0.97, key=self.key)
        self.assertEqual(result.shape, ())
        self.assertTrue(np.isfinite(result.item()))
        self.assertEqual(result.dtype, np.float32)

        self.assertEqual(result.shape, ())
        self.assertTrue(np.isfinite(result.item()))
        self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

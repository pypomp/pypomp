import jax
import unittest
import jax.numpy as np

from jax import vmap
from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.perfilter import perfilter
from pypomp.internal_functions import _perfilter_internal


def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return np.array([A, C, Q, R])

get_perthetas = vmap(get_thetas, in_axes = 0)

def transform_thetas(A, C, Q, R):
    return np.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])

class TestFitInternal_LG(unittest.TestCase):
    def setUp(self):
        fixed = False
        self.key = jax.random.PRNGKey(111)
        self.J = 10
        angle = 0.2
        angle2 = angle if fixed else -0.5
        A = np.array([[np.cos(angle2), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle2)]])
        C = np.eye(2)
        Q = np.array([[1, 1e-4],
                      [1e-4, 1]]) / 100
        R = np.array([[1, .1],
                      [.1, 1]]) / 10
        self.theta = transform_thetas(A, C, Q, R)
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
        self.sigmas = 0.02

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

    def test_internal_basic(self):
        val1, theta1 = perfilter(J=self.J, rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                 theta=self.theta, ys=self.ys, sigmas=self.sigmas, covars=self.covars, thresh=-1, 
                                 key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(np.isfinite(val1.item()))
        self.assertEqual(val1.dtype, np.float32)
        self.assertEqual(theta1.shape, (self.J, 16))
        theta1_new = get_perthetas(theta1)
        self.assertEqual(theta1_new.shape, (self.J, 4, 2, 2))

        val2, theta2 = perfilter(rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                 theta=self.theta, ys=self.ys, sigmas=self.sigmas)
        self.assertEqual(val2.shape, ())
        self.assertTrue(np.isfinite(val2.item()))
        self.assertEqual(val2.dtype, np.float32)
        self.assertEqual(theta2.shape, (50, 16))
        theta2_new = get_perthetas(theta2)
        self.assertEqual(theta2_new.shape, (50, 4, 2, 2))


    def test_class_basic(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)

        val1, theta1 = perfilter(pomp_obj, J=self.J, sigmas=self.sigmas, thresh=100)
        self.assertEqual(val1.shape, ())
        self.assertTrue(np.isfinite(val1.item()))
        self.assertEqual(val1.dtype, np.float32)
        self.assertEqual(theta1.shape, (self.J, 16))
        theta1_new = get_perthetas(theta1)
        self.assertEqual(theta1_new.shape, (self.J, 4, 2, 2))


        val2, theta2 = perfilter(pomp_obj, sigmas=self.sigmas)
        self.assertEqual(val2.shape, ())
        self.assertTrue(np.isfinite(val2.item()))
        self.assertEqual(val2.dtype, np.float32)
        self.assertEqual(theta2.shape, (50, 16))
        theta2_new = get_perthetas(theta2)
        self.assertEqual(theta2_new.shape, (50, 4, 2, 2))


        val3, theta3 = perfilter(pomp_obj, J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                                 dmeasures=self.dmeasures, theta=[], ys=[])
        self.assertEqual(val3.shape, ())
        self.assertTrue(np.isfinite(val3.item()))
        self.assertEqual(val3.dtype, np.float32)
        self.assertEqual(theta3.shape, (self.J, 16))
        theta3_new = get_perthetas(theta3)
        self.assertEqual(theta3_new.shape, (self.J, 4, 2, 2))


    def test_invalid_input(self):
        with self.assertRaises(ValueError) as text:
            perfilter()

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, thresh=-1, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, theta=self.theta, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                      dmeasures=self.dmeasures)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                      dmeasures=self.dmeasures, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocesses=self.rprocesses,
                      dmeasures=self.dmeasures, theta=self.theta)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(TypeError) as text:
            perfilter(J=self.J, sigmas=self.sigmas, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                      theta=self.theta, ys=self.ys)

        self.assertEqual(str(text.exception), "perfilter() got an unexpected keyword argument 'rprocess'")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

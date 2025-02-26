import jax
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.pfilter_pf import pfilter_pf
from pypomp.internal_functions import _pfilter_pf_internal


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
        self.theta =  transform_thetas(A, C, Q, R)
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

    def test_internal_basic(self):
        val1 = pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                          ys=self.ys, covars=self.covars, thresh=10, key=self.key)
        val2 = pfilter_pf(rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta,
                          ys=self.ys)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)

    def test_class_basic(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)
        val1 = pfilter_pf(pomp_obj, self.J, thresh=10, key=self.key)
        val2 = pfilter_pf(pomp_obj)
        val3 = pfilter_pf(pomp_obj, self.J, self.rinit, self.rprocess, self.dmeasure, theta=[], ys=[])
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)
        self.assertEqual(val3.shape, ())
        self.assertTrue(jnp.isfinite(val3.item()))
        self.assertEqual(val3.dtype, jnp.float32)

    def test_invalid_input(self):
        with self.assertRaises(ValueError) as text:
            pfilter_pf()

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, thresh=-1, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, theta=self.theta, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, ys=self.ys)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter_pf(J=self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, theta=self.theta)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

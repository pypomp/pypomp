import jax
import unittest
import jax.numpy as jnp

from tqdm import tqdm
from pypomp.pomp_class import Pomp
from pypomp.fit import fit
from pypomp.internal_functions import _fit_internal


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
        self.sigmas = 0.02

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

    def test_internal_mif_basic(self):
        mif_loglik1, mif_theta1 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                      dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                      ys=self.ys, sigmas=0.02, sigmas_init=1e-20, covars=None, M=2, a=0.9,
                                      thresh_mif=-1, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

    def test_class_mif_basic(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)

        mif_loglik1, mif_theta1 = fit(pomp_obj, J=self.J, Jh=10, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9,
                                      thresh_mif=-1, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = fit(pomp_obj, sigmas=0.02, sigmas_init=1e-20, mode="IF2")
        self.assertEqual(mif_loglik2.shape, (11,))
        self.assertEqual(mif_theta2.shape, (11, 100,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

    def test_invalid_mif_input(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)

        with self.assertRaises(ValueError) as text:
            fit(mode="IF")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, thresh_mif=-1, mode="IF")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, sigmas_init=1e-20, M=2, a=0.9, thresh_mif=-1, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, sigmas=0.02, M=2, a=0.9, thresh_mif=-1, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, M=2, a=0.9, thresh_mif=-1, mode="IF2")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=0.02, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas=0.02, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=0.02, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas_init=1e-20, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas=0.02, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, covars=None, M=2, a=0.9, thresh_mif=-1, mode="IF2")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

    def test_internal_GD_basic(self):
        ## ls = False
        # method = SGD
        GD_loglik1, GD_theta1 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="SGD", alpha=0.97,
                                    scale=True, mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

        # method = Newton
        GD_loglik3, GD_theta3 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="Newton", alpha=0.97,
                                    scale=True, mode="GD")
        self.assertEqual(GD_loglik3.shape, (3,))
        self.assertEqual(GD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta3.dtype, jnp.float32))

        # refined Newton
        GD_loglik5, GD_theta5 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="WeightedNewton",
                                    alpha=0.97, scale=True, mode="GD")
        self.assertEqual(GD_loglik5.shape, (3,))
        self.assertEqual(GD_theta5.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik5.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta5.dtype, jnp.float32))

        # BFGS
        GD_loglik7, GD_theta7 = fit(J=self.J, Jh=3, theta=self.theta, ys=self.ys, rinit=self.rinit,
                                    rprocess=self.rprocess, dmeasure=self.dmeasure, itns=2, method="BFGS", alpha=0.97,
                                    scale=True, mode="GD")
        self.assertEqual(GD_loglik7.shape, (3,))
        self.assertEqual(GD_theta7.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik7.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta7.dtype, jnp.float32))

    def test_class_GD_basic(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)

        ## ls = True
        GD_loglik1, GD_theta1 = fit(pomp_obj, J=self.J, Jh=3, itns=2, method="SGD", alpha=0.97, scale=True, ls=True,
                                    mode="GD")
        self.assertEqual(GD_loglik1.shape, (3,))
        self.assertEqual(GD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

        GD_loglik2, GD_theta2 = fit(pomp_obj, J=self.J, Jh=3, itns=2, method="Newton", alpha=0.97, scale=True, ls=True,
                                    mode="GD")
        self.assertEqual(GD_loglik2.shape, (3,))
        self.assertEqual(GD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta2.dtype, jnp.float32))

        GD_loglik3, GD_theta3 = fit(pomp_obj, J=self.J, Jh=3, itns=2, method="WeightedNewton", alpha=0.97, scale=True,
                                    ls=True, mode="GD")
        self.assertEqual(GD_loglik3.shape, (3,))
        self.assertEqual(GD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta3.dtype, jnp.float32))

        GD_loglik4, GD_theta4 = fit(pomp_obj, J=self.J, Jh=3, itns=2, method="BFGS", scale=True, ls=True, mode="GD")
        self.assertEqual(GD_loglik4.shape, (3,))
        self.assertEqual(GD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(GD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(GD_theta4.dtype, jnp.float32))

    def test_invalid_GD_input(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)

        with self.assertRaises(ValueError) as text:
            fit(mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(mode="GD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(rinit=self.rinit, rprocesses=self.rprocesses, dmeasures=self.dmeasures, theta=self.theta, ys=self.ys,
                mode="GD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, Jh=10, itns=2, alpha=0.97, scale=True, mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, Jh=10, itns=2, alpha=0.97, scale=True, mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, itns=2, alpha=0.97, scale=True, mode="SGD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

    def test_internal_IFAD_basic(self):
        IFAD_loglik1, IFAD_theta1 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD",
                                        itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (3,))
        self.assertEqual(IFAD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

        IFAD_loglik2, IFAD_theta2 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="Newton",
                                        itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik2.shape, (3,))
        self.assertEqual(IFAD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta2.dtype, jnp.float32))

        IFAD_loglik3, IFAD_theta3 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9,
                                        method="WeightedNewton", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik3.shape, (3,))
        self.assertEqual(IFAD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta3.dtype, jnp.float32))

        IFAD_loglik4, IFAD_theta4 = fit(J=self.J, Jh=3, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess,
                                        dmeasure=self.dmeasure, rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                                        ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="BFGS",
                                        itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik4.shape, (3,))
        self.assertEqual(IFAD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta4.dtype, jnp.float32))

    def test_class_IFAD_basic(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)

        IFAD_loglik1, IFAD_theta1 = fit(pomp_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2, 
                                        a=0.9, method="SGD", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (2,))
        self.assertEqual(IFAD_theta1.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

        IFAD_loglik2, IFAD_theta2 = fit(pomp_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2, 
                                        a=0.9, method="Newton", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik2.shape, (2,))
        self.assertEqual(IFAD_theta2.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta2.dtype, jnp.float32))

        IFAD_loglik3, IFAD_theta3 = fit(pomp_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2,
                                        a=0.9, method="WeightedNewton", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik3.shape, (2,))
        self.assertEqual(IFAD_theta3.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta3.dtype, jnp.float32))

        IFAD_loglik4, IFAD_theta4 = fit(pomp_obj, J=self.J, Jh=3, sigmas=self.sigmas, sigmas_init=1e-20, M=2, 
                                        a=0.9, method="BFGS", itns=1, alpha=0.97, mode="IFAD")
        self.assertEqual(IFAD_loglik4.shape, (2,))
        self.assertEqual(IFAD_theta4.shape, (2,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta4.dtype, jnp.float32))

    def test_invalid_IFAD_input(self):
        pomp_obj = Pomp(self.rinit, self.rproc, self.dmeas, self.ys, self.theta, self.covars)

        with self.assertRaises(ValueError) as text:
            fit(mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(mode="AD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, alpha=0.97,
                mode="AD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, sigmas=self.sigmas, M=2, a=0.9, method="SGD", itns=2, alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(pomp_obj, J=self.J, M=2, a=0.9, method="SGD", itns=2, alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing sigmas or sigmas_init")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, ls=True,
                alpha=0.97, mode="AD")

        self.assertEqual(str(text.exception), "Invalid Mode Input")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="SGD", itns=2, ls=True,
                alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas_init=1e-20, M=2, a=0.9, method="Newton", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, sigmas=self.sigmas, M=2, a=0.9, method="WeightedNewton", itns=2, ls=True, alpha=0.97,
                mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                rprocesses=self.rprocesses, dmeasures=self.dmeasures,
                ys=self.ys, M=2, a=0.9, method="BFGS", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                ys=self.ys, M=2, a=0.9, method="SGD", itns=2, ls=True, alpha=0.97, mode="IFAD")
        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing workhorse or sigmas")

        with self.assertRaises(ValueError) as text:
            fit(J=self.J, Jh=10, theta=self.theta, rinit=self.rinit, rprocesses=self.rprocesses,
                dmeasures=self.dmeasures,
                ys=self.ys, sigmas=self.sigmas, sigmas_init=1e-20, M=2, a=0.9, method="Newton", itns=2, ls=True,
                alpha=0.97, mode="IFAD")

        self.assertEqual(str(text.exception), "Invalid Argument Input with Missing Required Argument")


if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)

import jax
import unittest
import jax.numpy as np

from tqdm import tqdm
from pypomp.pomp_class import *


def get_thetas(theta):
    A = theta[0]
    C = theta[1]
    Q = theta[2]
    R = theta[3]
    return A, C, Q, R


def transform_thetas(theta):
    return np.array([A, C, Q, R])


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


class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        fixed = False
        self.key = jax.random.PRNGKey(111)
        self.J = 3
        angle = 0.2
        angle2 = angle if fixed else -0.5
        self.sigmas = 0.01
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

        self.rinit = custom_rinit
        self.rproc = custom_rproc
        self.dmeas = custom_dmeas
        self.rprocess = jax.vmap(custom_rproc, (0, None, 0, None))
        self.dmeasure = jax.vmap(custom_dmeas, (None, 0, None))
        self.rprocesses = jax.vmap(custom_rproc, (0, 0, 0, None))
        self.dmeasures = jax.vmap(custom_dmeas, (None, 0, 0))

    def test_basic_initialization(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        self.assertEqual(pomp_obj.covars, self.covars)
        obj_ys = pomp_obj.ys
        self.assertTrue(np.array_equal(obj_ys, self.ys))
        self.assertTrue(np.array_equal(pomp_obj.theta, self.theta))

        pomp_obj2 = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)
        self.assertEqual(pomp_obj2.covars, self.covars)
        self.assertTrue(np.array_equal(pomp_obj2.ys, self.ys))
        self.assertTrue(np.array_equal(pomp_obj2.theta, self.theta))

    def test_invalid_initialization(self):
        # missing parameters
        with self.assertRaises(TypeError):
            Pomp(None, custom_rproc, custom_dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(custom_rinit, None, custom_dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(custom_rinit, custom_rproc, None, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(custom_rinit, custom_rproc, custom_dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            Pomp(custom_rinit, custom_rproc, custom_dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            Pomp(custom_rinit, custom_rproc, custom_dmeas, self.theta, self.covars)

    def test_mop_valid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        mop_obj = pomp_obj.mop(self.J, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj.shape, ())
        self.assertTrue(np.isfinite(mop_obj.item()))
        self.assertEqual(mop_obj.dtype, np.float32)

        mop_obj_edge = pomp_obj.mop(1, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj_edge.shape, ())
        self.assertTrue(np.isfinite(mop_obj_edge.item()))
        self.assertEqual(mop_obj_edge.dtype, np.float32)

        # test mean
        mop_obj_mean = pomp_obj.mop_mean(self.J, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj_mean.shape, ())
        self.assertTrue(np.isfinite(mop_obj_mean.item()))
        self.assertEqual(mop_obj_mean.dtype, np.float32)

    def test_mop_invalid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        # missing values
        with self.assertRaises(TypeError):
            pomp_obj.mop(alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.mop(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            pomp_obj.mop(0, alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.mop(-1, alpha=0.97, key=self.key)
        with self.assertRaises(ValueError):
            pomp_obj.mop(np.array([10, 20]), alpha=0.97, key=self.key)

        value = pomp_obj.mop(self.J, alpha=np.inf, key=self.key)
        self.assertEqual(value.dtype, np.float32)
        self.assertEqual(value.shape, ())
        self.assertFalse(np.isfinite(value.item()))

        # undefined argument
        with self.assertRaises(TypeError):
            pomp_obj.mop(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            pomp_obj.mop(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha=0.97,
                         key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.mop(self.J, self.rinit, self.rprocess, self.dmeasure, alpha=0.97, key=self.key)

        ### mop_mean
        with self.assertRaises(TypeError):
            pomp_obj.mop_mean(alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.mop_mean(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            pomp_obj.mop_mean(0, alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.mop_mean(-1, alpha=0.97, key=self.key)
        with self.assertRaises(ValueError):
            pomp_obj.mop_mean(np.array([10, 20]), alpha=0.97, key=self.key)

        value_mean = pomp_obj.mop_mean(self.J, alpha=np.inf, key=self.key)
        self.assertEqual(value_mean.dtype, np.float32)
        self.assertEqual(value_mean.shape, ())
        self.assertFalse(np.isfinite(value_mean.item()))

        # undefined argument
        with self.assertRaises(TypeError):
            pomp_obj.mop_mean(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            pomp_obj.mop_mean(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                              alpha=0.97, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.mop_mean(self.J, self.rinit, self.rprocess, self.dmeasure, alpha=0.97, key=self.key)

    # pfilter
    def test_pfilter_valid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        pfilter_obj = pomp_obj.pfilter(self.J, thresh=-1, key=self.key)
        self.assertEqual(pfilter_obj.shape, ())
        self.assertTrue(np.isfinite(pfilter_obj.item()))
        self.assertEqual(pfilter_obj.dtype, np.float32)

        pfilter_obj_edge = pomp_obj.pfilter(1, thresh=10, key=self.key)
        self.assertEqual(pfilter_obj_edge.shape, ())
        self.assertTrue(np.isfinite(pfilter_obj_edge.item()))
        self.assertEqual(pfilter_obj_edge.dtype, np.float32)

        pfilter_obj_mean = pomp_obj.pfilter_mean(self.J, thresh=-1, key=self.key)
        self.assertEqual(pfilter_obj_mean.shape, ())
        self.assertTrue(np.isfinite(pfilter_obj_mean.item()))
        self.assertEqual(pfilter_obj_mean.dtype, np.float32)

    def test_pfilter_invalid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        # missing values
        with self.assertRaises(TypeError):
            pomp_obj.pfilter(thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.pfilter(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            pomp_obj.pfilter(0, thresh=100, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.pfilter(-1, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            pomp_obj.pfilter(np.array([10, 20]), key=self.key)
        # undefined argument
        with self.assertRaises(TypeError):
            pomp_obj.pfilter(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            pomp_obj.pfilter(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                             key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.pfilter(self.J, self.rinit, self.rprocess, self.dmeasure, key=self.key)

        ### pfilter_mean
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_mean(thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_mean(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_mean(0, thresh=100, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_mean(-1, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            pomp_obj.pfilter_mean(np.array([10, 20]), key=self.key)
        # undefined argument
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_mean(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_mean(self.theta, self.ys, self.J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                  key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.pfilter_mean(self.J, self.rinit, self.rprocess, self.dmeasure, key=self.key)

    def test_perfilter_valid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        val1, theta1 = pomp_obj.perfilter(self.J, self.sigmas, key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(np.isfinite(val1.item()))
        self.assertEqual(val1.dtype, np.float32)
        self.assertEqual(theta1.shape, (self.J, 4, 2, 2))

        val2, theta2 = pomp_obj.perfilter(1, 0, thresh=10, key=self.key)
        self.assertEqual(val2.shape, ())
        self.assertTrue(np.isfinite(val2.item()))
        self.assertEqual(val2.dtype, np.float32)
        self.assertEqual(theta2.shape, (1, 4, 2, 2))

        val3, theta3 = pomp_obj.perfilter_mean(self.J, self.sigmas, key=self.key)
        self.assertEqual(val3.shape, ())
        self.assertTrue(np.isfinite(val3.item()))
        self.assertEqual(val3.dtype, np.float32)
        self.assertEqual(theta3.shape, (self.J, 4, 2, 2))

        val4, theta4 = pomp_obj.perfilter_mean(1, 0, thresh=10, key=self.key)
        self.assertEqual(val4.shape, ())
        self.assertTrue(np.isfinite(val4.item()))
        self.assertEqual(val4.dtype, np.float32)
        self.assertEqual(theta4.shape, (1, 4, 2, 2))

    def test_perfilter_invalid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        # missing values
        with self.assertRaises(TypeError):
            pomp_obj.perfilter(sigmas=self.sigmas, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.perfilter(key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.perfilter(self.J, key=self.key)

        with self.assertRaises(IndexError):
            pomp_obj.perfilter(0, self.sigmas, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            pomp_obj.perfilter(-1, self.sigmas, thresh=100, key=self.key)

        # undefined arg
        with self.assertRaises(TypeError):
            pomp_obj.perfilter(self.J, self.sigmas, alpha=0.97, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.perfilter(self.theta, self.ys, self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess,
                               dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.perfilter(self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure,
                               key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.perfilter_mean(sigmas=self.sigmas, thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.perfilter_mean(key=self.key)

        with self.assertRaises(IndexError):
            pomp_obj.perfilter_mean(0, self.sigmas, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            pomp_obj.perfilter_mean(-1, self.sigmas, thresh=100, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.perfilter_mean(self.J, self.sigmas, alpha=0.97, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.perfilter_mean(self.theta, self.ys, self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess,
                                    dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.perfilter_mean(self.J, sigmas=0.02, rinit=self.rinit, rprocess=self.rprocess,
                                    dmeasure=self.dmeasure, key=self.key)

    def test_pfilter_pf_valid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        pfilter_obj = pomp_obj.pfilter_pf(self.J, thresh=-1, key=self.key)
        self.assertEqual(pfilter_obj.shape, ())
        self.assertTrue(np.isfinite(pfilter_obj.item()))
        self.assertEqual(pfilter_obj.dtype, np.float32)

        pfilter_obj_edge = pomp_obj.pfilter_pf(1, thresh=10, key=self.key)
        self.assertEqual(pfilter_obj_edge.shape, ())
        self.assertTrue(np.isfinite(pfilter_obj_edge.item()))
        self.assertEqual(pfilter_obj_edge.dtype, np.float32)

    def test_pfilter_pf_invalid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        # missing values
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_pf(thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_pf(key=self.key)
        # inapprpropriate input
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_pf(0, thresh=100, key=self.key)
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_pf(-1, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            pomp_obj.pfilter_pf(np.array([10, 20]), key=self.key)
        # undefined argument
        with self.assertRaises(TypeError):
            pomp_obj.pfilter_pf(self.J, a=0.97, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.pfilter_pf(self.theta, self.ys, self.J, rinit=self.rinit, rprocess=self.rprocess,
                                dmeasure=self.dmeasure, covars=self.covars, key=self.key)

        with self.assertRaises(TypeError):
            pomp_obj.pfilter_pf(self.J, rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, key=self.key)

    def test_fit_mif_valid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)

        mif_loglik1, mif_theta1 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(np.issubdtype(mif_loglik1.dtype, np.float32))
        self.assertTrue(np.issubdtype(mif_theta1.dtype, np.float32))

        mif_loglik2, mif_theta2 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, a=0.9, J=self.J, mode="IF2",
                                               monitor=False)
        self.assertEqual(mif_loglik2.shape, (0,))
        self.assertEqual(mif_theta2.shape, (3, self.J,) + self.theta.shape)
        self.assertTrue(np.issubdtype(mif_loglik2.dtype, np.float32))
        self.assertTrue(np.issubdtype(mif_theta2.dtype, np.float32))

        # M = 0
        mif_loglik3, mif_theta3 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=0, a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik3.shape, (1,))
        self.assertEqual(mif_theta3.shape, (1, self.J,) + self.theta.shape)
        self.assertTrue(np.issubdtype(mif_loglik3.dtype, np.float32))
        self.assertTrue(np.issubdtype(mif_theta3.dtype, np.float32))

        # M = -1
        mif_loglik4, mif_theta4 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=-1, a=0.9, J=self.J, mode="IF2")
        self.assertEqual(mif_loglik4.shape, (1,))
        self.assertEqual(mif_theta4.shape, (1, self.J,) + self.theta.shape)
        self.assertTrue(np.issubdtype(mif_loglik4.dtype, np.float32))
        self.assertTrue(np.issubdtype(mif_theta4.dtype, np.float32))

        mif_loglik5, mif_theta5 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, mode="IF2")
        self.assertEqual(mif_loglik5.shape, (11,))
        self.assertEqual(mif_theta5.shape, (11, 100,) + self.theta.shape)
        self.assertTrue(np.issubdtype(mif_loglik5.dtype, np.float32))
        self.assertTrue(np.issubdtype(mif_theta5.dtype, np.float32))

    def test_fit_mif_invalid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta, self.covars)
        # missing args
        with self.assertRaises(TypeError):
            pomp_obj.fit(mode="IF2")
        with self.assertRaises(TypeError):
            pomp_obj.fit(mode="IF")
        with self.assertRaises(TypeError):
            pomp_obj.fit(sigmas=0.02, M=1, mode="IF2")
        with self.assertRaises(TypeError):
            pomp_obj.fit(sigmas_init=1e-20, a=0.9, mode="IF2")

        # useless input
        with self.assertRaises(TypeError):
            pomp_obj.fit(rinit=self.rinit, rprocess=self.rprocess, dmeasure=self.dmeasure, rprocesses=self.rprocesses,
                         dmeasures=self.dmeasures, sigmas=0.02, sigmas_init=1e-20, mode="IF2")

    def test_fit_GD_valid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)

        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        
        for method in methods:
            with self.subTest(method=method):
                GD_loglik1, GD_theta1 = pomp_obj.fit(J=3, Jh=3, method=method, itns=1, alpha=0.97, scale=True, mode="GD")
                self.assertEqual(GD_loglik1.shape, (2,))
                self.assertEqual(GD_theta1.shape, (2,) + self.theta.shape)
                self.assertTrue(np.issubdtype(GD_loglik1.dtype, np.float32))
                self.assertTrue(np.issubdtype(GD_theta1.dtype, np.float32))
                
                if method in ["WeightedNewton", "BFGS"]:
                    GD_loglik2, GD_theta2 = pomp_obj.fit(J=3, Jh=3, method=method, itns=1, alpha=0.97, scale=True, ls=True, mode="GD")
                    self.assertEqual(GD_loglik2.shape, (2,))
                    self.assertEqual(GD_theta2.shape, (2,) + self.theta.shape)
                    self.assertTrue(np.issubdtype(GD_loglik2.dtype, np.float32))
                    self.assertTrue(np.issubdtype(GD_theta2.dtype, np.float32))

    def test_fit_GD_invalid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)

        with self.assertRaises(TypeError):
            pomp_obj.fit(mode="SGD")
        with self.assertRaises(TypeError):
            pomp_obj.fit(J=0, mode="GD")
        with self.assertRaises(TypeError):
            pomp_obj.fit(J=-1, mode="GD")
        with self.assertRaises(TypeError):
            pomp_obj.fit(Jh=0, mode="GD")
        with self.assertRaises(TypeError):
            pomp_obj.fit(Jh=-1, mode="GD")

        # useless input
        with self.assertRaises(TypeError):
            pomp_obj.fit(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, J=10, Jh=10, method="BFGS",
                         itns=2, alpha=0.97, scale=True, mode="GD")

    def test_fit_IFAD_valid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)
        IFAD_loglik1, IFAD_theta1 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="SGD", itns=2,
                                                 alpha=0.97, scale=True, mode="IFAD")
        self.assertEqual(IFAD_loglik1.shape, (3,))
        self.assertEqual(IFAD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(np.issubdtype(IFAD_loglik1.dtype, np.float32))
        self.assertTrue(np.issubdtype(IFAD_theta1.dtype, np.float32))

        IFAD_loglik2, IFAD_theta2 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="Newton",
                                                 itns=2, alpha=0.97, scale=True, ls=True, mode="IFAD")
        self.assertEqual(IFAD_loglik2.shape, (3,))
        self.assertEqual(IFAD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(np.issubdtype(IFAD_loglik2.dtype, np.float32))
        self.assertTrue(np.issubdtype(IFAD_theta2.dtype, np.float32))

        IFAD_loglik3, IFAD_theta3 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10,
                                                 method="WeightedNewton", itns=2, alpha=0.97, scale=True, mode="IFAD")
        self.assertEqual(IFAD_loglik3.shape, (3,))
        self.assertEqual(IFAD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(np.issubdtype(IFAD_loglik3.dtype, np.float32))
        self.assertTrue(np.issubdtype(IFAD_theta3.dtype, np.float32))

        IFAD_loglik4, IFAD_theta4 = pomp_obj.fit(sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="BFGS",
                                                 itns=2, alpha=0.97, scale=True, mode="IFAD")
        self.assertEqual(IFAD_loglik4.shape, (3,))
        self.assertEqual(IFAD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(np.issubdtype(IFAD_loglik4.dtype, np.float32))
        self.assertTrue(np.issubdtype(IFAD_theta4.dtype, np.float32))

    def test_fit_IFAD_invalid(self):
        pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)
        # missing
        with self.assertRaises(TypeError):
            pomp_obj.fit()
        with self.assertRaises(TypeError):
            pomp_obj.fit(mode="ADIF")
        with self.assertRaises(TypeError):
            pomp_obj.fit(mode="IFAD")
        with self.assertRaises(TypeError):
            pomp_obj.fit(sigmas=self.sigmas, mode="IFAD")

        # useless input
        with self.assertRaises(TypeError):
            pomp_obj.fit(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses, self.dmeasures,
                         sigmas=0.02, sigmas_init=1e-20, M=2, J=10, Jh=10, method="SGD", itns=2, alpha=0.97, scale=True,
                         mode="IFAD")

    if __name__ == "__main__":
        unittest.main(argv=[''], verbosity=2, exit=False)

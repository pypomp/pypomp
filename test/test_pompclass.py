import jax
import unittest
import jax.numpy as jnp

from pypomp.pomp_class import Pomp
from pypomp.LG import LG



def get_thetas(theta):
    A = theta[0:4].reshape(2, 2)
    C = theta[4:8].reshape(2, 2)
    Q = theta[8:12].reshape(2, 2)
    R = theta[12:16].reshape(2, 2)
    return A, C, Q, R


def transform_thetas(A, C, Q, R):
    return jnp.concatenate([A.flatten(), C.flatten(), Q.flatten(), R.flatten()])


class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        self.LG = LG()
        self.J = 5
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars
        self.sigmas = 0.02
        self.key = jax.random.key(111)

        self.rinit = self.LG.rinit.struct
        self.rproc = self.LG.rproc.struct
        self.dmeas = self.LG.dmeas.struct
        self.rprocess = self.LG.rproc.struct_pf
        self.dmeasure = self.LG.dmeas.struct_pf
        self.rprocesses = self.LG.rproc.struct_per
        self.dmeasures = self.LG.dmeas.struct_per

    def test_basic_initialization(self):
        self.assertEqual(self.LG.covars, self.covars)
        obj_ys = self.LG.ys
        self.assertTrue(jnp.array_equal(obj_ys, self.ys))
        self.assertTrue(jnp.array_equal(self.LG.theta, self.theta))

    def test_invalid_initialization(self):
        # missing parameters
        with self.assertRaises(TypeError):
            Pomp(None, self.rproc, self.dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, None, self.dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, None, self.ys, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            Pomp(self.rinit, self.rproc, self.dmeas, self.theta, self.covars)

    def test_mop_valid(self):
        mop_obj = self.LG.mop(self.J, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj.shape, ())
        self.assertTrue(jnp.isfinite(mop_obj.item()))
        self.assertEqual(mop_obj.dtype, jnp.float32)

        mop_obj_edge = self.LG.mop(1, alpha=0.97, key=self.key)
        self.assertEqual(mop_obj_edge.shape, ())
        self.assertTrue(jnp.isfinite(mop_obj_edge.item()))
        self.assertEqual(mop_obj_edge.dtype, jnp.float32)

    def test_mop_invalid(self):
        # missing values
        with self.assertRaises(TypeError):
            self.LG.mop(alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            self.LG.mop(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            self.LG.mop(0, alpha=0.97, key=self.key)
        with self.assertRaises(TypeError):
            self.LG.mop(-1, alpha=0.97, key=self.key)
        with self.assertRaises(ValueError):
            self.LG.mop(jnp.array([10, 20]), alpha=0.97, key=self.key)

        value = self.LG.mop(self.J, alpha=jnp.inf, key=self.key)
        self.assertEqual(value.dtype, jnp.float32)
        self.assertEqual(value.shape, ())
        self.assertFalse(jnp.isfinite(value.item()))

        # undefined argument
        with self.assertRaises(TypeError):
            self.LG.mop(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            self.LG.mop(
                self.theta,
                self.ys,
                self.J,
                self.rinit,
                self.rprocess,
                self.dmeasure,
                self.covars,
                alpha=0.97,
                key=self.key,
            )

        with self.assertRaises(TypeError):
            self.LG.mop(
                self.J,
                self.rinit,
                self.rprocess,
                self.dmeasure,
                alpha=0.97,
                key=self.key,
            )

    # pfilter
    def test_pfilter_valid(self):
        pfilter_obj = self.LG.pfilter(self.J, thresh=-1, key=self.key)
        self.assertEqual(pfilter_obj.shape, ())
        self.assertTrue(jnp.isfinite(pfilter_obj.item()))
        self.assertEqual(pfilter_obj.dtype, jnp.float32)

        pfilter_obj_edge = self.LG.pfilter(1, thresh=10, key=self.key)
        self.assertEqual(pfilter_obj_edge.shape, ())
        self.assertTrue(jnp.isfinite(pfilter_obj_edge.item()))
        self.assertEqual(pfilter_obj_edge.dtype, jnp.float32)

    def test_pfilter_invalid(self):
        # missing values
        with self.assertRaises(TypeError):
            self.LG.pfilter(thresh=-1, key=self.key)
        with self.assertRaises(TypeError):
            self.LG.pfilter(key=self.key)
        # inappropriate input
        with self.assertRaises(TypeError):
            self.LG.pfilter(0, thresh=100, key=self.key)
        with self.assertRaises(TypeError):
            self.LG.pfilter(-1, thresh=100, key=self.key)
        with self.assertRaises(ValueError):
            self.LG.pfilter(jnp.array([10, 20]), key=self.key)
        # undefined argument
        with self.assertRaises(TypeError):
            self.LG.pfilter(self.J, a=0.97, key=self.key)

        # useless args
        with self.assertRaises(TypeError):
            self.LG.pfilter(
                self.theta,
                self.ys,
                self.J,
                self.rinit,
                self.rprocess,
                self.dmeasure,
                self.covars,
                key=self.key,
            )

        with self.assertRaises(TypeError):
            self.LG.pfilter(
                self.J, self.rinit, self.rprocess, self.dmeasure, key=self.key
            )

    def test_fit_mif_valid(self):
        mif_loglik1, mif_theta1 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=2,
            a=0.9,
            J=self.J,
            mode="IF2",
            key=self.key,
        )
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(
            mif_theta1.shape,
            (
                3,
                self.J,
            )
            + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=2,
            a=0.9,
            J=self.J,
            mode="IF2",
            monitor=False,
            key=self.key,
        )
        self.assertEqual(mif_loglik2.shape, (0,))
        self.assertEqual(
            mif_theta2.shape,
            (
                3,
                self.J,
            )
            + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

        # M = 0
        mif_loglik3, mif_theta3 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=0,
            a=0.9,
            J=self.J,
            mode="IF2",
            key=self.key,
        )
        self.assertEqual(mif_loglik3.shape, (1,))
        self.assertEqual(
            mif_theta3.shape,
            (
                1,
                self.J,
            )
            + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta3.dtype, jnp.float32))

        # M = -1
        mif_loglik4, mif_theta4 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=-1,
            a=0.9,
            J=self.J,
            mode="IF2",
            key=self.key,
        )
        self.assertEqual(mif_loglik4.shape, (1,))
        self.assertEqual(
            mif_theta4.shape,
            (
                1,
                self.J,
            )
            + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta4.dtype, jnp.float32))

        mif_loglik5, mif_theta5 = self.LG.fit(
            sigmas=0.02, sigmas_init=1e-20, mode="IF2", key=self.key
        )
        self.assertEqual(mif_loglik5.shape, (11,))
        self.assertEqual(
            mif_theta5.shape,
            (
                11,
                100,
            )
            + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik5.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta5.dtype, jnp.float32))

    def test_fit_mif_invalid(self):
        # missing args
        with self.assertRaises(TypeError):
            self.LG.fit(mode="IF2", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(mode="IF", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(sigmas=0.02, M=1, mode="IF2", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(sigmas_init=1e-20, a=0.9, mode="IF2", key=self.key)

        # useless input
        with self.assertRaises(TypeError):
            self.LG.fit(
                rinit=self.rinit,
                rprocess=self.rprocess,
                dmeasure=self.dmeasure,
                rprocesses=self.rprocesses,
                dmeasures=self.dmeasures,
                sigmas=0.02,
                sigmas_init=1e-20,
                mode="IF2",
                key=self.key,
            )

    def test_fit_GD_valid(self):
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]

        for method in methods:
            with self.subTest(method=method):
                GD_loglik1, GD_theta1 = self.LG.fit(
                    J=3,
                    Jh=3,
                    method=method,
                    itns=1,
                    alpha=0.97,
                    scale=True,
                    mode="GD",
                    key=self.key,
                )
                self.assertEqual(GD_loglik1.shape, (2,))
                self.assertEqual(GD_theta1.shape, (2,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(GD_loglik1.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(GD_theta1.dtype, jnp.float32))

                if method in ["WeightedNewton", "BFGS"]:
                    GD_loglik2, GD_theta2 = self.LG.fit(
                        J=3,
                        Jh=3,
                        method=method,
                        itns=1,
                        alpha=0.97,
                        scale=True,
                        ls=True,
                        mode="GD",
                        key=self.key,
                    )
                    self.assertEqual(GD_loglik2.shape, (2,))
                    self.assertEqual(GD_theta2.shape, (2,) + self.theta.shape)
                    self.assertTrue(jnp.issubdtype(GD_loglik2.dtype, jnp.float32))
                    self.assertTrue(jnp.issubdtype(GD_theta2.dtype, jnp.float32))

    def test_fit_GD_invalid(self):
        with self.assertRaises(TypeError):
            self.LG.fit(mode="SGD", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(J=0, mode="GD", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(J=-1, mode="GD", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(Jh=0, mode="GD", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(Jh=-1, mode="GD", key=self.key)

        # useless input
        with self.assertRaises(TypeError):
            self.LG.fit(
                self.theta,
                self.ys,
                self.rinit,
                self.rprocess,
                self.dmeasure,
                J=10,
                Jh=10,
                method="BFGS",
                itns=2,
                alpha=0.97,
                scale=True,
                mode="GD",
                key=self.key,
            )

    def test_fit_IFAD_valid(self):
        # pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)
        IFAD_loglik1, IFAD_theta1 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=2,
            J=10,
            Jh=10,
            method="SGD",
            itns=2,
            alpha=0.97,
            scale=True,
            mode="IFAD",
            key=self.key,
        )
        self.assertEqual(IFAD_loglik1.shape, (3,))
        self.assertEqual(IFAD_theta1.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

        IFAD_loglik2, IFAD_theta2 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=2,
            J=10,
            Jh=10,
            method="Newton",
            itns=2,
            alpha=0.97,
            scale=True,
            ls=True,
            mode="IFAD",
            key=self.key,
        )
        self.assertEqual(IFAD_loglik2.shape, (3,))
        self.assertEqual(IFAD_theta2.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta2.dtype, jnp.float32))

        IFAD_loglik3, IFAD_theta3 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=2,
            J=10,
            Jh=10,
            method="WeightedNewton",
            itns=2,
            alpha=0.97,
            scale=True,
            mode="IFAD",
            key=self.key,
        )
        self.assertEqual(IFAD_loglik3.shape, (3,))
        self.assertEqual(IFAD_theta3.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik3.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta3.dtype, jnp.float32))

        IFAD_loglik4, IFAD_theta4 = self.LG.fit(
            sigmas=0.02,
            sigmas_init=1e-20,
            M=2,
            J=10,
            Jh=10,
            method="BFGS",
            itns=2,
            alpha=0.97,
            scale=True,
            mode="IFAD",
            key=self.key,
        )
        self.assertEqual(IFAD_loglik4.shape, (3,))
        self.assertEqual(IFAD_theta4.shape, (3,) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(IFAD_loglik4.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(IFAD_theta4.dtype, jnp.float32))

    def test_fit_IFAD_invalid(self):
        # pomp_obj = Pomp(custom_rinit, custom_rproc, custom_dmeas, self.ys, self.theta)
        # missing
        with self.assertRaises(TypeError):
            self.LG.fit(key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(mode="ADIF", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(mode="IFAD", key=self.key)
        with self.assertRaises(TypeError):
            self.LG.fit(sigmas=self.sigmas, mode="IFAD", key=self.key)

        # useless input
        with self.assertRaises(TypeError):
            self.LG.fit(
                self.theta,
                self.ys,
                self.rinit,
                self.rprocess,
                self.dmeasure,
                self.rprocesses,
                self.dmeasures,
                sigmas=0.02,
                sigmas_init=1e-20,
                M=2,
                J=10,
                Jh=10,
                method="SGD",
                itns=2,
                alpha=0.97,
                scale=True,
                mode="IFAD",
                key=self.key,
            )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

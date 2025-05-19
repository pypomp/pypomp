import jax
import unittest
import jax.numpy as jnp

import pypomp as pp


class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        self.LG = pp.LG()
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
            pp.Pomp(None, self.rproc, self.dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            pp.Pomp(self.rinit, None, self.dmeas, self.ys, self.theta)
        with self.assertRaises(TypeError):
            pp.Pomp(self.rinit, self.rproc, None, self.ys, self.theta)
        with self.assertRaises(TypeError):
            pp.Pomp(self.rinit, self.rproc, self.dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            pp.Pomp(self.rinit, self.rproc, self.dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            pp.Pomp(self.rinit, self.rproc, self.dmeas, None, self.theta)
        with self.assertRaises(TypeError):
            pp.Pomp(self.rinit, self.rproc, self.dmeas, self.ys, None)
        with self.assertRaises(TypeError):
            pp.Pomp(self.rinit, self.rproc, self.dmeas, self.theta, self.covars)

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
        configurations = [
            {"M": 2, "loglik_shape": (3,), "monitor": True, "theta_shape": 3},
            {"M": 2, "loglik_shape": (0,), "monitor": False, "theta_shape": 3},
            {"M": 0, "loglik_shape": (1,), "monitor": True, "theta_shape": 1},
            {"M": -1, "loglik_shape": (1,), "monitor": True, "theta_shape": 1},
            {"M": 10, "loglik_shape": (11,), "monitor": True, "theta_shape": 11},
        ]

        for config in configurations:
            with self.subTest(config=config):
                mif_loglik, mif_theta = self.LG.fit(
                    sigmas=0.02,
                    sigmas_init=1e-20,
                    M=config["M"],
                    a=0.9,
                    J=self.J,
                    mode="IF2",
                    monitor=config["monitor"],
                    key=self.key,
                )
                self.assertEqual(mif_loglik.shape, config["loglik_shape"])
                self.assertEqual(
                    mif_theta.shape,
                    (config["theta_shape"], self.J) + self.theta.shape,
                )
                self.assertTrue(jnp.issubdtype(mif_loglik.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(mif_theta.dtype, jnp.float32))

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
        for method in ["SGD", "Newton", "WeightedNewton", "BFGS"]:
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
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for method in methods:
            IFAD_loglik, IFAD_theta = self.LG.fit(
                sigmas=0.02,
                sigmas_init=1e-20,
                M=2,
                J=10,
                Jh=10,
                method=method,
                itns=2,
                alpha=0.97,
                scale=True,
                mode="IFAD",
                key=self.key,
            )
            self.assertEqual(IFAD_loglik.shape, (3,))
            self.assertEqual(IFAD_theta.shape, (3,) + self.theta.shape)
            self.assertTrue(jnp.issubdtype(IFAD_loglik.dtype, jnp.float32))
            self.assertTrue(jnp.issubdtype(IFAD_theta.dtype, jnp.float32))

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

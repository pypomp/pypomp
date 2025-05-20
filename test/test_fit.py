import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import LG
from pypomp.fit import fit


class TestFit_LG(unittest.TestCase):
    def setUp(self):
        self.LG = LG()
        self.ys = self.LG.ys
        self.covars = None
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.theta = self.LG.theta
        self.sigmas_long = jnp.array([0.02] * (len(self.theta) - 1) + [0])
        self.J = 5
        self.key = jax.random.key(111)

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

    def test_internal_mif_basic(self):
        mif_loglik1, mif_theta1 = fit(
            J=self.J,
            Jh=3,
            theta=self.theta,
            rinit=self.rinit,
            rproc=self.rproc,
            dmeas=self.dmeas,
            ys=self.ys,
            sigmas=0.02,
            sigmas_init=1e-20,
            covars=None,
            M=2,
            a=0.9,
            thresh_mif=-1,
            mode="IF2",
            key=self.key,
        )
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(
            mif_theta1.shape,
            (3, self.J) + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

    def test_class_mif_basic(self):
        mif_loglik1, mif_theta1 = fit(
            self.LG,
            J=self.J,
            Jh=10,
            sigmas=self.sigmas,
            sigmas_init=1e-20,
            M=2,
            a=0.9,
            thresh_mif=-1,
            mode="IF2",
            key=self.key,
        )
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(
            mif_theta1.shape,
            (3, self.J) + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = fit(
            self.LG, sigmas=self.sigmas, sigmas_init=1e-20, mode="IF2", key=self.key
        )
        self.assertEqual(mif_loglik2.shape, (11,))
        self.assertEqual(
            mif_theta2.shape,
            (11, 100) + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

        # check that sigmas isn't modified by mif
        self.assertEqual(self.sigmas, 0.02)

        # check that sigmas array input works
        mif_loglik3, mif_theta3 = fit(
            self.LG,
            sigmas=self.sigmas_long,
            sigmas_init=1e-20,
            mode="IF2",
            key=self.key,
        )
        # check that sigmas isn't modified by mif when passed as an array
        self.assertTrue(
            (self.sigmas_long == jnp.array([0.02] * (len(self.theta) - 1) + [0])).all()
        )
        # check that the last parameter is never perturbed
        self.assertTrue((mif_theta3[:, :, 15] == mif_theta3[0, 0, 15]).all())
        # check that some other parameter is perturbed
        self.assertTrue((mif_theta3[:, 0, 0] != mif_theta3[0, 0, 0]).any())

    def test_invalid_mif_input(self):
        with self.assertRaises(ValueError) as text:
            fit(mode="IF2", key=self.key)
        self.assertEqual(
            str(text.exception), "Invalid Argument Input with Missing Required Argument"
        )

        with self.assertRaises(ValueError) as text:
            fit(
                self.LG,
                J=self.J,
                sigmas=0.02,
                sigmas_init=1e-20,
                M=2,
                a=0.9,
                thresh_mif=-1,
                mode="IF",
                key=self.key,
            )
        self.assertEqual(str(text.exception), "Invalid Mode Input")

        for arg in ["sigmas", "sigmas_init"]:
            with self.subTest(arg=arg):
                with self.assertRaises(ValueError) as text:
                    fit(
                        self.LG,
                        J=self.J,
                        **{arg: getattr(self, arg)},
                        M=2,
                        a=0.9,
                        thresh_mif=-1,
                        mode="IF2",
                        key=self.key,
                    )
                self.assertEqual(
                    str(text.exception),
                    "Invalid Argument Input with Missing sigmas or sigmas_init",
                )

        with self.assertRaises(ValueError) as text:
            fit(
                J=self.J,
                theta=self.theta,
                rinit=self.rinit,
                rproc=self.rproc,
                dmeas=self.dmeas,
                ys=self.ys,
                sigmas=0.02,
                sigmas_init=1e-20,
                covars=None,
                M=2,
                a=0.9,
                thresh_mif=-1,
                mode="IF",
                key=self.key,
            )
        self.assertEqual(str(text.exception), "Invalid Mode Input")

        for arg in ["sigmas", "sigmas_init"]:
            with self.subTest(arg=arg):
                with self.assertRaises(ValueError) as text:
                    fit(
                        J=self.J,
                        theta=self.theta,
                        rinit=self.rinit,
                        rproc=self.rproc,
                        dmeas=self.dmeas,
                        ys=self.ys,
                        **{arg: getattr(self, arg)},
                        covars=None,
                        M=2,
                        a=0.9,
                        thresh_mif=-1,
                        mode="IF2",
                        key=self.key,
                    )
                self.assertEqual(
                    str(text.exception),
                    "Invalid Argument Input with Missing sigmas or sigmas_init",
                )

    def test_internal_GD_basic(self):
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for method in methods:
            with self.subTest(method=method):
                GD_loglik, GD_theta = fit(
                    J=self.J,
                    Jh=3,
                    theta=self.theta,
                    ys=self.ys,
                    rinit=self.rinit,
                    rproc=self.rproc,
                    dmeas=self.dmeas,
                    itns=2,
                    method=method,
                    alpha=0.97,
                    scale=True,
                    mode="GD",
                    key=self.key,
                )
                self.assertEqual(GD_loglik.shape, (3,))
                self.assertEqual(GD_theta.shape, (3,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(GD_loglik.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(GD_theta.dtype, jnp.float32))

    def test_class_GD_basic(self):
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for method in methods:
            with self.subTest(method=method):
                GD_loglik, GD_theta = fit(
                    self.LG,
                    J=self.J,
                    Jh=3,
                    itns=2,
                    method=method,
                    alpha=0.97,
                    scale=True,
                    ls=True,
                    mode="GD",
                    key=self.key,
                )
                self.assertEqual(GD_loglik.shape, (3,))
                self.assertEqual(GD_theta.shape, (3,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(GD_loglik.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(GD_theta.dtype, jnp.float32))

    def test_invalid_GD_input(self):
        with self.assertRaises(ValueError) as text:
            fit(mode="SGD", key=self.key)
        self.assertEqual(
            str(text.exception), "Invalid Argument Input with Missing Required Argument"
        )

        with self.assertRaises(ValueError) as text:
            fit(mode="GD", key=self.key)
        self.assertEqual(
            str(text.exception), "Invalid Argument Input with Missing Required Argument"
        )

        with self.assertRaises(ValueError) as text:
            fit(
                J=self.J,
                Jh=10,
                theta=self.theta,
                rinit=self.rinit,
                rproc=self.rproc,
                dmeas=self.dmeas,
                ys=self.ys,
                itns=2,
                alpha=0.97,
                scale=True,
                mode="SGD",
                key=self.key,
            )
        self.assertEqual(str(text.exception), "Invalid Mode Input")

    def test_internal_IFAD_basic(self):
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for method in methods:
            with self.subTest(method=method):
                IFAD_loglik, IFAD_theta = fit(
                    J=self.J,
                    Jh=3,
                    theta=self.theta,
                    rinit=self.rinit,
                    rproc=self.rproc,
                    dmeas=self.dmeas,
                    ys=self.ys,
                    sigmas=self.sigmas,
                    sigmas_init=1e-20,
                    M=2,
                    a=0.9,
                    method=method,
                    itns=2,
                    ls=True,
                    alpha=0.97,
                    mode="IFAD",
                    key=self.key,
                )
                self.assertEqual(IFAD_loglik.shape, (3,))
                self.assertEqual(IFAD_theta.shape, (3,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(IFAD_loglik.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(IFAD_theta.dtype, jnp.float32))

    def test_class_IFAD_basic(self):
        methods = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for method in methods:
            with self.subTest(method=method):
                IFAD_loglik1, IFAD_theta1 = fit(
                    self.LG,
                    J=self.J,
                    Jh=3,
                    sigmas=self.sigmas,
                    sigmas_init=1e-20,
                    M=2,
                    a=0.9,
                    method=method,
                    itns=1,
                    alpha=0.97,
                    mode="IFAD",
                    key=self.key,
                )
                self.assertEqual(IFAD_loglik1.shape, (2,))
                self.assertEqual(IFAD_theta1.shape, (2,) + self.theta.shape)
                self.assertTrue(jnp.issubdtype(IFAD_loglik1.dtype, jnp.float32))
                self.assertTrue(jnp.issubdtype(IFAD_theta1.dtype, jnp.float32))

    def test_invalid_IFAD_input(self):
        with self.assertRaises(ValueError) as text:
            fit(mode="IFAD", key=self.key)
        self.assertEqual(
            str(text.exception), "Invalid Argument Input with Missing Required Argument"
        )

        with self.assertRaises(ValueError) as text:
            fit(mode="AD", key=self.key)
        self.assertEqual(
            str(text.exception), "Invalid Argument Input with Missing Required Argument"
        )

        with self.assertRaises(ValueError) as text:
            fit(
                self.LG,
                J=self.J,
                sigmas=self.sigmas,
                sigmas_init=1e-20,
                M=2,
                a=0.9,
                method="SGD",
                itns=2,
                alpha=0.97,
                mode="AD",
                key=self.key,
            )
        self.assertEqual(str(text.exception), "Invalid Mode Input")

        for arg in ["sigmas", "sigmas_init"]:
            with self.subTest(arg=arg), self.assertRaises(ValueError) as text:
                fit(
                    self.LG,
                    J=self.J,
                    **{arg: getattr(self, arg)},
                    M=2,
                    a=0.9,
                    method="SGD",
                    itns=2,
                    alpha=0.97,
                    mode="IFAD",
                    key=self.key,
                )
            self.assertEqual(
                str(text.exception),
                "Invalid Argument Input with Missing sigmas or sigmas_init",
            )

        with self.assertRaises(ValueError) as text:
            fit(
                J=self.J,
                Jh=10,
                theta=self.theta,
                rinit=self.rinit,
                rproc=self.rproc,
                dmeas=self.dmeas,
                ys=self.ys,
                sigmas=self.sigmas,
                sigmas_init=1e-20,
                M=2,
                a=0.9,
                method="SGD",
                itns=2,
                ls=True,
                alpha=0.97,
                mode="AD",
                key=self.key,
            )
        self.assertEqual(str(text.exception), "Invalid Mode Input")

        for method in ["SGD", "Newton", "WeightedNewton", "BFGS"]:
            for arg in ["sigmas", "sigmas_init"]:
                with self.subTest(method=method):
                    with self.assertRaises(ValueError) as text:
                        fit(
                            J=self.J,
                            Jh=10,
                            theta=self.theta,
                            rinit=self.rinit,
                            rproc=self.rproc,
                            dmeas=self.dmeas,
                            ys=self.ys,
                            **{arg: getattr(self, arg)},
                            M=2,
                            a=0.9,
                            method=method,
                            itns=2,
                            ls=True,
                            alpha=0.97,
                            mode="IFAD",
                            key=self.key,
                        )
                    self.assertEqual(
                        str(text.exception),
                        "Invalid Argument Input with Missing sigmas or sigmas_init",
                    )

        with self.assertRaises(ValueError) as text:
            fit(
                J=self.J,
                Jh=10,
                theta=self.theta,
                rinit=self.rinit,
                ys=self.ys,
                sigmas=self.sigmas,
                sigmas_init=1e-20,
                M=2,
                a=0.9,
                method="Newton",
                itns=2,
                ls=True,
                alpha=0.97,
                mode="IFAD",
                key=self.key,
            )
        self.assertEqual(
            str(text.exception), "Invalid Argument Input with Missing Required Argument"
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

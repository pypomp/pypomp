import jax
import unittest
import jax.numpy as jnp

import pypomp as pp


class TestFit_LG(unittest.TestCase):
    def setUp(self):
        # Set default values for tests
        self.LG = pp.LG()
        self.ys = self.LG.ys
        self.covars = None
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.theta = self.LG.theta
        self.sigmas_long = jnp.array([0.02] * (len(self.theta) - 1) + [0])
        self.J = 5
        self.key = jax.random.key(111)
        self.a = 0.9
        self.M = 2

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

    def test_internal_mif_basic(self):
        mif_out1 = pp.mif(
            J=self.J,
            theta=self.theta,
            rinit=self.rinit,
            rproc=self.rproc,
            dmeas=self.dmeas,
            ys=self.ys,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas_init,
            covars=None,
            M=self.M,
            a=self.a,
            key=self.key,
            monitor=True,
        )
        self.assertEqual(mif_out1["logLik"].shape, (3,))
        self.assertEqual(
            mif_out1["thetas"].shape,
            (3, self.J) + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_out1["logLik"].dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_out1["thetas"].dtype, jnp.float32))

    def test_class_mif_basic(self):
        for J, M in [(self.J, 2), (100, 10)]:
            mif_out1 = self.LG.mif(
                J=J,
                M=M,
                sigmas=self.sigmas,
                sigmas_init=1e-20,
                a=self.a,
                key=self.key,
                monitor=True,
            )
            self.assertEqual(mif_out1["logLik"].shape, (M + 1,))
            self.assertEqual(mif_out1["thetas"].shape, (M + 1, J) + self.theta.shape)
            self.assertTrue(jnp.issubdtype(mif_out1["logLik"].dtype, jnp.float32))
            self.assertTrue(jnp.issubdtype(mif_out1["thetas"], jnp.float32))

        # check that sigmas isn't modified by mif
        self.assertEqual(self.sigmas, 0.02)

        # check that sigmas array input works
        mif_out2 = self.LG.mif(
            sigmas=self.sigmas_long,
            sigmas_init=1e-20,
            J=self.J,
            M=2,
            a=0.9,
            key=self.key,
        )
        # check that sigmas isn't modified by mif when passed as an array
        self.assertTrue(
            (self.sigmas_long == jnp.array([0.02] * (len(self.theta) - 1) + [0])).all()
        )
        # check that the last parameter is never perturbed
        self.assertTrue(
            (mif_out2["thetas"][:, :, 15] == mif_out2["thetas"][0, 0, 15]).all()
        )
        # check that some other parameter is perturbed
        self.assertTrue(
            (mif_out2["thetas"][:, 0, 0] != mif_out2["thetas"][0, 0, 0]).any()
        )

    def test_invalid_mif_input(self):
        with self.assertRaises(ValueError):
            self.LG.mif(
                sigmas=self.sigmas,
                sigmas_init=self.sigmas_init,
                M=self.M,
                a=self.a,
                J=-1,
                key=self.key,
            )

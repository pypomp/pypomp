import jax
import unittest
import jax.numpy as jnp

import pypomp as pp


class TestFit_LG(unittest.TestCase):
    def setUp(self):
        # Set default values for tests
        self.LG = pp.LG()
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.sigmas_long = jnp.array([0.02] * (len(self.LG.theta[0]) - 1) + [0])
        self.J = 5
        self.key = jax.random.key(111)
        self.a = 0.987
        self.M = 2

    def test_class_mif_basic(self):
        for J, M in [(self.J, 2), (100, 10)]:
            self.LG.mif(
                J=J,
                M=M,
                sigmas=self.sigmas,
                sigmas_init=1e-20,
                a=self.a,
                key=self.key,
            )
            mif_out1 = self.LG.results[-1]
            self.assertEqual(mif_out1["logLiks"][0].shape, (M + 1,))
            self.assertEqual(
                mif_out1["thetas_out"][0].shape, (M + 1, J) + (len(self.LG.theta[0]),)
            )
            self.assertTrue(jnp.issubdtype(mif_out1["logLiks"][0].dtype, jnp.floating))
            self.assertTrue(jnp.issubdtype(mif_out1["thetas_out"][0], jnp.float32))

        # check that sigmas isn't modified by mif
        self.assertEqual(self.sigmas, 0.02)

        # check that sigmas array input works
        self.LG.mif(
            sigmas=self.sigmas_long,
            sigmas_init=1e-20,
            J=self.J,
            M=2,
            a=0.9,
            key=self.key,
        )
        mif_out2 = self.LG.results[-1]
        # check that sigmas isn't modified by mif when passed as an array
        self.assertTrue(
            (
                self.sigmas_long
                == jnp.array([0.02] * (len(self.LG.theta[0]) - 1) + [0])
            ).all()
        )
        # check that the last parameter is never perturbed
        self.assertTrue(
            (
                mif_out2["thetas_out"][0][:, :, 15]
                == mif_out2["thetas_out"][0][0, 0, 15]
            ).all()
        )
        # check that some other parameter is perturbed
        self.assertTrue(
            (
                mif_out2["thetas_out"][0][:, 0, 0] != mif_out2["thetas_out"][0][0, 0, 0]
            ).any()
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

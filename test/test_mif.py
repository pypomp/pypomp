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
        mif_loglik1, mif_theta1 = pp.mif(
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
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(
            mif_theta1.shape,
            (3, self.J) + self.theta.shape,
        )
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

    def test_class_mif_basic(self):
        mif_loglik1, mif_theta1 = self.LG.mif(
            J=self.J,
            sigmas=self.sigmas,
            sigmas_init=1e-20,
            M=2,
            a=self.a,
            key=self.key,
            monitor=True,
        )
        self.assertEqual(mif_loglik1.shape, (3,))
        self.assertEqual(mif_theta1.shape, (3, self.J) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik1.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta1.dtype, jnp.float32))

        mif_loglik2, mif_theta2 = self.LG.mif(
            J=100,
            M=10,
            sigmas=self.sigmas,
            sigmas_init=1e-20,
            a=self.a,
            key=self.key,
            monitor=True,
        )
        self.assertEqual(mif_loglik2.shape, (11,))
        self.assertEqual(mif_theta2.shape, (11, 100) + self.theta.shape)
        self.assertTrue(jnp.issubdtype(mif_loglik2.dtype, jnp.float32))
        self.assertTrue(jnp.issubdtype(mif_theta2.dtype, jnp.float32))

        # check that sigmas isn't modified by mif
        self.assertEqual(self.sigmas, 0.02)

        # check that sigmas array input works
        mif_loglik3, mif_theta3 = self.LG.mif(
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
        self.assertTrue((mif_theta3[:, :, 15] == mif_theta3[0, 0, 15]).all())
        # check that some other parameter is perturbed
        self.assertTrue((mif_theta3[:, 0, 0] != mif_theta3[0, 0, 0]).any())

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

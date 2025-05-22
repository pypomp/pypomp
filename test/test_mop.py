import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import LG
from pypomp.mop import mop


class TestMop_LG(unittest.TestCase):
    def setUp(self):
        self.LG = LG()
        self.J = 5
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars
        self.sigmas = 0.02
        self.key = jax.random.key(111)

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

    def test_internal_basic(self):
        val1 = mop(
            J=self.J,
            rinit=self.rinit,
            rproc=self.rproc,
            dmeas=self.dmeas,
            theta=self.theta,
            ys=self.ys,
            alpha=0.97,
            key=self.key,
        )
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

    def test_class_basic(self):
        val1 = self.LG.mop(J=self.J, alpha=0.97, key=self.key)
        val2 = self.LG.mop(
            J=self.J,
            rinit=self.rinit,
            rproc=self.rproc,
            dmeas=self.dmeas,
            key=self.key,
        )
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)

    def test_invalid_input(self):
        arguments = [
            {"key": self.key},
            {"J": self.J, "key": self.key},
            {"J": self.J, "alpha": 0.97, "key": self.key},
            {"J": self.J, "theta": self.theta, "ys": self.ys, "key": self.key},
            {
                "J": self.J,
                "rinit": self.rinit,
                "rproc": self.rproc,
                "dmeas": self.dmeas,
                "key": self.key,
            },
            {
                "J": self.J,
                "rinit": self.rinit,
                "rproc": self.rproc,
                "ys": self.ys,
                "dmeas": self.dmeas,
                "key": self.key,
            },
            {
                "J": self.J,
                "rinit": self.rinit,
                "rproc": self.rproc,
                "dmeas": self.dmeas,
                "theta": self.theta,
                "key": self.key,
            },
        ]
        for arg in arguments:
            with self.assertRaises(TypeError):
                mop(**arg)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

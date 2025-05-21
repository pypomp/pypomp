import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import LG
from pypomp.pfilter import pfilter


class TestPfilter_LG(unittest.TestCase):
    def setUp(self):
        self.LG = LG()
        self.key = jax.random.key(111)
        self.J = 5
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

    def test_function_basic(self):
        val1 = pfilter(
            J=self.J,
            rinit=self.rinit,
            rproc=self.rproc,
            dmeas=self.dmeas,
            theta=self.theta,
            ys=self.ys,
            covars=self.covars,
            thresh=10,
            key=self.key,
        )
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

    def test_class_basic(self):
        val1 = self.LG.pfilter(J=self.J, key=self.key)
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

    def test_invalid_input(self):
        arguments = [
            {"J": self.J, "key": self.key},
            {"J": self.J, "thresh": -1, "key": self.key},
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
                "dmeas": self.dmeas,
                "ys": self.ys,
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
            with self.assertRaises(ValueError) as text:
                pfilter(**arg)
            self.assertEqual(
                str(text.exception), "Missing rinit, rproc, dmeas, theta, or ys."
            )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

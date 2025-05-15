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
        self.rprocess = self.LG.rprocess
        self.dmeasure = self.LG.dmeasure
        self.rprocesses = self.LG.rprocesses
        self.dmeasures = self.LG.dmeasures

    def test_internal_basic(self):
        val1 = pfilter(
            J=self.J,
            rinit=self.rinit,
            rprocess=self.rprocess,
            dmeasure=self.dmeasure,
            theta=self.theta,
            ys=self.ys,
            covars=self.covars,
            thresh=10,
            key=self.key,
        )
        val2 = pfilter(
            rinit=self.rinit,
            rprocess=self.rprocess,
            dmeasure=self.dmeasure,
            theta=self.theta,
            ys=self.ys,
            key=self.key,
        )
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)

    def test_class_basic(self):
        val1 = pfilter(self.LG, self.J, thresh=10, key=self.key)
        val2 = pfilter(self.LG, key=self.key)
        val3 = pfilter(
            self.LG,
            self.J,
            self.rinit,
            self.rprocess,
            self.dmeasure,
            theta=[],
            ys=[],
            key=self.key,
        )
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)
        self.assertEqual(val2.shape, ())
        self.assertTrue(jnp.isfinite(val2.item()))
        self.assertEqual(val2.dtype, jnp.float32)
        self.assertEqual(val3.shape, ())
        self.assertTrue(jnp.isfinite(val3.item()))
        self.assertEqual(val3.dtype, jnp.float32)

    def test_invalid_input(self):
        arguments = [
            {"key": self.key},
            {"J": self.J, "key": self.key},
            {"J": self.J, "thresh": -1, "key": self.key},
            {"J": self.J, "theta": self.theta, "ys": self.ys, "key": self.key},
            {
                "J": self.J,
                "rinit": self.rinit,
                "rprocess": self.rprocess,
                "dmeasure": self.dmeasure,
                "key": self.key,
            },
            {
                "J": self.J,
                "rinit": self.rinit,
                "rprocess": self.rprocess,
                "dmeasure": self.dmeasure,
                "ys": self.ys,
                "key": self.key,
            },
            {
                "J": self.J,
                "rinit": self.rinit,
                "rprocess": self.rprocess,
                "dmeasure": self.dmeasure,
                "theta": self.theta,
                "key": self.key,
            },
        ]
        for arg in arguments:
            with self.assertRaises(ValueError) as text:
                pfilter(**arg)
            self.assertEqual(str(text.exception), "Invalid Arguments Input")


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

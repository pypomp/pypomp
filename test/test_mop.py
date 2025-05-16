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

        self.rinit = self.LG.rinit.struct
        self.rprocess = self.LG.rproc.struct_pf
        self.dmeasure = self.LG.dmeas.struct_pf
        self.rprocesses = self.LG.rproc.struct_per
        self.dmeasures = self.LG.dmeas.struct_per

    def test_internal_basic(self):
        val1 = mop(
            J=self.J,
            rinit=self.rinit,
            rprocess=self.rprocess,
            dmeasure=self.dmeasure,
            theta=self.theta,
            ys=self.ys,
            alpha=0.97,
            key=self.key,
        )
        val2 = mop(
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
        val1 = mop(self.LG, J=self.J, alpha=0.97, key=self.key)
        val2 = mop(self.LG, key=self.key)
        val3 = mop(
            self.LG,
            J=self.J,
            rinit=self.rinit,
            rprocess=self.rprocess,
            dmeasure=self.dmeasure,
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
            {"J": self.J, "alpha": 0.97, "key": self.key},
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
                "ys": self.ys,
                "dmeasure": self.dmeasure,
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
                mop(**arg)
            self.assertEqual(str(text.exception), "Invalid Arguments Input")


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

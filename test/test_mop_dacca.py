import jax
import unittest
import jax.numpy as jnp

from pypomp.dacca import dacca
from pypomp.mop import mop


class TestMop_Dacca(unittest.TestCase):
    def setUp(self):
        self.dacca = dacca()
        self.J = 3
        self.key = jax.random.key(111)
        self.ys = self.dacca.ys
        self.theta = self.dacca.theta
        self.covars = self.dacca.covars

        self.rinit = self.dacca.rinit.struct
        self.rprocess = self.dacca.rproc.struct_pf
        self.dmeasure = self.dacca.dmeas.struct_pf
        self.rprocesses = self.dacca.rproc.struct_per
        self.dmeasures = self.dacca.dmeas.struct_per

    def test_internal_basic(self):
        val1 = mop(
            J=self.J,
            rinit=self.rinit,
            rprocess=self.rprocess,
            dmeasure=self.dmeasure,
            theta=self.theta,
            ys=self.ys,
            covars=self.covars,
            alpha=0.95,
            key=self.key,
        )
        self.assertEqual(val1.shape, ())
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

    def test_class_basic(self):
        val = mop(self.dacca, self.J, alpha=0.95, key=self.key)
        self.assertEqual(val.shape, ())
        self.assertTrue(jnp.isfinite(val.item()))
        self.assertEqual(val.dtype, jnp.float32)

    def test_invalid_input(self):
        with self.assertRaises(ValueError) as text:
            mop(key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            mop(J=self.J, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        # without inputting 'covars'
        with self.assertRaises(TypeError) as text:
            mop(
                J=self.J,
                rinit=self.rinit,
                rprocess=self.rprocess,
                dmeasure=self.dmeasure,
                theta=self.theta,
                ys=self.ys,
                alpha=0.95,
                key=self.key,
            )

        self.assertEqual(str(text.exception), "'NoneType' object is not subscriptable")


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

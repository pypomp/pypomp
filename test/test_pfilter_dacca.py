import jax
import unittest
import jax.numpy as jnp

from pypomp.dacca import dacca
from pypomp.pfilter import pfilter


class TestPfilterInternal_Dacca(unittest.TestCase):
    def setUp(self):
        self.dacca = dacca()
        self.J = 3
        self.key = jax.random.key(111)
        self.ys = self.dacca.ys
        self.theta = self.dacca.theta
        self.covars = self.dacca.covars

        self.rinit = self.dacca.rinit
        self.rproc = self.dacca.rproc
        self.dmeas = self.dacca.dmeas

    def test_internal_basic(self):
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
        val = pfilter(self.dacca, self.J, thresh=-1, key=self.key)
        self.assertEqual(val.shape, ())
        self.assertTrue(jnp.isfinite(val.item()))
        self.assertEqual(val.dtype, jnp.float32)

    def test_invalid_input(self):
        with self.assertRaises(ValueError) as text:
            pfilter(key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        with self.assertRaises(ValueError) as text:
            pfilter(J=self.J, key=self.key)

        self.assertEqual(str(text.exception), "Invalid Arguments Input")

        # without inputting 'covars'
        with self.assertRaises(TypeError) as text:
            pfilter(
                J=self.J,
                rinit=self.rinit,
                rproc=self.rproc,
                dmeas=self.dmeas,
                theta=self.theta,
                ys=self.ys,
                thresh=-1,
                key=self.key,
            )

        self.assertEqual(str(text.exception), "'NoneType' object is not subscriptable")


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

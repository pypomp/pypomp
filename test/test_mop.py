import jax
import unittest
import jax.numpy as jnp
import pypomp as pp


class TestMop_LG(unittest.TestCase):
    def setUp(self):
        self.LG = pp.LG()
        self.J = 5
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars
        self.sigmas = 0.02
        self.key = jax.random.key(111)

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

    def test_class_basic(self):
        val = self.LG.mop(J=self.J, alpha=0.97, key=self.key)
        self.assertEqual(val[0].shape, ())
        self.assertTrue(jnp.isfinite(val[0].item()))
        self.assertEqual(val[0].dtype, jnp.float32)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

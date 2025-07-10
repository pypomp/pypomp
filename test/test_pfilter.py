import jax
import unittest
import jax.numpy as jnp

from pypomp.LG import LG


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

    def test_class_basic(self):
        self.LG.pfilter(J=self.J, key=self.key)
        val1 = self.LG.results_history[-1]["logLiks"][0]
        self.assertEqual(val1.shape, (1,))
        self.assertTrue(jnp.isfinite(val1.item()))
        self.assertEqual(val1.dtype, jnp.float32)

    def test_reps(self):
        theta_list = [
            self.theta[0],
            {k: v * 2 for k, v in self.theta[0].items()},
        ]
        self.LG.pfilter(J=self.J, key=self.key, reps=2, theta=theta_list)
        val1 = self.LG.results_history[-1]["logLiks"][0]
        self.assertEqual(val1.shape, (2,))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

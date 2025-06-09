import jax
import unittest
import pypomp as pp


class TestSimulate_LG(unittest.TestCase):
    def setUp(self):
        self.LG = pp.LG()
        self.key = jax.random.key(111)
        self.J = 5
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars
        self.nsim = 1

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.rmeas = self.LG.rmeas

    def test_internal_basic(self):
        val = self.LG.simulate(nsim=self.nsim, key=self.key)

        self.assertIsInstance(val, dict)
        self.assertIn("X_sims", val)
        self.assertIn("Y_sims", val)
        self.assertEqual(
            val["X_sims"].shape, (len(self.ys) + 1, self.rmeas.ydim, self.nsim)
        )

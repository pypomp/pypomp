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

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.rmeas = self.LG.rmeas

    def test_internal_basic(self):
        val1 = pp.simulate(
            rinit=self.rinit,
            rproc=self.rproc,
            rmeas=self.rmeas,
            ylen=len(self.ys),
            theta=self.theta,
            covars=self.covars,
            Nsim=1,
            key=self.key,
        )
        self.assertIsInstance(val1, dict)
        self.assertIn("X", val1)
        self.assertIn("Y", val1)

        val2 = self.LG.simulate(Nsim=1, key=self.key)
        self.assertIsInstance(val2, dict)
        self.assertIn("X", val2)
        self.assertIn("Y", val2)

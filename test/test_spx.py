import jax
import unittest
import pypomp as pp


class TestSPX(unittest.TestCase):
    def setUp(self):
        self.spx_model = pp.spx()
        self.J = 3
        self.key = jax.random.key(111)

    def test_spx_pfilter_basic(self):
        self.spx_model.pfilter(J=self.J, key=self.key, reps=1)
        self.assertIsInstance(self.spx_model.results_history, list)
        self.assertGreater(len(self.spx_model.results_history), 0)

    def test_spx_mif_basic(self):
        self.spx_model.mif(
            sigmas=0.02,
            sigmas_init=0.1,
            J=self.J,
            key=self.key,
            M=1,
            a=0.5,
        )
        self.assertIsInstance(self.spx_model.results_history, list)
        self.assertGreater(len(self.spx_model.results_history), 0)

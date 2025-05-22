import jax
import unittest

import pypomp as pp


class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        self.LG1 = pp.LG()
        self.LG2 = pp.LG()
        self.panel = pp.PanelPomp({"LG1": self.LG1, "LG2": self.LG2})
        self.J = 5
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.M = 2
        self.a = 0.9
        self.key = jax.random.key(111)

    def test_basic_initialization(self):
        self.assertIsInstance(self.panel, pp.PanelPomp)
        with self.assertRaises(TypeError):
            pp.PanelPomp(self.LG1)
        with self.assertRaises(TypeError):
            pp.PanelPomp({"LG1": self.LG1, "LG2": 2})

    def test_simulate(self):
        # Test that simulate runs to completion
        sim = self.panel.simulate(key=self.key)
        self.assertIsInstance(sim, dict)

    def test_pfilter(self):
        # Test that pfilter runs to completion
        pfilter_out = self.panel.pfilter(J=self.J, key=self.key)
        self.assertIsInstance(pfilter_out, dict)

    def test_mif(self):
        # Test that mif runs to completion
        mif_out = self.panel.mif(
            J=self.J,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas_init,
            M=self.M,
            a=self.a,
            key=self.key,
        )
        self.assertIsInstance(mif_out, dict)

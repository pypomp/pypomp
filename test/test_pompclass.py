import jax
import unittest

import pypomp as pp


class TestPompClass_LG(unittest.TestCase):
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
        self.rinitalizer = self.LG.rinit.struct_pf
        self.rprocess = self.LG.rproc.struct_pf
        self.dmeasure = self.LG.dmeas.struct_pf
        self.rinitalizers = self.LG.rinit.struct_per
        self.rprocesses = self.LG.rproc.struct_per
        self.dmeasures = self.LG.dmeas.struct_per

    def test_basic_initialization(self):
        self.assertEqual(self.LG.covars, self.covars)

    def test_invalid_initialization(self):
        for arg in ["ys", "theta", "rinit", "rproc", "dmeas"]:
            with self.assertRaises(Exception):
                kwargs = {
                    "ys": self.ys,
                    "theta": self.theta,
                    "rinit": self.rinit,
                    "rproc": self.rproc,
                    "dmeas": self.dmeas,
                }
                kwargs[arg] = None
                pp.Pomp(**kwargs)

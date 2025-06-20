import jax
import unittest
import numpy as np

import pypomp as pp


class TestPompClass_LG(unittest.TestCase):
    def setUp(self):
        self.LG = pp.LG()
        self.J = 3
        self.ys = self.LG.ys
        self.theta = self.LG.theta
        self.covars = self.LG.covars
        self.sigmas = 0.02
        self.a = 0.5
        self.M = 2
        self.key = jax.random.key(111)

        self.rinit = self.LG.rinit
        self.rproc = self.LG.rproc
        self.dmeas = self.LG.dmeas

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

    def test_sample_params(self):
        param_bounds = {
            "R0": (0, 100),
            "sigma": (0, 100),
            "gamma": (0, 100),
        }
        n = 10
        key = jax.random.key(1)
        param_sets = pp.Pomp.sample_params(param_bounds, n, key)
        self.assertEqual(len(param_sets), n)
        for params in param_sets:
            param_names = list(params.keys())
            self.assertEqual(param_names, list(param_bounds.keys()))
            for param_name, value in params.items():
                self.assertIsInstance(value, float)

    def test_theta_carryover(self):
        # Check that theta estimate from mif is correctly carried over to attribute and traces
        theta_order = list(self.LG.theta[0].keys())
        self.LG.mif(
            J=self.J,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas,
            M=self.M,
            a=self.a,
            key=self.key,
        )
        self.assertEqual(theta_order, list(self.LG.theta[0].keys()))
        self.LG.pfilter(J=self.J, reps=2)
        self.assertEqual(list(self.LG.results[-1]["theta"][0].keys()), theta_order)
        self.assertEqual(
            list(self.LG.results[-1]["theta"][0].values()),
            np.mean(self.LG.results[-2]["thetas_out"][0][-1], axis=0).values.tolist(),
        )
        traces = self.LG.traces()
        # Only compare the parameter values
        self.assertEqual(
            traces.iloc[-1, 4:].values.tolist(), traces.iloc[-2, 4:].values.tolist()
        )
        pass

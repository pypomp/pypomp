import jax
import unittest
import jax.numpy as jnp

import pypomp as pp


class TestFit_LG(unittest.TestCase):
    def setUp(self):
        # Set default values for tests
        self.LG = pp.LG()
        self.ys = self.LG.ys
        self.covars = None
        self.theta = self.LG.theta
        self.J = 5
        self.key = jax.random.key(111)
        self.M = 2

    def test_class_GD_basic(self):
        optimizers = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for optimizer in optimizers:
            with self.subTest(optimizer=optimizer):
                self.LG.train(
                    J=self.J,
                    M=self.M,
                    optimizer=optimizer,
                    scale=True,
                    ls=True,
                    n_monitors=1,
                    key=self.key,
                )
                GD_out = self.LG.results_history[-1]
                traces = GD_out["traces"]
                # Check shape for first replicate
                self.assertEqual(
                    traces.sel(replicate=0).shape,
                    (self.M + 1, len(self.LG.theta[0]) + 1),
                )  # +1 for logLik column
                # Check that "logLik" is in variable coordinate
                self.assertIn("logLik", list(traces.coords["variable"].values))
                # Check that all parameter names are in variable coordinate
                for param in self.LG.theta[0].keys():
                    self.assertIn(param, list(traces.coords["variable"].values))

    def test_invalid_GD_input(self):
        with self.assertRaises(ValueError):
            # Check that an error is thrown when J is not positive
            self.LG.train(
                J=0,
                M=self.M,
                scale=True,
                ls=True,
                key=self.key,
            )

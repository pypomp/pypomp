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
        self.Jh = 5
        self.key = jax.random.key(111)
        self.M = 2
        self.itns = 2

    def test_class_GD_basic(self):
        optimizers = ["SGD", "Newton", "WeightedNewton", "BFGS"]
        for optimizer in optimizers:
            with self.subTest(optimizer=optimizer):
                self.LG.train(
                    J=self.J,
                    Jh=self.Jh,
                    itns=self.itns,
                    optimizer=optimizer,
                    scale=True,
                    ls=True,
                    key=self.key,
                )
                GD_out = self.LG.results_history[-1]
                self.assertEqual(GD_out["logLiks"][0].shape, (3,))
                self.assertEqual(
                    GD_out["thetas_out"][0].shape, (3,) + (len(self.theta[0]),)
                )
                self.assertTrue(jnp.issubdtype(GD_out["logLiks"][0], jnp.float32))
                self.assertTrue(jnp.issubdtype(GD_out["thetas_out"][0], jnp.float32))

    def test_invalid_GD_input(self):
        with self.assertRaises(ValueError):
            self.LG.train(
                J=0,
                Jh=self.Jh,
                itns=self.itns,
                scale=True,
                ls=True,
                key=self.key,
            )
        with self.assertRaises(ValueError):
            self.LG.train(
                J=self.J,
                Jh=0,
                itns=self.itns,
                scale=True,
                ls=True,
                key=self.key,
            )

import jax
import unittest
import jax.numpy as jnp

import pypomp as pp


class TestFit_LG(unittest.TestCase):
    def setUp(self):
        # Set default values for tests
        self.LG = pp.LG()
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.sigmas_long = jnp.array([0.02] * (len(self.LG.theta[0]) - 1) + [0])
        self.J = 5
        self.key = jax.random.key(111)
        self.a = 0.987
        self.M = 2

    def test_class_mif_basic(self):
        for J, M in [(self.J, 2), (100, 10)]:
            self.LG.mif(
                J=J,
                M=M,
                sigmas=self.sigmas,
                sigmas_init=1e-20,
                a=self.a,
                key=self.key,
            )
            mif_out1 = self.LG.results_history[-1]
            self.assertEqual(
                mif_out1["traces"][0].shape, (M + 1, len(self.LG.theta[0]) + 1)
            )  # +1 for logLik column
            self.assertTrue("logLik" in mif_out1["traces"][0].columns)
            self.assertTrue(
                all(
                    param in mif_out1["traces"][0].columns
                    for param in self.LG.theta[0].keys()
                )
            )

        # check that sigmas isn't modified by mif
        self.assertEqual(self.sigmas, 0.02)

        # check that sigmas array input works
        self.LG.mif(
            sigmas=self.sigmas_long,
            sigmas_init=1e-20,
            J=self.J,
            M=2,
            a=0.9,
            key=self.key,
        )
        mif_out2 = self.LG.results_history[-1]
        # check that sigmas isn't modified by mif when passed as an array
        self.assertTrue(
            (
                self.sigmas_long
                == jnp.array([0.02] * (len(self.LG.theta[0]) - 1) + [0])
            ).all()
        )
        # check that the last parameter is never perturbed (assuming it's the 16th parameter)
        param_names = list(self.LG.theta[0].keys())
        last_param = param_names[15] if len(param_names) > 15 else param_names[-1]
        self.assertTrue(
            (
                mif_out2["traces"][0][last_param]
                == mif_out2["traces"][0][last_param].iloc[0]
            ).all()
        )
        # check that some other parameter is perturbed
        first_param = param_names[0]
        self.assertTrue(
            (
                mif_out2["traces"][0][first_param]
                != mif_out2["traces"][0][first_param].iloc[0]
            ).any()
        )

    def test_invalid_mif_input(self):
        with self.assertRaises(ValueError):
            self.LG.mif(
                sigmas=self.sigmas,
                sigmas_init=self.sigmas_init,
                M=self.M,
                a=self.a,
                J=-1,
                key=self.key,
            )

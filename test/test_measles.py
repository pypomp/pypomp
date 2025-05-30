import jax.numpy as jnp
import unittest
import pypomp as pp


class Test_Measles(unittest.TestCase):
    def setUp(self):
        self.measles = pp.UKMeasles.Pomp(
            unit=["London"],
            theta={
                "R0": float(jnp.log(56.8)),
                "sigma": float(jnp.log(28.9)),
                "gamma": float(jnp.log(30.4)),
                "iota": float(jnp.log(2.9)),
                "rho": 0.488,
                "sigmaSE": float(jnp.log(0.0878)),
                "psi": float(jnp.log(0.116)),
                "cohort": 0.557,
                "amplitude": 0.554,
                "S_0": 2.97e-02,
                "E_0": 5.17e-05,
                "I_0": 5.14e-05,
                "R_0": 9.70e-01,
            },
        )

    def test_measles_pomp(self):
        x = self.measles
        "breakpoint placeholder"

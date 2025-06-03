import jax.numpy as jnp
import unittest
import pypomp as pp
import jax
import matplotlib.pyplot as plt


class Test_Measles(unittest.TestCase):
    def setUp(self):
        init_params = jnp.array([2.97e-02, 5.17e-05, 5.14e-05, 9.70e-01])
        init_params_T = jnp.log(init_params / jnp.sum(init_params))
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
                "S_0": float(init_params_T[0]),
                "E_0": float(init_params_T[1]),
                "I_0": float(init_params_T[2]),
                "R_0": float(init_params_T[4]),
            },
        )

    def test_measles_pomp(self):
        x = self.measles
        # out1 = x.simulate(key=jax.random.key(1), Nsim=2)
        out2 = x.simulate(
            key=jax.random.key(1),
            Nsim=1,  # , times=self.measles.ys.index[0:10]
        )

        if True:
            fig, axs = plt.subplots(4, 1, sharex=True)
            for i in range(4):
                axs[i].plot(
                    out2["X_sims"].coords["time"],
                    out2["X_sims"].sel(sim=0, element=i),
                )
                axs[i].set_title(f"Element {i}")
            plt.xlabel("time")
            plt.ylabel("Value")
            plt.title("London")
            plt.show()

        "breakpoint placeholder"

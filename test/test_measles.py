import jax.numpy as jnp
import unittest
import pypomp as pp
import jax
# import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)


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
            # dt=7 / 365.25,
        )
        self.J = 3
        self.key = jax.random.key(1)
        self.M = 2
        self.sigmas = 0.02
        self.sigmas_init = 0.1
        self.a = 0.987

    def test_measles_sim(self):
        x = self.measles
        # out1 = x.simulate(key=jax.random.key(1), nsim=2)
        out2 = x.simulate(
            key=jax.random.key(1),
            nsim=1,  # times=self.measles.ys.index[0:1]
        )

        # if False:  # Process and obs plots
        #     fig, axs = plt.subplots(7, 1, sharex=True)
        #     sim_n = 0
        #     for i in range(6):
        #         axs[i].plot(
        #             out2["X_sims"].coords["time"],
        #             out2["X_sims"].sel(sim=sim_n, element=i),
        #         )
        #         axs[i].set_title(f"Element {i}")
        #     axs[6].plot(
        #         out2["Y_sims"].coords["time"], out2["Y_sims"].sel(sim=sim_n, element=0)
        #     )
        #     axs[6].set_title("Observed")
        #     plt.xlabel("time")
        #     plt.ylabel("Value")
        #     plt.title("London")
        #     plt.show()

        # if False:  # Covars plots
        #     fig, axs = plt.subplots(2, 1, sharex=True)
        #     for i, key in enumerate(["pop", "birthrate"]):
        #         axs[i].plot(x.covars.index, x.covars[key], label=key)
        #         axs[i].set_title(key)
        #         axs[i].set_xlabel("Time")
        #         axs[i].set_ylabel(key.capitalize())
        #         axs[i].legend()
        #     plt.tight_layout()
        #     plt.show()

    def test_measles_pfilter(self):
        self.measles.pfilter(J=self.J, key=self.key)

    def test_measles_mif(self):
        self.measles.mif(
            J=self.J,
            key=self.key,
            M=self.M,
            sigmas=self.sigmas,
            sigmas_init=self.sigmas_init,
            a=self.a,
        )

    def test_measles_mop(self):
        self.measles.mop(J=self.J, key=self.key)

    # def test_measles_train(self):
    #     self.measles.train(J=self.J, Jh=self.J, key=self.key)

import jax.numpy as jnp
import pytest
import pypomp as pp
import jax
import numpy as np

# import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="function")
def london():
    init_params = jnp.array([2.97e-02, 5.17e-05, 5.14e-05, 9.70e-01])
    init_params_T = jnp.log(init_params / jnp.sum(init_params))
    measles = pp.UKMeasles.Pomp(
        unit=["London"],
        theta={
            "R0": float(jnp.log(56.8)),
            "sigma": float(jnp.log(28.9)),
            "gamma": float(jnp.log(30.4)),
            "iota": float(jnp.log(2.9)),
            "rho": float(pp.logit(0.488)),
            "sigmaSE": float(jnp.log(0.0878)),
            "psi": float(jnp.log(0.116)),
            "cohort": float(pp.logit(0.557)),
            "amplitude": float(pp.logit(0.554)),
            "S_0": float(init_params_T[0]),
            "E_0": float(init_params_T[1]),
            "I_0": float(init_params_T[2]),
            "R_0": float(init_params_T[4]),
        },
        # dt=7 / 365.25,
    )
    J = 3
    key = jax.random.key(1)
    M = 2
    sigmas = 0.02
    sigmas_init = 0.1
    a = 0.987
    return measles, J, key, M, sigmas, sigmas_init, a


def test_measles_sim(london):
    measles, J, key, M, sigmas, sigmas_init, a = london
    x = measles
    # out1 = x.simulate(key=jax.random.key(1), nsim=2)
    out2 = x.simulate(
        key=jax.random.key(1),
        nsim=1,  # times=self.measles.ys.index[0:1]
    )[0]

    # if True:  # Process and obs plots
    #     import matplotlib.pyplot as plt

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


def test_measles_pfilter(london):
    measles, J, key, M, sigmas, sigmas_init, a = london
    measles.pfilter(J=J, key=key)


def test_measles_pfilter_600(london):
    import time

    measles, J, key, M, sigmas, sigmas_init, a = london

    time_start = time.time()
    measles.pfilter(J=600, key=key)
    measles.results()
    time_taken = time.time() - time_start
    print(f"Time taken: {time_taken} seconds")
    print(measles.results())
    pass


def test_measles_mif(london):
    measles, J, key, M, sigmas, sigmas_init, a = london
    measles.mif(
        J=J,
        key=key,
        M=M,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        a=a,
    )


def test_measles_mop(london):
    measles, J, key, M, sigmas, sigmas_init, a = london
    measles.mop(J=J, key=key)


# Commenting out for now due to errors
# def test_measles_train(self):
#     self.measles.train(M=1, J=self.J, key=self.key)


def test_measles_clean():
    data = pp.UKMeasles.subset(clean=True)
    london_cleaned = np.isnan(
        data["measles"]
        .loc[
            (data["measles"]["unit"] == "London")
            & (data["measles"]["date"] == "1955-08-26"),
            "cases",
        ]
        .values
    )
    assert london_cleaned
    london_cleaned2 = np.isnan(
        data["measles"]
        .loc[
            (data["measles"]["unit"] == "London")
            & (data["measles"]["date"] == "1955-08-19"),
            "cases",
        ]
        .values
    )
    assert not london_cleaned2

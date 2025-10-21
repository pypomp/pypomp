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
            "R_0": float(init_params_T[3]),
        },
        # dt=7 / 365.25,
    )
    rw_sd = pp.RWSigma(
        sigmas={
            "R0": 0.02,
            "sigma": 0.02,
            "gamma": 0.02,
            "iota": 0.02,
            "rho": 0.02,
            "sigmaSE": 0.02,
            "psi": 0.02,
            "cohort": 0.02,
            "amplitude": 0.02,
            "S_0": 0.01,
            "E_0": 0.01,
            "I_0": 0.01,
            "R_0": 0.01,
        },
        init_names=["S_0", "E_0", "I_0", "R_0"],
    )
    J = 3
    key = jax.random.key(1)
    M = 2
    a = 0.5
    return measles, rw_sd, J, key, M, a


def test_measles_sim(london):
    measles, rw_sd, J, key, M, a = london
    measles.simulate(key=key, nsim=1)


def test_measles_pfilter(london):
    measles, rw_sd, J, key, M, a = london
    measles.pfilter(J=J, key=key)


def test_measles_mif(london):
    measles, rw_sd, J, key, M, a = london
    measles.mif(
        J=J,
        key=key,
        M=M,
        rw_sd=rw_sd,
        a=a,
    )


def test_measles_mop(london):
    measles, rw_sd, J, key, M, a = london
    measles.mop(J=J, key=key)


def test_measles_train(london):
    measles, rw_sd, J, key, M, a = london
    measles.train(M=1, J=J, key=key)


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

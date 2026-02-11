import pytest
import pypomp as pp
import jax
import numpy as np

# import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)


BASE_THETA = {
    "R0": 56.8,
    "sigma": 28.9,
    "gamma": 30.4,
    "iota": 2.9,
    "rho": 0.488,
    "sigmaSE": 0.0878,
    "psi": 0.116,
    "cohort": 0.557,
    "amplitude": 0.554,
    "S_0": 2.97e-02,
    "E_0": 5.17e-05,
    "I_0": 5.14e-05,
    "R_0": 9.70e-01,
    "mu": 0.02,
    "alpha": 1.0,
}

DEFAULT_J = 3
DEFAULT_KEY = jax.random.key(1)
DEFAULT_M = 2
DEFAULT_A = 0.5


@pytest.fixture(scope="function")
def london():
    theta = BASE_THETA.copy()
    del theta["mu"]
    del theta["alpha"]
    measles = pp.UKMeasles.Pomp(
        unit=["London"],
        theta=theta,
        clean=True,
        model="001b",
        # dt=7 / 365.25,
    )

    return measles


@pytest.fixture(scope="function")
def default_rw_sd():
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
    return rw_sd


@pytest.fixture(scope="function")
def london_003():
    theta = BASE_THETA.copy()
    measles = pp.UKMeasles.Pomp(
        unit=["London"],
        theta=theta,
        model="003",
    )
    return measles


def test_other_models():
    base = BASE_THETA
    models = {
        "001": base,
        "001c": base,
        "003": base,
        "002": {
            "R0": base["R0"],
            "sigma": base["sigma"],
            "gamma": base["gamma"],
            "iota1": np.log(base["iota"]),
            "iota2": 0.1,
            "rho": base["rho"],
            "sigmaSE": base["sigmaSE"],
            "psi": base["psi"],
            "cohort": base["cohort"],
            "amplitude": base["amplitude"],
            "S_0": base["S_0"],
            "E_0": base["E_0"],
            "I_0": base["I_0"],
            "R_0": base["R_0"],
        },
    }

    key = jax.random.key(0)
    for model, theta in models.items():
        mod_obj = pp.UKMeasles.Pomp(
            unit=["London"],
            theta=theta,
            model=model,
            clean=True,
        )
        mod_obj.simulate(key=key, nsim=1)
        mod_obj.pfilter(J=2, key=key)


def test_measles_sim(london):
    measles = london
    measles.simulate(key=DEFAULT_KEY, nsim=1)


def test_measles_pfilter(london):
    measles = london
    measles.pfilter(J=DEFAULT_J, key=DEFAULT_KEY)

    # Test that double precision works
    jax.config.update("jax_enable_x64", True)
    measles.pfilter(J=DEFAULT_J, key=DEFAULT_KEY)
    jax.config.update("jax_enable_x64", False)


def test_measles_mif(london, default_rw_sd):
    measles = london
    measles.mif(
        J=DEFAULT_J,
        key=DEFAULT_KEY,
        M=DEFAULT_M,
        rw_sd=default_rw_sd,
        a=DEFAULT_A,
    )


def test_measles_mop(london):
    measles = london
    measles.mop(J=DEFAULT_J, key=DEFAULT_KEY)


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


def _test_measles_003_train(london_003):
    measles = london_003
    eta = {param: 0.1 for param in measles.canonical_param_names}
    eta["mu"] = 0.0
    eta["alpha"] = 0.0
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
            "mu": 0.0,
            "alpha": 0.0,
        },
        init_names=["S_0", "E_0", "I_0", "R_0"],
    )
    # with jax.disable_jit():
    # measles.mif(J=DEFAULT_J, key=DEFAULT_KEY, M=DEFAULT_M, rw_sd=rw_sd, a=DEFAULT_A)
    measles.train(J=DEFAULT_J, key=DEFAULT_KEY, M=DEFAULT_M, eta=eta, n_monitors=0)
    df = measles.results().drop(columns=["logLik", "se"])
    assert not df.isnull().to_numpy().any()

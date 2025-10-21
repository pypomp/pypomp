import jax
import pypomp as pp
import pytest


@pytest.fixture(scope="function")
def simple():
    dacca = pp.dacca()
    J = 2
    key = jax.random.key(111)
    ys = dacca.ys
    rw_sd = pp.RWSigma(
        sigmas={
            "gamma": 0.02,
            "m": 0.02,
            "rho": 0.02,
            "epsilon": 0.02,
            "omega": 0.02,
            "c": 0.02,
            "beta_trend": 0.02,
            "sigma": 0.02,
            "tau": 0.02,
            "b1": 0.02,
            "b2": 0.02,
            "b3": 0.02,
            "b4": 0.02,
            "b5": 0.02,
            "b6": 0.02,
            "omega1": 0.02,
            "omega2": 0.02,
            "omega3": 0.02,
            "omega4": 0.02,
            "omega5": 0.02,
            "omega6": 0.02,
        },
        init_names=[],
    )
    return dacca, rw_sd, J, key, ys


def test_dacca_basic(simple):
    # Check whether dacca.mif() finishes running.
    dacca, rw_sd, J, key, ys = simple
    dacca.mif(
        rw_sd=rw_sd,
        J=J,
        thresh=-1,
        key=key,
        M=1,
        a=0.9,
    )


def test_dacca_nstep():
    # Check that dacca.train() runs without error when nstep is specified.
    dacca_nstep = pp.dacca(nstep=10, dt=None)
    dacca_nstep.train(J=2, M=1, key=jax.random.key(111))


def test_dacca_dt():
    # Check that dacca.train() runs without error when dt is specified and nstep
    # happens to be the same for every observation interval.
    dacca_dt = pp.dacca(nstep=None, dt=1 / 240)
    dacca_dt.train(J=2, M=1, key=jax.random.key(111))

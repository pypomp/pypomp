import jax
import pypomp as pp
import pytest


@pytest.fixture(scope="function")
def simple():
    dacca = pp.dacca()
    J = 3
    key = jax.random.key(111)
    ys = dacca.ys
    return dacca, J, key, ys


def test_dacca_basic(simple):
    # Check whether dacca.mif() finishes running.
    dacca, J, key, ys = simple
    dacca.mif(
        sigmas=0.02,
        sigmas_init=0.1,
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

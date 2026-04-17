import jax
import pypomp as pp
import pytest


@pytest.fixture(scope="function")
def simple():
    dacca = pp.models.dacca()
    J = 2
    key = jax.random.key(111)
    ys = dacca.ys
    rw_sd = pp.RWSigma(
        sigmas={
            "gamma": 0.02,
            "m": 0.02,
            "rho": 0.02,
            "epsilon": 0.02,
            "c": 0.02,
            "beta_trend": 0.02,
            "sigma": 0.02,
            "tau": 0.02,
            "alpha": 0.02,
            "delta": 0.02,
            "S_0": 0.02,
            "I_0": 0.02,
            "Y_0": 0.02,
            "R1_0": 0.02,
            "R2_0": 0.02,
            "R3_0": 0.02,
            **{f"bs{i}": 0.02 for i in range(1, 7)},
            **{f"omegas{i}": 0.02 for i in range(1, 7)},
        },
        init_names=[],
    )
    return dacca, rw_sd, J, key, ys


def test_dacca_pfilter(simple):
    dacca, rw_sd, J, key, ys = simple
    dacca.pfilter(J=1000, reps=6, key=key)
    logLik = pp.maths.logmeanexp(dacca.results_history[-1].logLiks)
    # Threshold used to be 2.0, but the logLik increased somewhat after updating the
    # model to round the starting state values, like the R pomp model. Kind of surprised
    # that made a meaningful difference.
    # -3748.6 comes from what I understand the R pomp MLE to be.
    assert abs(logLik - -3748.6) < 2.5


def test_dacca_basic(simple):
    # Check whether dacca.mif() runs without error.
    dacca, rw_sd, J, key, ys = simple
    dacca.mif(rw_sd=rw_sd, J=J, key=key, M=1, a=0.5)


def test_dacca_nstep():
    # Check that dacca.train() runs without error when nstep is specified.
    dacca_nstep = pp.models.dacca(nstep=10, dt=None)
    eta = {param: 0.2 for param in dacca_nstep.canonical_param_names}
    dacca_nstep.train(J=2, M=1, eta=eta, key=jax.random.key(111))


def test_dacca_dt():
    # Check that dacca.train() runs without error when dt is specified and nstep
    # happens to be the same for every observation interval.
    dacca_dt = pp.models.dacca(nstep=None, dt=1 / 240)
    eta = {param: 0.2 for param in dacca_dt.canonical_param_names}
    dacca_dt.train(J=2, M=1, eta=eta, key=jax.random.key(111))

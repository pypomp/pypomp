"""Shared fixtures for test_pomp_abc.py and test_pomp_pmcmc.py.

Each expensive POMP model is built *once* per module (scope="module") and
reset to a clean mutable state before every test (scope="function"), following
the same two-level pattern used in test_pomp_mif.py / test_pomp_pfilter.py.
"""

from copy import deepcopy

import jax
import pandas as pd
import pytest

import pypomp as pp


# ---------------------------------------------------------------------------
# SIR model (used by most ABC and PMCMC tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sir_module():
    """Build a SIR Pomp once per module."""
    model = pp.models.sir(seed=42)
    theta = model.theta
    return model, theta


@pytest.fixture(scope="function")
def sir(sir_module):
    """Return a per-test SIR with cleared results_history and reset theta."""
    model_orig, theta = sir_module
    model = deepcopy(model_orig)
    model.results_history.clear()
    model.theta = theta
    return model


# ---------------------------------------------------------------------------
# Static-normal POMP (used by a handful of PMCMC quantitative tests)
# ---------------------------------------------------------------------------


def _build_static_normal_pomp():
    """One-observation POMP with a deterministic normal likelihood in theta['mu']."""

    def rinit(theta_, key, covars, t0):
        return {"X": 0.0}

    def rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"]}

    def dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(
            Y_["Y"], loc=theta_["mu"], scale=theta_["sigma"]
        )

    def rmeas(X_, theta_, key, covars, t):
        return {"Y": theta_["mu"] + theta_["sigma"] * jax.random.normal(key, ())}

    return pp.Pomp(
        ys=pd.DataFrame({"Y": [1.0]}, index=[1.0]),
        theta=pp.PompParameters({"mu": 0.0, "sigma": 0.1}),
        statenames=["X"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        nstep=1,
    )


@pytest.fixture(scope="module")
def static_normal_pomp_module():
    """Build the static-normal POMP once per module."""
    model = _build_static_normal_pomp()
    theta = model.theta
    return model, theta


@pytest.fixture(scope="function")
def static_normal_pomp(static_normal_pomp_module):
    """Return a per-test static-normal POMP with reset mutable state."""
    model_orig, theta = static_normal_pomp_module
    model = deepcopy(model_orig)
    model.results_history.clear()
    model.theta = theta
    return model


# ---------------------------------------------------------------------------
# Deterministic-measurement POMP (ABC distance verification)
# ---------------------------------------------------------------------------


def _build_deterministic_measurement_pomp():
    """One-observation POMP with exact ABC distance in theta['mu']."""

    def rinit(theta_, key, covars, t0):
        return {"X": 0.0}

    def rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"]}

    def rmeas(X_, theta_, key, covars, t):
        return {"Y": theta_["mu"]}

    return pp.Pomp(
        ys=pd.DataFrame({"Y": [1.0]}, index=[1.0]),
        theta=pp.PompParameters({"mu": 0.0}),
        statenames=["X"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        rmeas=rmeas,
        nstep=1,
    )


@pytest.fixture(scope="module")
def deterministic_meas_pomp_module():
    """Build the deterministic-measurement POMP once per module."""
    model = _build_deterministic_measurement_pomp()
    theta = model.theta
    return model, theta


@pytest.fixture(scope="function")
def deterministic_meas_pomp(deterministic_meas_pomp_module):
    """Return a per-test deterministic-measurement POMP with reset mutable state."""
    model_orig, theta = deterministic_meas_pomp_module
    model = deepcopy(model_orig)
    model.results_history.clear()
    model.theta = theta
    return model

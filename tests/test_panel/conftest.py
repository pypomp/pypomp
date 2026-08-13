from copy import deepcopy

import jax
import pandas as pd
import pytest

import pypomp as pp
from tests.helpers.models import lg_panel, sir_panel
from tests.helpers.params import measles_rw_sd as measles_rw_sigma
from tests.helpers.params import uniform_rw_sd


@pytest.fixture(scope="module")
def measles_panel_setup_pomps_module():
    AK_mles = pp.models.UKMeasles.AK_mles()
    london_theta = {str(k): float(v) for k, v in AK_mles["London"].items()}
    hastings_theta = {str(k): float(v) for k, v in AK_mles["Hastings"].items()}
    london = pp.models.UKMeasles.Pomp(
        unit="London",
        theta=pp.PompParameters(london_theta),
    )
    hastings = pp.models.UKMeasles.Pomp(
        unit="Hastings",
        theta=pp.PompParameters(hastings_theta),
    )
    return london, hastings, AK_mles


@pytest.fixture(scope="module")
def measles_rw_sd():
    return measles_rw_sigma()


@pytest.fixture(scope="module")
def measles_panel_setup_some_shared_module(
    measles_panel_setup_pomps_module, measles_rw_sd
):
    london, hastings, AK_mles = measles_panel_setup_pomps_module
    unit_specific = AK_mles[["London", "Hastings"]].drop(labels=["gamma", "cohort"])
    shared = (
        AK_mles[["London", "Hastings"]]
        .loc[["gamma", "cohort"], :]
        .mean(axis=1)
        .to_frame(name="shared")
    )
    assert isinstance(shared, pd.DataFrame)
    theta = (
        pp.PanelParameters(theta=[{"shared": shared, "unit_specific": unit_specific}])
        * 2
    )
    panel = pp.PanelPomp(
        Pomp_dict={"London": london, "Hastings": hastings},
        theta=theta,
    )
    assert panel.theta is not None
    key = jax.random.key(0)
    fresh_key = panel.fresh_key
    return panel, measles_rw_sd, theta, key, fresh_key


@pytest.fixture(scope="function")
def measles_panel_setup_some_shared(measles_panel_setup_some_shared_module):
    panel_orig, rw_sd, theta, key, fresh_key = measles_panel_setup_some_shared_module
    panel = deepcopy(panel_orig)
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    panel.fresh_key = fresh_key
    return panel, rw_sd, key


def _lg_panel_setup(sharing, n_reps):
    panel = lg_panel(sharing=sharing, n_reps=n_reps)
    return panel, uniform_rw_sd(panel), panel.theta, jax.random.key(0), panel.fresh_key


@pytest.fixture(scope="module")
def lg_panel_setup_some_shared_module():
    return _lg_panel_setup("some", n_reps=2)


@pytest.fixture(scope="function")
def lg_panel_setup_some_shared(lg_panel_setup_some_shared_module):
    panel_orig, rw_sd, theta, key, fresh_key = lg_panel_setup_some_shared_module
    panel = deepcopy(panel_orig)
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    panel.fresh_key = fresh_key
    return panel, rw_sd, key


@pytest.fixture(scope="module")
def lg_panel_setup_specific_only_module():
    return _lg_panel_setup("none", n_reps=2)


@pytest.fixture(scope="function")
def lg_panel_setup_specific_only(lg_panel_setup_specific_only_module):
    panel_orig, rw_sd, theta, key, fresh_key = lg_panel_setup_specific_only_module
    panel = deepcopy(panel_orig)
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    panel.fresh_key = fresh_key
    return panel, rw_sd, key


@pytest.fixture(scope="module")
def lg_panel_setup_shared_only_module():
    return _lg_panel_setup("all", n_reps=1)


@pytest.fixture(scope="function")
def lg_panel_setup_shared_only(lg_panel_setup_shared_only_module):
    panel_orig, rw_sd, theta, key, fresh_key = lg_panel_setup_shared_only_module
    panel = deepcopy(panel_orig)
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    panel.fresh_key = fresh_key
    return panel, rw_sd, key


@pytest.fixture(scope="module")
def lg_panel_mp_module(lg_panel_setup_some_shared_module):
    panel_orig, rw_sd, theta, key, fresh_key = lg_panel_setup_some_shared_module
    panel = deepcopy(panel_orig)
    J = 2
    M = 2
    a = 0.5
    panel.mif(J=J, rw_sd=rw_sd.geometric_cooling(a=a), M=M, key=key)
    panel.pfilter(J=J)
    results_history = deepcopy(panel.results_history)
    fresh_key = panel.fresh_key
    return (
        panel_orig,
        rw_sd,
        key,
        J,
        M,
        a,
        theta,
        fresh_key,
        results_history,
    )


@pytest.fixture(scope="function")
def lg_panel_mp(lg_panel_mp_module):
    (
        panel_orig,
        rw_sd,
        key,
        J,
        M,
        a,
        theta,
        fresh_key,
        results_history,
    ) = lg_panel_mp_module
    panel = deepcopy(panel_orig)
    panel.results_history = deepcopy(results_history)
    panel.theta = deepcopy(theta)
    panel.fresh_key = fresh_key
    return panel, rw_sd, key, J, M, a


# ---------------------------------------------------------------------------
# SIR panel fixtures for test_panel_dpop_train.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sir_panel_dpop_module():
    """Build the all-unit-specific SIR panel once per module."""
    panel = sir_panel(sharing="none")
    return panel, panel.theta


@pytest.fixture(scope="function")
def sir_panel_dpop(sir_panel_dpop_module):
    """Per-test SIR panel with cleared results_history and reset theta."""
    panel_orig, theta = sir_panel_dpop_module
    panel = deepcopy(panel_orig)
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    return panel


@pytest.fixture(scope="module")
def sir_panel_with_shared_dpop_module():
    """Build the shared-params SIR panel once per module."""
    panel = sir_panel(sharing="some")
    return panel, panel.theta


@pytest.fixture(scope="function")
def sir_panel_with_shared_dpop(sir_panel_with_shared_dpop_module):
    """Per-test shared-params SIR panel with reset mutable state."""
    panel_orig, theta = sir_panel_with_shared_dpop_module
    panel = deepcopy(panel_orig)
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    return panel

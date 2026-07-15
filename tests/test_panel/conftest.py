import pandas as pd
import jax
import pypomp as pp
import pytest
from copy import deepcopy


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
    return pp.RWSigma(
        sigmas={
            "gamma": 0.02,
            "cohort": 0.02,
            "amplitude": 0.02,
            "sigmaSE": 0.02,
            "psi": 0.02,
            "iota": 0.02,
            "rho": 0.02,
            "R0": 0.02,
            "sigma": 0.02,
            "S_0": 0.01,
            "E_0": 0.01,
            "I_0": 0.01,
            "R_0": 0.01,
        },
        init_names=["S_0", "E_0", "I_0", "R_0"],
    )


@pytest.fixture(scope="module")
def measles_panel_setup_specific_only_module(
    measles_panel_setup_pomps_module, measles_rw_sd
):
    london, hastings, AK_mles = measles_panel_setup_pomps_module
    unit_specific = AK_mles[["London", "Hastings"]]
    assert isinstance(unit_specific, pd.DataFrame)
    theta = (
        pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}]) * 2
    )
    panel = pp.PanelPomp(
        Pomp_dict={"London": london, "Hastings": hastings},
        theta=theta,
    )
    key = jax.random.key(0)
    assert panel.theta is not None
    fresh_key = panel.fresh_key
    return panel, measles_rw_sd, theta, key, fresh_key


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
def measles_panel_setup_specific_only(measles_panel_setup_specific_only_module):
    panel, rw_sd, theta, key, fresh_key = measles_panel_setup_specific_only_module
    panel.results_history.clear()
    panel.theta = theta
    panel.fresh_key = fresh_key
    return panel, rw_sd, key


@pytest.fixture(scope="function")
def measles_panel_setup_some_shared(measles_panel_setup_some_shared_module):
    panel, rw_sd, theta, key, fresh_key = measles_panel_setup_some_shared_module
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    panel.fresh_key = fresh_key
    return panel, rw_sd, key


@pytest.fixture(scope="module")
def measles_panel_mp_module(measles_panel_setup_specific_only_module):
    panel, rw_sd, theta, key, fresh_key = measles_panel_setup_specific_only_module
    J = 2
    M = 2
    a = 0.5
    panel.mif(J=J, rw_sd=rw_sd.geometric_cooling(a=a), M=M, key=key)
    panel.pfilter(J=J)
    results_history = panel.results_history
    fresh_key = panel.fresh_key
    return (
        panel,
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
def measles_panel_mp(measles_panel_mp_module):
    (
        panel,
        rw_sd,
        key,
        J,
        M,
        a,
        theta,
        fresh_key,
        results_history,
    ) = measles_panel_mp_module
    panel.results_history = results_history
    panel.theta = deepcopy(theta)
    panel.fresh_key = fresh_key
    return panel, rw_sd, key, J, M, a


@pytest.fixture(scope="module")
def lg_panel_setup_some_shared_module():
    lg1 = pp.models.LG()
    lg2 = pp.models.LG()
    # Create PanelParameters with some shared and some unit-specific
    shared_names = ["A11", "C11"]
    unit_specific_names = [
        n for n in lg1.canonical_param_names if n not in shared_names
    ]

    # Simple averaging for shared
    p1, p2 = lg1.theta[0], lg2.theta[0]
    shared_df = pd.DataFrame(
        {"shared": [(p1[n] + p2[n]) / 2 for n in shared_names]},
        index=pd.Index(shared_names),
    )

    unit_specific_df = pd.DataFrame(
        {
            "unit1": [p1[n] for n in unit_specific_names],
            "unit2": [p2[n] for n in unit_specific_names],
        },
        index=pd.Index(unit_specific_names),
    )

    theta = (
        pp.PanelParameters(
            theta=[{"shared": shared_df, "unit_specific": unit_specific_df}]
        )
        * 2
    )
    panel = pp.PanelPomp(
        Pomp_dict={"unit1": lg1, "unit2": lg2},
        theta=theta,
    )
    key = jax.random.key(0)
    fresh_key = panel.fresh_key

    # Create simple rw_sd for LG
    rw_sd = pp.RWSigma(
        sigmas={n: 0.02 for n in lg1.canonical_param_names}, init_names=[]
    )

    return panel, rw_sd, theta, key, fresh_key


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
    lg1 = pp.models.LG()
    lg2 = pp.models.LG()
    # Create PanelParameters with only unit-specific
    p1, p2 = lg1.theta[0], lg2.theta[0]
    unit_specific_df = pd.DataFrame(
        {
            "unit1": [p1[n] for n in lg1.canonical_param_names],
            "unit2": [p2[n] for n in lg2.canonical_param_names],
        },
        index=pd.Index(lg1.canonical_param_names),
    )

    theta = (
        pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific_df}])
        * 2
    )
    panel = pp.PanelPomp(
        Pomp_dict={"unit1": lg1, "unit2": lg2},
        theta=theta,
    )
    key = jax.random.key(0)
    fresh_key = panel.fresh_key

    # Create simple rw_sd for LG
    rw_sd = pp.RWSigma(
        sigmas={n: 0.02 for n in lg1.canonical_param_names}, init_names=[]
    )

    return panel, rw_sd, theta, key, fresh_key


@pytest.fixture(scope="function")
def lg_panel_setup_specific_only(lg_panel_setup_specific_only_module):
    panel_orig, rw_sd, theta, key, fresh_key = lg_panel_setup_specific_only_module
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


import numpy as np  # noqa: E402 (module-level import at end of conftest is fine)


def _build_sir_panel_dpop():
    """Build a 2-unit all-unit-specific SIR panel for dpop_train tests."""
    test_times = np.arange(1 / 52, 5 / 52, 1 / 52)
    sir1 = pp.models.sir(seed=100, times=test_times)
    sir2 = pp.models.sir(seed=200, times=test_times)
    param_names = sir1.canonical_param_names
    theta1 = sir1.theta[0]
    theta2 = sir2.theta[0]
    unit_specific = pd.DataFrame(
        {
            "unit1": [theta1[p] for p in param_names],
            "unit2": [theta2[p] for p in param_names],
        },
        index=pd.Index(param_names),
    )
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])
    panel = pp.PanelPomp(Pomp_dict={"unit1": sir1, "unit2": sir2}, theta=theta)
    return panel, theta


def _build_sir_panel_with_shared_dpop():
    """Build a 2-unit SIR panel with gamma and mu shared for dpop_train tests."""
    test_times = np.arange(1 / 52, 5 / 52, 1 / 52)
    sir1 = pp.models.sir(seed=100, times=test_times)
    sir2 = pp.models.sir(seed=200, times=test_times)
    param_names = sir1.canonical_param_names
    theta1 = sir1.theta[0]
    theta2 = sir2.theta[0]

    shared_names = ["gamma", "mu"]
    unit_names_param = [p for p in param_names if p not in shared_names]

    shared = pd.DataFrame(
        {"shared": [(theta1[p] + theta2[p]) / 2 for p in shared_names]},
        index=pd.Index(shared_names),
    )
    unit_specific = pd.DataFrame(
        {
            "unit1": [theta1[p] for p in unit_names_param],
            "unit2": [theta2[p] for p in unit_names_param],
        },
        index=pd.Index(unit_names_param),
    )
    theta = pp.PanelParameters(
        theta=[{"shared": shared, "unit_specific": unit_specific}]
    )
    panel = pp.PanelPomp(Pomp_dict={"unit1": sir1, "unit2": sir2}, theta=theta)
    return panel, theta


@pytest.fixture(scope="module")
def sir_panel_dpop_module():
    """Build the all-unit-specific SIR panel once per module."""
    return _build_sir_panel_dpop()


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
    return _build_sir_panel_with_shared_dpop()


@pytest.fixture(scope="function")
def sir_panel_with_shared_dpop(sir_panel_with_shared_dpop_module):
    """Per-test shared-params SIR panel with reset mutable state."""
    panel_orig, theta = sir_panel_with_shared_dpop_module
    panel = deepcopy(panel_orig)
    panel.results_history.clear()
    panel.theta = deepcopy(theta)
    return panel

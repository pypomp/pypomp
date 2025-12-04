import pandas as pd
import jax
import pypomp as pp
import pytest
from copy import deepcopy


@pytest.fixture(scope="module")
def measles_panel_setup_pomps_module():
    AK_mles = pp.UKMeasles.AK_mles()
    london_theta = AK_mles["London"].to_dict()
    hastings_theta = AK_mles["Hastings"].to_dict()
    london = pp.UKMeasles.Pomp(
        unit=["London"],
        theta=london_theta,
    )
    hastings = pp.UKMeasles.Pomp(
        unit=["Hastings"],
        theta=hastings_theta,
    )
    return london, hastings, AK_mles


@pytest.fixture(scope="module")
def measles_panel_setup_specific_only_module(measles_panel_setup_pomps_module):
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
    rw_sd = pp.RWSigma(
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
    return panel, rw_sd, theta, key, fresh_key


@pytest.fixture(scope="module")
def measles_panel_setup_some_shared_module(measles_panel_setup_pomps_module):
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
    rw_sd = pp.RWSigma(
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
    return panel, rw_sd, theta, key, fresh_key


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
    rw_sd = pp.RWSigma(
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
    panel.mif(J=J, rw_sd=rw_sd, M=M, a=a, key=key)
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

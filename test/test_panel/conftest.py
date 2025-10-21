import pandas as pd
import jax
import pypomp as pp
import pytest


@pytest.fixture(scope="module")
def measles_panel_setup_module():
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
    unit_specific = AK_mles[["London", "Hastings"]]
    assert isinstance(unit_specific, pd.DataFrame)
    panel = pp.PanelPomp(
        Pomp_dict={"London": london, "Hastings": hastings},
        shared=None,
        unit_specific=unit_specific,
    )
    key = jax.random.key(0)
    shared = panel.shared
    unit_specific = panel.unit_specific
    fresh_key = panel.fresh_key
    return panel, key, shared, unit_specific, fresh_key


@pytest.fixture(scope="function")
def measles_panel_setup(measles_panel_setup_module):
    panel, key, shared, unit_specific, fresh_key = measles_panel_setup_module
    panel.results_history.clear()
    panel.shared = shared
    panel.unit_specific = unit_specific
    panel.fresh_key = fresh_key
    return panel, key


@pytest.fixture(scope="function")
def measles_panel_setup2(measles_panel_setup):
    panel, key = measles_panel_setup
    panel.unit_specific = panel.unit_specific * 2
    return panel, key


@pytest.fixture(scope="module")
def measles_panel_mp_module(measles_panel_setup_module):
    panel, key, shared, unit_specific, fresh_key = measles_panel_setup_module
    J = 2
    M = 2
    a = 0.5
    sigmas = 0.02
    sigmas_init = 0.1
    panel.mif(J=J, key=key, sigmas=sigmas, sigmas_init=sigmas_init, M=M, a=a)
    panel.pfilter(J=J)
    results_history = panel.results_history
    fresh_key = panel.fresh_key
    return (
        panel,
        key,
        J,
        M,
        a,
        sigmas,
        sigmas_init,
        shared,
        unit_specific,
        fresh_key,
        results_history,
    )


@pytest.fixture(scope="function")
def measles_panel_mp(measles_panel_mp_module):
    (
        panel,
        key,
        J,
        M,
        a,
        sigmas,
        sigmas_init,
        shared,
        unit_specific,
        fresh_key,
        results_history,
    ) = measles_panel_mp_module
    panel.results_history = results_history
    panel.shared = shared
    panel.unit_specific = unit_specific
    panel.fresh_key = fresh_key
    return panel, key, J, M, a, sigmas, sigmas_init

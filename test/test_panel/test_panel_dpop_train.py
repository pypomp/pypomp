from copy import deepcopy
import jax
import jax.numpy as jnp
import pytest
import pypomp as pp
from pypomp.results import PanelPompDpopTrainResult


def _get_sir_panel():
    """Build a panel of 2 SIR units for testing panel DPOP train."""
    sir1 = pp.sir(seed=100)
    sir2 = pp.sir(seed=200)

    # All parameters are unit-specific (no shared)
    import pandas as pd

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

    theta = pp.PanelParameters(
        theta=[{"shared": None, "unit_specific": unit_specific}]
    )

    panel = pp.PanelPomp(
        Pomp_dict={"unit1": sir1, "unit2": sir2},
        theta=theta,
    )
    return panel


def _get_sir_panel_with_shared():
    """Build a panel of 2 SIR units with shared and unit-specific params."""
    sir1 = pp.sir(seed=100)
    sir2 = pp.sir(seed=200)

    import pandas as pd

    param_names = sir1.canonical_param_names
    theta1 = sir1.theta[0]
    theta2 = sir2.theta[0]

    # Make gamma and mu shared; the rest unit-specific
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

    panel = pp.PanelPomp(
        Pomp_dict={"unit1": sir1, "unit2": sir2},
        theta=theta,
    )
    return panel


@pytest.mark.parametrize("chunk_size", [1, 2], ids=["chunk1", "chunk2"])
def test_panel_dpop_train_adam(chunk_size):
    panel = _get_sir_panel()
    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=chunk_size,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    assert res.method == "dpop_train"
    assert res.shared_traces.shape[0] == 1  # n_reps
    assert res.shared_traces.shape[1] == M + 1
    assert res.unit_traces.shape[0] == 1  # n_reps
    assert res.unit_traces.shape[1] == M + 1
    assert res.unit_traces.shape[3] == len(panel.get_unit_names())  # U
    assert res.process_weight_state == "logw"
    assert res.decay == 0.0


def test_panel_dpop_train_sgd():
    panel = _get_sir_panel()
    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer="SGD",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(42),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    assert res.optimizer == "SGD"


def test_panel_dpop_train_with_decay():
    panel = _get_sir_panel()
    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer="Adam",
        alpha=0.8,
        decay=0.1,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    assert res.decay == 0.1


def test_panel_dpop_train_with_shared():
    panel = _get_sir_panel_with_shared()
    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    # Should have shared params in the shared traces
    shared_vars = list(res.shared_traces.coords["variable"].values)
    assert "logLik" in shared_vars
    assert "gamma" in shared_vars
    assert "mu" in shared_vars


def test_panel_dpop_train_to_dataframe():
    panel = _get_sir_panel_with_shared()
    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    df = res.to_dataframe()
    assert "shared logLik" in df.columns
    assert "unit logLik" in df.columns

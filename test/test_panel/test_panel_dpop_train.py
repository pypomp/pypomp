from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
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


def test_panel_dpop_train_multi_replicate():
    """Multiple replicates should produce independent traces."""
    panel = _get_sir_panel()
    import pandas as pd

    param_names = panel.canonical_param_names
    unit_names = panel.get_unit_names()

    # Build 2-replicate theta
    base_theta = panel.theta.theta[0]
    theta = pp.PanelParameters(theta=[deepcopy(base_theta), deepcopy(base_theta)])

    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=theta,
        chunk_size=1,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    assert res.shared_traces.shape[0] == 2  # 2 replicates
    assert res.unit_traces.shape[0] == 2


def test_panel_dpop_train_reproducibility():
    """Same key and same initial state should produce identical results."""
    J, M = 2, 2
    kwargs = dict(
        J=J,
        M=M,
        eta=0.01,
        chunk_size=1,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(99),
    )

    panel1 = _get_sir_panel()
    panel1.dpop_train(theta=deepcopy(panel1.theta), **kwargs)
    res1 = panel1.results_history[-1]

    panel2 = _get_sir_panel()
    panel2.dpop_train(theta=deepcopy(panel2.theta), **kwargs)
    res2 = panel2.results_history[-1]

    np.testing.assert_array_equal(
        np.array(res1.unit_traces), np.array(res2.unit_traces)
    )


def test_panel_dpop_train_params_change():
    """Parameters should actually change after training iterations."""
    panel = _get_sir_panel()
    J, M = 2, 5
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
    unit_vars = [v for v in res.unit_traces.coords["variable"].values if v != "unitLogLik"]
    initial = res.unit_traces.sel(replicate=0, iteration=0, variable=unit_vars).values
    final = res.unit_traces.sel(replicate=0, iteration=M, variable=unit_vars).values
    assert not np.allclose(initial, final), "Parameters should change after training"


def test_panel_dpop_train_invalid_J():
    panel = _get_sir_panel()
    with pytest.raises(ValueError, match="J should be greater than 0"):
        panel.dpop_train(
            J=0,
            M=2,
            eta=0.01,
            theta=deepcopy(panel.theta),
            process_weight_state="logw",
            key=jax.random.key(0),
        )


def test_panel_dpop_train_invalid_M():
    panel = _get_sir_panel()
    with pytest.raises(ValueError, match="M should be greater than 0"):
        panel.dpop_train(
            J=2,
            M=0,
            eta=0.01,
            theta=deepcopy(panel.theta),
            process_weight_state="logw",
            key=jax.random.key(0),
        )


def test_panel_dpop_train_missing_process_weight_state():
    panel = _get_sir_panel()
    with pytest.raises(ValueError, match="dpop_train requires a process-weight state"):
        panel.dpop_train(
            J=2,
            M=2,
            eta=0.01,
            theta=deepcopy(panel.theta),
            process_weight_state=None,
            key=jax.random.key(0),
        )


def test_panel_dpop_train_invalid_process_weight_state():
    panel = _get_sir_panel()
    with pytest.raises(ValueError, match="not found in statenames"):
        panel.dpop_train(
            J=2,
            M=2,
            eta=0.01,
            theta=deepcopy(panel.theta),
            process_weight_state="nonexistent_state",
            key=jax.random.key(0),
        )


def test_panel_dpop_train_chunk_size_consistency():
    """chunk_size=1 and chunk_size=U should give the same initial loglik."""
    panel = _get_sir_panel()
    J, M = 2, 1
    key = jax.random.key(42)

    panel.dpop_train(
        J=J, M=M, eta=0.01, theta=deepcopy(panel.theta),
        chunk_size=1, optimizer="SGD", alpha=0.8,
        process_weight_state="logw", key=key,
    )
    res1 = panel.results_history[-1]

    panel.dpop_train(
        J=J, M=M, eta=0.01, theta=deepcopy(panel.theta),
        chunk_size=2, optimizer="SGD", alpha=0.8,
        process_weight_state="logw", key=key,
    )
    res2 = panel.results_history[-1]

    # Initial loglik (iteration 0) should be identical regardless of chunk_size
    ll1_init = float(res1.shared_traces.sel(replicate=0, iteration=0, variable="logLik"))
    ll2_init = float(res2.shared_traces.sel(replicate=0, iteration=0, variable="logLik"))
    np.testing.assert_allclose(ll1_init, ll2_init, rtol=2e-3)


def test_panel_dpop_train_per_param_eta():
    """Per-parameter learning rates via dict eta."""
    panel = _get_sir_panel_with_shared()
    param_names = panel.canonical_param_names
    eta_dict = {p: 0.01 for p in param_names}
    # Set one shared param to have a different learning rate
    eta_dict["gamma"] = 0.001

    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=eta_dict,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer="Adam",
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)

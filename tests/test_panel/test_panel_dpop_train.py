from copy import deepcopy
from typing import Any
import jax
import numpy as np
import pytest
import pypomp as pp
from pypomp.core.results import PanelPompDpopTrainResult


# Short times series for fast test execution
_test_times = np.arange(1 / 52, 5 / 52, 1 / 52)


def _get_sir_panel():
    """Build a panel of 2 SIR units for testing panel DPOP train."""
    sir1 = pp.models.sir(seed=100, times=_test_times)
    sir2 = pp.models.sir(seed=200, times=_test_times)

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

    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])

    panel = pp.PanelPomp(
        Pomp_dict={"unit1": sir1, "unit2": sir2},
        theta=theta,
    )
    return panel


def _get_sir_panel_n_units(n_units):
    """Build a SIR panel with all parameters unit-specific."""
    import pandas as pd

    pomps = {
        f"unit{i + 1}": pp.models.sir(seed=100 + i, times=_test_times)
        for i in range(n_units)
    }
    first = next(iter(pomps.values()))
    param_names = first.canonical_param_names
    unit_specific = pd.DataFrame(
        {unit: [pomp.theta[0][p] for p in param_names] for unit, pomp in pomps.items()},
        index=pd.Index(param_names),
    )
    theta = pp.PanelParameters(theta=[{"shared": None, "unit_specific": unit_specific}])
    return pp.PanelPomp(Pomp_dict=pomps, theta=theta)


def _get_sir_panel_with_shared():
    """Build a panel of 2 SIR units with shared and unit-specific params."""
    sir1 = pp.models.sir(seed=100, times=_test_times)
    sir2 = pp.models.sir(seed=200, times=_test_times)

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


def test_panel_dpop_train_comprehensive():
    """Comprehensive test checking Adam optimizer, decay, alpha cooling, parameters change, and dimensions."""
    panel = _get_sir_panel()
    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer=pp.Adam(),
        alpha=0.8,
        alpha_cooling=0.5,
        decay=0.1,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    assert res.method == "dpop_train"
    assert res.shared_traces.dims == ("theta_idx", "iteration", "variable")
    assert res.unit_traces.dims == ("theta_idx", "iteration", "unit", "variable")
    assert res.shared_traces.shape[0] == 1  # n_reps
    assert res.shared_traces.shape[1] == M + 1
    assert res.unit_traces.shape[0] == 1  # n_reps
    assert res.unit_traces.shape[1] == M + 1
    assert res.unit_traces.shape[2] == len(panel.get_unit_names())  # U
    assert res.process_weight_state == "logw"
    assert res.decay == 0.1
    assert res.alpha == 0.8
    assert res.alpha_cooling == 0.5
    assert np.all(np.isfinite(np.asarray(res.shared_traces.sel(variable="logLik"))))

    # Check parameter change
    unit_vars = [
        v for v in res.unit_traces.coords["variable"].values if v != "unitLogLik"
    ]
    initial = res.unit_traces.sel(theta_idx=0, iteration=0, variable=unit_vars).values
    final = res.unit_traces.sel(theta_idx=0, iteration=M, variable=unit_vars).values
    assert not np.allclose(initial, final), "Parameters should change after training"


def test_panel_dpop_train_sgd():
    """Verify SGD optimizer runs successfully."""
    panel = _get_sir_panel()
    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer=pp.SGD(),
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(42),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    assert res.optimizer == pp.SGD()


def test_panel_dpop_train_shared_dataframe_and_eta():
    """Verify models with shared parameters, dict eta (per-param rates), and dataframe conversion."""
    panel = _get_sir_panel_with_shared()
    param_names = panel.canonical_param_names
    eta_dict = {p: 0.01 for p in param_names}
    eta_dict["gamma"] = 0.001

    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=eta_dict,
        theta=deepcopy(panel.theta),
        chunk_size=1,
        optimizer=pp.Adam(),
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

    df = res.to_dataframe()
    tr = res.traces()
    assert "theta_idx" in df.columns
    assert "theta_idx" in tr.columns
    assert "replicate" not in df.columns
    assert "replicate" not in tr.columns
    assert "shared logLik" in df.columns
    assert "unit logLik" in df.columns


@pytest.mark.parametrize("chunk_size", [3, 5], ids=["nondivisor", "larger_than_U"])
def test_panel_dpop_train_adjusts_nondividing_chunk_size(chunk_size):
    """Verify chunk size gets adjusted when it is invalid."""
    panel = _get_sir_panel()
    panel.dpop_train(
        J=2,
        M=2,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=chunk_size,
        optimizer=pp.Adam(),
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompDpopTrainResult)
    assert res.unit_traces.shape[2] == 2


def test_panel_dpop_train_multi_replicate():
    """Multiple replicates should produce independent traces."""
    panel = _get_sir_panel()

    # Build 2-replicate theta
    base_theta = panel.theta.params(as_list=True)[0]
    theta = pp.PanelParameters(theta=[deepcopy(base_theta), deepcopy(base_theta)])

    J, M = 2, 2
    panel.dpop_train(
        J=J,
        M=M,
        eta=0.01,
        theta=theta,
        chunk_size=1,
        optimizer=pp.Adam(),
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
    kwargs: dict[str, Any] = dict(
        J=J,
        M=M,
        eta=0.01,
        chunk_size=1,
        optimizer=pp.Adam(),
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(99),
    )

    panel1 = _get_sir_panel()
    panel1.dpop_train(theta=deepcopy(panel1.theta), **kwargs)
    res1 = panel1.results_history[-1]
    assert isinstance(res1, PanelPompDpopTrainResult)

    panel2 = _get_sir_panel()
    panel2.dpop_train(theta=deepcopy(panel2.theta), **kwargs)
    res2 = panel2.results_history[-1]
    assert isinstance(res2, PanelPompDpopTrainResult)

    np.testing.assert_array_equal(
        np.array(res1.unit_traces), np.array(res2.unit_traces)
    )


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


def test_panel_dpop_train_invalid_optimizer():
    panel = _get_sir_panel()
    # FullMatrixAdam optimizer is unsupported in low-level _panel_dpop_train_internal
    with pytest.raises(
        ValueError,
        match="Optimizer 'FullMatrixAdam' not supported for panel dpop_train",
    ):
        panel.dpop_train(
            J=2,
            M=2,
            eta=0.01,
            theta=deepcopy(panel.theta),
            optimizer=pp.FullMatrixAdam(),
            process_weight_state="logw",
            key=jax.random.key(0),
        )


def test_panel_dpop_train_invalid_theta_type():
    panel = _get_sir_panel()
    with pytest.raises(
        TypeError, match="theta must be a PanelParameters instance or None"
    ):
        panel.dpop_train(
            J=2,
            M=2,
            eta=0.01,
            theta="not_a_panel_parameters_object",  # type: ignore
            process_weight_state="logw",
            key=jax.random.key(0),
        )


def test_panel_dpop_train_missing_theta_and_self_theta():
    panel = _get_sir_panel()
    panel._theta = None  # type: ignore
    with pytest.raises(
        ValueError, match="theta must be provided or self.theta must exist"
    ):
        panel.dpop_train(
            J=2,
            M=2,
            eta=0.01,
            theta=None,
            process_weight_state="logw",
            key=jax.random.key(0),
        )

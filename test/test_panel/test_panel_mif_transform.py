"""
Integration tests for parameter transformations in PanelPomp.mif method.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pypomp as pp
import pytest


@pytest.fixture
def panel_pomp_with_transform():
    """Create a simple PanelPomp model with parameter transformation."""
    # Create two simple LG models for the panel
    LG1 = pp.LG()
    LG2 = pp.LG()

    # Define transformations
    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        result = {}
        for k, v in theta.items():
            if k.startswith("Q") or k.startswith("R"):
                result[k] = jnp.log(v)
            else:
                result[k] = v
        return result

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        result = {}
        for k, v in theta.items():
            if k.startswith("Q") or k.startswith("R"):
                result[k] = jnp.exp(v)
            else:
                result[k] = v
        return result

    # Set transformations on both models
    LG1.par_trans = pp.ParTrans(to_est, from_est)
    LG2.par_trans = pp.ParTrans(to_est, from_est)

    # Get initial values from one of the LG models to ensure consistency
    # LG models have all parameters: A1-A4, C1-C4, Q1-Q4, R1-R4
    # Use values from LG1 for all parameters
    theta_base = LG1.theta[0]

    # Create panel with shared parameters (A and C matrices) and unit-specific (Q and R)
    shared_param_names = ["A1", "A2", "A3", "A4", "C1", "C2", "C3", "C4"]
    unit_param_names = ["Q1", "Q2", "Q3", "Q4", "R1", "R2", "R3", "R4"]

    shared_params = pd.DataFrame(
        index=pd.Index(shared_param_names),
        data={"shared": [theta_base[name] for name in shared_param_names]},
    )

    unit_specific_params = pd.DataFrame(
        index=pd.Index(unit_param_names),
        data={
            "unit1": [theta_base[name] * 0.8 for name in unit_param_names],
            "unit2": [theta_base[name] * 1.2 for name in unit_param_names],
        },
    )

    panel = pp.PanelPomp(
        Pomp_dict={"unit1": LG1, "unit2": LG2},
        shared=shared_params,
        unit_specific=unit_specific_params,
    )

    return panel


def test_panel_mif_traces_transformed(panel_pomp_with_transform):
    """Test that PanelPomp mif traces are properly transformed back to natural space."""
    panel = panel_pomp_with_transform

    # Get canonical param names before running mif
    shared_names = panel.canonical_shared_param_names
    unit_names = panel.canonical_unit_param_names

    # Set up mif parameters
    all_param_names = list(shared_names) + list(unit_names)
    rw_sd = pp.RWSigma(
        sigmas={k: 0.02 for k in all_param_names},
        init_names=[],
    )

    # Run panel mif with small parameters for quick test
    panel.mif(J=5, M=2, rw_sd=rw_sd, a=0.5, key=jax.random.key(42))

    # Get the results
    results = panel.results_history[-1]

    # Check shared traces if they exist
    if "shared_traces" in results:
        shared_traces = results["shared_traces"]

        # Check that Q and R parameters (if in shared) are in natural space
        for param in shared_names:
            if param.startswith("Q") or param.startswith("R"):
                param_values = shared_traces.sel(variable=param).values
                param_values = param_values[~np.isnan(param_values)]
                assert np.all(param_values > 0), (
                    f"Shared parameter {param} should be positive"
                )

    # Check unit-specific traces if they exist
    if "unit_traces" in results:
        unit_traces = results["unit_traces"]

        # Check that Q and R parameters are in natural space
        for param in unit_names:
            if param.startswith("Q") or param.startswith("R"):
                param_values = unit_traces.sel(variable=param).values
                param_values = param_values[~np.isnan(param_values)]
                assert np.all(param_values > 0), (
                    f"Unit parameter {param} should be positive"
                )


def test_panel_transform_vectorized():
    """Test that transform_panel_traces works with vectorization."""

    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.log(v) if v > 0 else v for k, v in theta.items()}

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Create test traces
    # Shape: (n_reps=2, n_iters=3, n_shared+1=3) where [:,:,0] is loglik
    shared_traces = np.array(
        [
            [
                [np.nan, 1.0, 2.0],  # rep 0, iter 0
                [-10.0, 1.5, 2.5],  # rep 0, iter 1
                [-11.0, 2.0, 3.0],
            ],  # rep 0, iter 2
            [
                [np.nan, 0.5, 1.0],  # rep 1, iter 0
                [-9.0, 0.8, 1.2],  # rep 1, iter 1
                [-9.5, 1.0, 1.5],
            ],  # rep 1, iter 2
        ]
    )

    # Shape: (n_reps=2, n_iters=3, n_spec+1=2, n_units=2)
    unit_traces = np.array(
        [
            [
                [[np.nan, np.nan], [1.0, 1.5]],  # rep 0, iter 0
                [[-5.0, -6.0], [2.0, 2.5]],  # rep 0, iter 1
                [[-5.5, -6.5], [2.5, 3.0]],
            ],  # rep 0, iter 2
            [
                [[np.nan, np.nan], [0.5, 0.8]],  # rep 1, iter 0
                [[-4.0, -5.0], [1.0, 1.2]],  # rep 1, iter 1
                [[-4.5, -5.5], [1.5, 1.8]],
            ],  # rep 1, iter 2
        ]
    )

    shared_param_names = ["shared1", "shared2"]
    unit_param_names = ["unit1"]
    unit_names = ["u1", "u2"]

    # Transform from estimation to natural space
    shared_out, unit_out = par_trans.transform_panel_traces(
        shared_traces=shared_traces,
        unit_traces=unit_traces,
        shared_param_names=shared_param_names,
        unit_param_names=unit_param_names,
        unit_names=unit_names,
        direction="from_est",
    )

    assert shared_out is not None
    assert unit_out is not None

    # Check shapes preserved
    assert shared_out.shape == shared_traces.shape
    assert unit_out.shape == unit_traces.shape

    # Check logliks unchanged
    assert np.isnan(shared_out[0, 0, 0])
    assert np.abs(shared_out[0, 1, 0] - (-10.0)) < 1e-6

    # Check parameters transformed (exp of log values)
    assert shared_out[0, 0, 1] > shared_traces[0, 0, 1]  # exp(1) > 1
    assert unit_out[0, 0, 1, 0] > unit_traces[0, 0, 1, 0]  # exp(1) > 1

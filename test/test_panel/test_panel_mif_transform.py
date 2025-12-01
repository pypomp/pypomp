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
    """
    Test that with rw_sd=0, parameters remain unchanged after transformation cycle.
    """
    panel = panel_pomp_with_transform

    # Capture initial parameters in natural space before running mif
    # Deep copy to avoid mutations during mif

    initial_shared = [df.copy() for df in panel.shared]
    initial_unit_specific = [df.copy() for df in panel.unit_specific]

    # Get canonical param names before running mif
    shared_names = panel.canonical_shared_param_names
    unit_names = panel.canonical_unit_param_names

    # Set up mif parameters with zero random walk standard deviation
    # This means parameters should be transformed to perturbation scale,
    # remain unchanged, and then transformed back to natural scale
    all_param_names = list(shared_names) + list(unit_names)
    rw_sd = pp.RWSigma(
        sigmas={k: 0.0 for k in all_param_names},
        init_names=[],
    )

    panel.mif(J=2, M=1, rw_sd=rw_sd, a=0.5, key=jax.random.key(42))

    # Check that shared parameters are unchanged
    if initial_shared is not None and panel.shared is not None:
        # Compare initial and final shared parameters
        for rep_idx in range(len(panel.shared)):
            initial_df = initial_shared[rep_idx]
            final_df = panel.shared[rep_idx]

            for param in shared_names:
                initial_val = initial_df.loc[param, "shared"]
                final_val = final_df.loc[param, "shared"]
                assert np.allclose(
                    initial_val,
                    final_val,
                    rtol=1e-6,
                    atol=1e-6,
                ), (
                    f"Shared parameter {param} changed from {initial_val} to {final_val} "
                    "with rw_sd=0"
                )

    # Check that unit-specific parameters are unchanged
    if initial_unit_specific is not None and panel.unit_specific is not None:
        # Compare initial and final unit-specific parameters
        for rep_idx in range(len(panel.unit_specific)):
            initial_df = initial_unit_specific[rep_idx]
            final_df = panel.unit_specific[rep_idx]

            for param in unit_names:
                for unit in final_df.columns:
                    initial_val = initial_df.loc[param, unit]
                    final_val = final_df.loc[param, unit]
                    assert np.allclose(
                        initial_val,
                        final_val,
                        rtol=1e-6,
                        atol=1e-6,
                    ), (
                        f"Unit parameter {param} for {unit} changed from {initial_val} "
                        f"to {final_val} with rw_sd=0"
                    )

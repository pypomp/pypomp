"""
Integration tests for parameter transformations in mif and train methods.
These tests verify that traces are properly transformed from estimation space to natural space.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pypomp as pp
import pytest


@pytest.fixture
def simple_pomp_with_transform():
    """Create a simple POMP model with parameter transformation."""
    # Simple linear Gaussian model
    LG = pp.LG()

    # Define transformations that log-transform positive parameters
    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        # Transform Q and R parameters to log scale
        result = {}
        for k, v in theta.items():
            if k.startswith("Q") or k.startswith("R"):
                result[k] = jnp.log(v)
            else:
                result[k] = v
        return result

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        # Transform back from log scale
        result = {}
        for k, v in theta.items():
            if k.startswith("Q") or k.startswith("R"):
                result[k] = jnp.exp(v)
            else:
                result[k] = v
        return result

    # Set the transformation
    LG.par_trans = pp.ParTrans(to_est, from_est)

    return LG


def test_mif_traces_transformed(simple_pomp_with_transform):
    """Test that mif traces are properly transformed back to natural space."""
    LG = simple_pomp_with_transform

    # Set up mif parameters
    rw_sd = pp.RWSigma(
        sigmas={k: 0.02 for k in LG.canonical_param_names},
        init_names=[],
    )

    # Run mif with small parameters for quick test
    LG.mif(J=5, M=2, rw_sd=rw_sd, a=0.5, key=jax.random.key(42))

    # Get the traces
    traces = LG.results_history[-1]["traces"]

    # Check that traces exist and have the right structure
    assert traces is not None
    assert "variable" in traces.dims
    assert "iteration" in traces.dims

    # Get parameter names
    param_names = [v for v in traces.coords["variable"].values if v != "logLik"]

    # Check that Q and R parameters are in natural space (positive)
    for param in param_names:
        if param.startswith("Q") or param.startswith("R"):
            # All values should be positive (in natural space)
            param_values = traces.sel(variable=param).values
            # Skip NaN values at iteration 0
            param_values = param_values[~np.isnan(param_values)]
            assert np.all(param_values > 0), (
                f"Parameter {param} should be positive in natural space"
            )

            # Values should be reasonable (not in log space which would be negative or near zero)
            # In log space, these would typically be negative (log of values < 1)
            # In natural space, they should be around their initialized values
            assert np.all(param_values > 0.0001), (
                f"Parameter {param} values seem to be in log space"
            )


def test_train_traces_transformed(simple_pomp_with_transform):
    """Test that train traces are properly transformed back to natural space."""
    LG = simple_pomp_with_transform

    # Run train with small parameters for quick test
    LG.train(J=5, M=2, optimizer="Newton", key=jax.random.key(42))

    # Get the traces
    traces = LG.results_history[-1]["traces"]

    # Check that traces exist and have the right structure
    assert traces is not None
    assert "variable" in traces.dims
    assert "iteration" in traces.dims

    # Get parameter names
    param_names = [v for v in traces.coords["variable"].values if v != "logLik"]

    # Check that Q and R parameters are in natural space (positive)
    for param in param_names:
        if param.startswith("Q") or param.startswith("R"):
            # All values should be positive (in natural space)
            param_values = traces.sel(variable=param).values
            # Skip NaN values
            param_values = param_values[~np.isnan(param_values)]
            assert np.all(param_values > 0), (
                f"Parameter {param} should be positive in natural space"
            )

            # Values should be reasonable (not in log space)
            assert np.all(param_values > 0.0001), (
                f"Parameter {param} values seem to be in log space"
            )


def test_transform_roundtrip():
    """Test that transform_array does a proper round-trip transformation."""

    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Test with multiple parameter sets (like traces)
    original = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
            [2.0, 3.0, 4.0],
        ]
    )
    param_names = ["param1", "param2", "param3"]

    # Round trip: natural -> est -> natural
    est_space = par_trans.transform_array(original, param_names, direction="to_est")
    back_to_nat = par_trans.transform_array(
        est_space, param_names, direction="from_est"
    )

    # Should be very close to original
    assert np.allclose(back_to_nat, original, rtol=1e-6)

    # Verify that est_space is actually different (in log space)
    assert not np.allclose(est_space, original)
    # In log space, values should be different
    assert np.all(est_space < original)  # log(x) < x for x > 1

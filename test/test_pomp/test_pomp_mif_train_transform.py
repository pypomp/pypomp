"""
Integration tests for parameter transformations in mif and train methods.
These tests verify that traces are properly transformed from estimation space to natural space.
"""

import numpy as np
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
    """Test that with rw_sd=0, parameters remain unchanged after transformation cycle."""
    LG = simple_pomp_with_transform

    # Capture initial parameters in natural space before running mif
    # Deep copy to avoid mutations during mif
    initial_theta = [{k: v for k, v in theta.items()} for theta in LG.theta]

    # Set up mif parameters with zero random walk standard deviation
    # This means parameters should be transformed to perturbation scale,
    # remain unchanged, and then transformed back to natural scale
    rw_sd = pp.RWSigma(
        sigmas={k: 0.0 for k in LG.canonical_param_names},
        init_names=[],
    )

    # Run mif with zero rw_sd - parameters should remain unchanged
    LG.mif(J=2, M=1, rw_sd=rw_sd, a=0.5, key=jax.random.key(42))

    # Check that parameters are unchanged
    for rep_idx in range(len(LG.theta)):
        initial_params = initial_theta[rep_idx]
        final_params = LG.theta[rep_idx]

        for param_name in LG.canonical_param_names:
            initial_val = initial_params[param_name]
            final_val = final_params[param_name]
            assert np.allclose(
                initial_val,
                final_val,
                rtol=1e-6,
                atol=1e-6,
            ), (
                f"Parameter {param_name} changed from {initial_val} to {final_val} "
                "with rw_sd=0"
            )


def test_train_traces_transformed(simple_pomp_with_transform):
    """Test that with M=0, parameters remain unchanged after transformation cycle."""
    LG = simple_pomp_with_transform

    # Capture initial parameters in natural space before running train
    # Deep copy to avoid mutations during train
    initial_theta = [{k: v for k, v in theta.items()} for theta in LG.theta]

    # Run train with M=0 (no iterations) - parameters should be transformed
    # to estimation space, remain unchanged (no optimization), and transformed
    # back to natural scale
    LG.train(J=2, M=0, eta=0.2, optimizer="Newton", key=jax.random.key(42))

    # Check that parameters are unchanged
    for rep_idx in range(len(LG.theta)):
        initial_params = initial_theta[rep_idx]
        final_params = LG.theta[rep_idx]

        for param_name in LG.canonical_param_names:
            initial_val = initial_params[param_name]
            final_val = final_params[param_name]
            assert np.allclose(
                initial_val,
                final_val,
                rtol=1e-6,
                atol=1e-6,
            ), (
                f"Parameter {param_name} changed from {initial_val} to {final_val} "
                "with M=0"
            )

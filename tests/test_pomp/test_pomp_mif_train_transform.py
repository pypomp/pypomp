"""
Integration tests for parameter transformations in mif and train methods.
These tests verify that traces are properly transformed from estimation space to natural space.
"""

from copy import deepcopy
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.types import ParamDict


# Define transformations that log-transform positive parameters
def _to_est(theta: ParamDict) -> ParamDict:
    # Transform Q and R parameters to log scale
    result = {}
    for k, v in theta.items():
        if k.startswith("Q") or k.startswith("R"):
            result[k] = jnp.log(v)
        else:
            result[k] = v
    return result


def _from_est(theta: ParamDict) -> ParamDict:
    # Transform back from log scale
    result = {}
    for k, v in theta.items():
        if k.startswith("Q") or k.startswith("R"):
            result[k] = jnp.exp(v)
        else:
            result[k] = v
    return result


@pytest.fixture
def simple_pomp_with_transform():
    """Create a simple POMP model with parameter transformation."""
    # Simple linear Gaussian model
    LG = pp.models.lg()

    # Set the transformation
    LG.par_trans = pp.ParTrans(_to_est, _from_est)

    return LG


def test_mif_traces_transformed(simple_pomp_with_transform):
    """Test that with rw_sd=0, parameters remain unchanged after transformation cycle."""
    LG = simple_pomp_with_transform

    # Capture initial parameters in natural space before running mif
    # Deep copy to avoid mutations during mif
    initial_theta = [{k: v for k, v in theta.items()} for theta in LG.theta]

    # Set up mif parameters with zero random walk standard deviation
    rw_sd = pp.RWSigma(
        sigmas={k: 0.0 for k in LG.canonical_param_names},
        init_names=[],
    ).geometric_cooling(a=0.5)

    # Run mif with zero rw_sd - parameters should remain unchanged
    with patch.object(
        LG.par_trans,
        "_transform_array",
        wraps=LG.par_trans._transform_array,
    ) as spy_transform:
        LG.mif(J=2, M=1, rw_sd=rw_sd, key=jax.random.key(1))

        # Should transform to est space, then traces and final theta back from est space
        assert spy_transform.call_count == 3
        calls = spy_transform.call_args_list
        assert calls[0].kwargs.get("direction") == "to_est"
        assert calls[1].kwargs.get("direction") == "from_est"
        assert calls[2].kwargs.get("direction") == "from_est"

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
    eta = pp.LearningRate({param: 0.2 for param in LG.canonical_param_names})
    with patch.object(
        LG.par_trans,
        "_transform_array",
        wraps=LG.par_trans._transform_array,
    ) as spy_transform:
        LG.train(J=2, M=0, eta=eta, optimizer=pp.Newton(), key=jax.random.key(1))

        assert spy_transform.call_count == 2
        calls = spy_transform.call_args_list
        assert calls[0].kwargs.get("direction") == "to_est"
        assert calls[1].kwargs.get("direction") == "from_est"
        # Verify the intermediate parameter array differed from natural parameters
        thetas_est_arg = calls[1].args[0]
        thetas_nat_arg = calls[0].args[0]
        assert not np.allclose(
            np.asarray(thetas_est_arg[:, 0, :]), np.asarray(thetas_nat_arg)
        )

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

    # Check that self.theta.logLik is 1D
    assert LG.theta.logLik.ndim == 1
    assert LG.theta.logLik.shape == (LG.theta.num_replicates(),)


def test_functional_train_traces_transformed(simple_pomp_with_transform):
    """Test that F.train transforms parameters to est scale and back to natural scale."""
    LG = simple_pomp_with_transform
    struct = LG.to_struct()
    param_names = LG.canonical_param_names
    theta_array = LG.theta.to_jax_array(param_names)
    key = jax.random.key(123)
    keys = jnp.array(jax.random.split(key, theta_array.shape[0]))
    eta = pp.LearningRate({param: 0.01 for param in param_names})

    # Run with M=0: parameters should pass through to_est and from_est unchanged
    with patch.object(
        struct.par_trans,
        "_transform_array",
        wraps=struct.par_trans._transform_array,
    ) as spy_transform:
        neg_logliks, theta_traces = pp.functional.train(
            struct=struct,
            thetas_array=theta_array,
            J=2,
            M=0,
            eta=eta,
            keys=keys,
            optimizer=pp.Adam(),
            alpha=0.0,
        )

        assert spy_transform.call_count == 2
        calls = spy_transform.call_args_list
        assert calls[0].kwargs.get("direction") == "to_est"
        assert calls[1].kwargs.get("direction") == "from_est"
        # Verify intermediate estimation space differed from natural space (log scale)
        thetas_est_arg = calls[1].args[0]
        assert not np.allclose(
            np.asarray(thetas_est_arg[:, 0, :]), np.asarray(theta_array)
        )

    # Output trace should be in natural space and match initial natural parameters
    np.testing.assert_allclose(theta_traces[:, 0, :], theta_array, rtol=1e-6, atol=1e-6)
    for i, name in enumerate(param_names):
        if name.startswith("Q") or name.startswith("R"):
            assert np.all(theta_traces[:, :, i] > 0)


def test_train_parity_with_transform(simple_pomp_with_transform):
    """Test that Pomp.train matches F.train when non-trivial par_trans is present."""
    LG = simple_pomp_with_transform
    param_names = LG.canonical_param_names
    key = jax.random.key(123)
    eta = pp.LearningRate({param: 0.01 for param in param_names})
    optimizer = pp.Adam(scale=False, ls=False, c=0.0, max_ls_itn=1)

    theta_array = LG.theta.to_jax_array(param_names)

    # Run OOP train
    LG_copy = deepcopy(LG)
    LG_copy.train(J=2, M=1, eta=eta, optimizer=optimizer, key=key, alpha=0.0)
    result = LG_copy.results_history[-1]

    # Run functional train directly with natural parameters
    _, new_key = jax.random.split(key)
    keys = jnp.array(jax.random.split(new_key, theta_array.shape[0]))
    nLLs, theta_traces = pp.functional.train(
        LG.to_struct(),
        theta_array,
        J=2,
        M=1,
        eta=eta,
        keys=keys,
        optimizer=optimizer,
        alpha=0.0,
        alpha_cooling=1.0,
        thresh=0.0,
        n_monitors=1,
    )

    # Verify parity on natural scale
    np.testing.assert_allclose(
        np.asarray(result.payload["traces"].sel(variable="logLik")),
        -np.asarray(nLLs),
        rtol=1e-6,
        atol=1e-6,
    )
    for p in param_names:
        np.testing.assert_allclose(
            np.asarray(result.payload["traces"].sel(variable=p)),
            np.asarray(theta_traces)[:, :, param_names.index(p)],
            rtol=1e-6,
            atol=1e-6,
        )

    # Verify final theta logLik is 1D
    assert LG_copy.theta.logLik.ndim == 1
    assert LG_copy.theta.logLik.shape == (LG_copy.theta.num_replicates(),)


def test_mif_transform_before_tiling(simple_pomp_with_transform):
    """Test that Pomp.mif transforms parameters before tiling across J particles.

    Verifies that _transform_array with direction="to_est" receives a 2D parameter
    array (n_reps, n_params), rather than a 3D J-tiled array (n_reps, J, n_params).
    """
    LG = simple_pomp_with_transform
    rw_sd = pp.RWSigma(
        sigmas={k: 0.0 for k in LG.canonical_param_names},
        init_names=[],
    ).geometric_cooling(a=0.5)

    J = 5
    with patch.object(
        LG.par_trans,
        "_transform_array",
        wraps=LG.par_trans._transform_array,
    ) as spy_transform:
        LG.mif(J=J, M=1, rw_sd=rw_sd, key=jax.random.key(123))

        to_est_calls = [
            call
            for call in spy_transform.call_args_list
            if call.kwargs.get("direction") == "to_est"
        ]
        assert len(to_est_calls) >= 1
        thetas_arg = (
            to_est_calls[0].args[0]
            if to_est_calls[0].args
            else to_est_calls[0].kwargs.get("param_array")
        )
        assert thetas_arg is not None
        assert thetas_arg.ndim == 2, (
            f"thetas_arg has shape {thetas_arg.shape}, expected 2D without J={J}"
        )

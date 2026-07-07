"""
Tests for parameter transformation applied to traces and arrays in mif and train methods.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pypomp as pp
from pypomp.types import ParamDict
import pytest


def test_transform_array_single_param_set():
    """Test transform_array with a single parameter set (1D array)."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {
            "pos_param": jnp.log(theta["pos_param"]),
            "standard_param": theta["standard_param"],
        }

    def from_est(theta: ParamDict) -> ParamDict:
        return {
            "pos_param": jnp.exp(theta["pos_param"]),
            "standard_param": theta["standard_param"],
        }

    par_trans = pp.ParTrans(to_est, from_est)

    # Test NumPy input
    param_array_np = np.array([1.0, 2.0])
    param_names = ["pos_param", "standard_param"]

    est_np = par_trans._transform_array(param_array_np, param_names, direction="to_est")
    assert isinstance(est_np, np.ndarray)
    assert est_np.shape == (2,)
    assert np.abs(est_np[0] - np.log(1.0)) < 1e-6
    assert np.abs(est_np[1] - 2.0) < 1e-6

    nat_np = par_trans._transform_array(est_np, param_names, direction="from_est")
    assert isinstance(nat_np, np.ndarray)
    assert nat_np.shape == (2,)
    assert np.abs(nat_np[0] - 1.0) < 1e-6
    assert np.abs(nat_np[1] - 2.0) < 1e-6

    # Test JAX input
    param_array_jax = jnp.array([1.0, 2.0])
    est_jax = par_trans._transform_array(
        param_array_jax, param_names, direction="to_est"
    )
    assert isinstance(est_jax, jax.Array)
    assert est_jax.shape == (2,)
    assert np.abs(est_jax[0] - np.log(1.0)) < 1e-6

    nat_jax = par_trans._transform_array(est_jax, param_names, direction="from_est")
    assert isinstance(nat_jax, jax.Array)
    assert np.abs(nat_jax[0] - 1.0) < 1e-6


def test_transform_array_multiple_param_sets():
    """Test transform_array with multiple parameter sets (2D array)."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {
            "pos_param": jnp.log(theta["pos_param"]),
            "standard_param": theta["standard_param"],
        }

    def from_est(theta: ParamDict) -> ParamDict:
        return {
            "pos_param": jnp.exp(theta["pos_param"]),
            "standard_param": theta["standard_param"],
        }

    par_trans = pp.ParTrans(to_est, from_est)

    param_array = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    param_names = ["pos_param", "standard_param"]

    est_array = par_trans._transform_array(param_array, param_names, direction="to_est")
    assert est_array.shape == (3, 2)
    assert np.abs(est_array[0, 0] - np.log(1.0)) < 1e-6
    assert np.abs(est_array[1, 0] - np.log(2.0)) < 1e-6
    assert np.abs(est_array[2, 0] - np.log(3.0)) < 1e-6

    nat_array = par_trans._transform_array(est_array, param_names, direction="from_est")
    assert nat_array.shape == (3, 2)
    assert np.allclose(nat_array, param_array, rtol=1e-6)


def test_transform_array_default_transformation():
    """Test transform_array with default (identity) transformation."""
    par_trans = pp.ParTrans()

    param_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    param_names = ["param1", "param2"]

    est_array = par_trans._transform_array(param_array, param_names, direction="to_est")
    assert np.allclose(est_array, param_array)

    nat_array = par_trans._transform_array(est_array, param_names, direction="from_est")
    assert np.allclose(nat_array, param_array)


def test_transform_panel_array_shared_only():
    """Test transform_panel_array with shared parameters only."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Shape: (n_reps, n_iters, n_shared)
    shared_array = np.array(
        [
            [
                [2.0, 3.0],  # iter 0: shared_param1=log(2), shared_param2=log(3)
                [2.5, 3.5],  # iter 1: shared_param1=log(2.5), shared_param2=log(3.5)
            ]
        ]
    )

    shared_param_names = ["shared_param1", "shared_param2"]

    shared_out, unit_out = par_trans._transform_panel_array(
        shared_array=shared_array,
        unit_array=None,
        shared_names=shared_param_names,
        unit_specific_names=[],
        direction="from_est",
    )

    assert isinstance(shared_out, np.ndarray)
    assert unit_out is None
    assert shared_out.shape == shared_array.shape

    # Check parameters are transformed: exp(x)
    assert np.abs(shared_out[0, 0, 0] - np.exp(2.0)) < 1e-6
    assert np.abs(shared_out[0, 0, 1] - np.exp(3.0)) < 1e-6
    assert np.abs(shared_out[0, 1, 0] - np.exp(2.5)) < 1e-6
    assert np.abs(shared_out[0, 1, 1] - np.exp(3.5)) < 1e-6


def test_transform_panel_array_unit_only():
    """Test transform_panel_array with unit-specific parameters only."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Shape: (n_reps, n_iters, n_units, n_spec)
    unit_array = np.array(
        [
            [
                [
                    [1.0],  # iter 0, unit 1: unit_param1
                    [1.5],  # iter 0, unit 2: unit_param1
                ],
                [
                    [2.0],  # iter 1, unit 1: unit_param1
                    [2.5],  # iter 1, unit 2: unit_param1
                ],
            ]
        ]
    )

    unit_param_names = ["unit_param1"]

    shared_out, unit_out = par_trans._transform_panel_array(
        shared_array=None,
        unit_array=unit_array,
        shared_names=[],
        unit_specific_names=unit_param_names,
        direction="from_est",
    )

    assert shared_out is None
    assert isinstance(unit_out, np.ndarray)
    assert unit_out.shape == unit_array.shape

    # Check parameters are transformed
    assert np.abs(unit_out[0, 0, 0, 0] - np.exp(1.0)) < 1e-6
    assert np.abs(unit_out[0, 0, 1, 0] - np.exp(1.5)) < 1e-6
    assert np.abs(unit_out[0, 1, 0, 0] - np.exp(2.0)) < 1e-6
    assert np.abs(unit_out[0, 1, 1, 0] - np.exp(2.5)) < 1e-6


def test_transform_panel_array_both():
    """Test transform_panel_array with both shared and unit-specific parameters."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) if v > 0 else v for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Shape: (n_reps, n_iters, n_shared)
    shared_array = np.array(
        [
            [
                [1.0],  # iter 0
                [1.5],  # iter 1
            ]
        ]
    )

    # Shape: (n_reps, n_iters, n_units, n_spec)
    unit_array = np.array(
        [
            [
                [
                    [2.0],  # iter 0, unit 1
                    [2.5],  # iter 0, unit 2
                ],
                [
                    [3.0],  # iter 1, unit 1
                    [3.5],  # iter 1, unit 2
                ],
            ]
        ]
    )

    shared_param_names = ["shared_param"]
    unit_param_names = ["unit_param"]

    # Test NumPy input preservation
    shared_out, unit_out = par_trans._transform_panel_array(
        shared_array=shared_array,
        unit_array=unit_array,
        shared_names=shared_param_names,
        unit_specific_names=unit_param_names,
        direction="from_est",
    )

    assert isinstance(shared_out, np.ndarray)
    assert isinstance(unit_out, np.ndarray)
    assert shared_out.shape == shared_array.shape
    assert unit_out.shape == unit_array.shape

    assert np.abs(shared_out[0, 0, 0] - np.exp(1.0)) < 1e-6
    assert np.abs(shared_out[0, 1, 0] - np.exp(1.5)) < 1e-6
    assert np.abs(unit_out[0, 0, 0, 0] - np.exp(2.0)) < 1e-6
    assert np.abs(unit_out[0, 1, 1, 0] - np.exp(3.5)) < 1e-6

    # Test JAX input preservation
    shared_out_jax, unit_out_jax = par_trans._transform_panel_array(
        shared_array=jnp.array(shared_array),
        unit_array=jnp.array(unit_array),
        shared_names=shared_param_names,
        unit_specific_names=unit_param_names,
        direction="from_est",
    )
    assert isinstance(shared_out_jax, jax.Array)
    assert isinstance(unit_out_jax, jax.Array)


def test_transform_roundtrip():
    """Test that transform_array does a proper round-trip transformation."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    original = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
            [2.0, 3.0, 4.0],
        ]
    )
    param_names = ["param1", "param2", "param3"]

    est_space = par_trans._transform_array(original, param_names, direction="to_est")
    back_to_nat = par_trans._transform_array(
        est_space, param_names, direction="from_est"
    )

    assert np.allclose(back_to_nat, original, rtol=1e-6)
    assert not np.allclose(est_space, original)
    assert np.all(est_space < original)


def test_transform_array_invalid_direction():
    """Test that _transform_array raises ValueError on invalid direction."""
    par_trans = pp.ParTrans()
    with pytest.raises(ValueError, match="Invalid direction"):
        par_trans._transform_array(np.array([1.0]), ["p1"], direction="invalid")  # type: ignore


def test_transform_array_3d():
    """Test _transform_array with a 3D parameter trace (e.g. n_reps, n_iters, n_params)."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    original = np.array(
        [
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
        ]
    )
    param_names = ["p1", "p2"]

    est = par_trans._transform_array(original, param_names, direction="to_est")
    assert est.shape == (2, 3, 2)
    assert np.allclose(est, np.log(original))

    nat = par_trans._transform_array(est, param_names, direction="from_est")
    assert nat.shape == (2, 3, 2)
    assert np.allclose(nat, original)


def test_transform_panel_array_invalid_direction():
    """Test that _transform_panel_array raises ValueError on invalid direction."""
    par_trans = pp.ParTrans()
    with pytest.raises(ValueError, match="Invalid direction"):
        par_trans._transform_panel_array(None, None, [], [], direction="invalid")  # type: ignore


def test_transform_panel_array_both_none():
    """Test that _transform_panel_array returns (None, None) when both traces are None."""
    par_trans = pp.ParTrans()
    shared_out, unit_out = par_trans._transform_panel_array(
        shared_array=None,
        unit_array=None,
        shared_names=["shared"],
        unit_specific_names=["unit"],
        direction="to_est",
    )
    assert shared_out is None
    assert unit_out is None


def test_transform_panel_array_unit_traces_only_no_spec():
    """Test that unit_array is simply returned unchanged when unit_array is not None but n_spec == 0."""
    par_trans = pp.ParTrans()

    unit_array = np.array([[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]])

    shared_out, unit_out = par_trans._transform_panel_array(
        shared_array=None,
        unit_array=unit_array,
        shared_names=["shared"],
        unit_specific_names=[],  # n_spec = 0
        direction="to_est",
    )

    assert shared_out is None
    assert unit_out is not None
    assert np.allclose(unit_out, unit_array)


def test_transform_panel_array_no_shared_context():
    """Test unit trace transformation when shared_array is None but n_shared > 0."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {
            "unit": theta["unit"] + theta.get("shared", 0.0),
            "shared": theta.get("shared", 0.0),
        }

    par_trans = pp.ParTrans(to_est=to_est)

    unit_array = np.array([[[[10.0]]]])  # unit=10.0

    shared_out, unit_out = par_trans._transform_panel_array(
        shared_array=None,
        unit_array=unit_array,
        shared_names=["shared"],  # n_shared = 1
        unit_specific_names=["unit"],
        direction="to_est",
    )

    assert shared_out is None
    assert unit_out is not None
    # 10.0 + shared_context (which should be 0.0 because shared_array is None) = 10.0
    assert abs(unit_out[0, 0, 0, 0] - 10.0) < 1e-6


def test_transform_panel_array_no_unit_context():
    """Test shared trace transformation when unit_array is None but n_spec > 0."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {
            "shared": theta["shared"] + theta.get("unit", 0.0),
            "unit": theta.get("unit", 0.0),
        }

    par_trans = pp.ParTrans(to_est=to_est)

    shared_array = np.array([[[10.0]]])  # shared=10.0

    shared_out, unit_out = par_trans._transform_panel_array(
        shared_array=shared_array,
        unit_array=None,
        shared_names=["shared"],
        unit_specific_names=["unit"],  # n_spec = 1
        direction="to_est",
    )

    assert unit_out is None
    assert shared_out is not None
    # 10.0 + unit_context (which should be 0.0 because unit_array is None) = 10.0
    assert abs(shared_out[0, 0, 0] - 10.0) < 1e-6

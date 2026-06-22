"""
Tests for parameter transformation applied to traces in mif and train methods.
"""

import numpy as np
import jax.numpy as jnp
import pypomp as pp
from pypomp.types import ParamDict


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

    # Test single parameter set
    param_array = np.array([1.0, 2.0])  # log(1.0) should stay ~0, 2.0 stays 2.0
    param_names = ["pos_param", "standard_param"]

    # Transform to estimation space
    est_array = par_trans._transform_array(param_array, param_names, direction="to_est")
    assert est_array.shape == (2,)
    assert np.abs(est_array[0] - np.log(1.0)) < 1e-6
    assert np.abs(est_array[1] - 2.0) < 1e-6

    # Transform back to natural space
    nat_array = par_trans._transform_array(est_array, param_names, direction="from_est")
    assert nat_array.shape == (2,)
    assert np.abs(nat_array[0] - 1.0) < 1e-6
    assert np.abs(nat_array[1] - 2.0) < 1e-6


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

    # Test multiple parameter sets (like a trace with M iterations)
    param_array = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    param_names = ["pos_param", "standard_param"]

    # Transform to estimation space
    est_array = par_trans._transform_array(param_array, param_names, direction="to_est")
    assert est_array.shape == (3, 2)
    assert np.abs(est_array[0, 0] - np.log(1.0)) < 1e-6
    assert np.abs(est_array[1, 0] - np.log(2.0)) < 1e-6
    assert np.abs(est_array[2, 0] - np.log(3.0)) < 1e-6
    assert np.abs(est_array[0, 1] - 2.0) < 1e-6
    assert np.abs(est_array[1, 1] - 3.0) < 1e-6
    assert np.abs(est_array[2, 1] - 4.0) < 1e-6

    # Transform back to natural space
    nat_array = par_trans._transform_array(est_array, param_names, direction="from_est")
    assert nat_array.shape == (3, 2)
    assert np.allclose(nat_array, param_array, rtol=1e-6)


def test_transform_array_default_transformation():
    """Test transform_array with default (identity) transformation."""
    par_trans = pp.ParTrans()  # Default: no transformation

    param_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    param_names = ["param1", "param2"]

    # Should be unchanged
    est_array = par_trans._transform_array(param_array, param_names, direction="to_est")
    assert np.allclose(est_array, param_array)

    nat_array = par_trans._transform_array(est_array, param_names, direction="from_est")
    assert np.allclose(nat_array, param_array)


def test_transform_panel_traces_shared_only():
    """Test transform_panel_traces with shared parameters only."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Shape: (n_reps, n_iters, n_shared+1)
    shared_traces = np.array(
        [
            [
                [
                    np.nan,
                    2.0,
                    3.0,
                ],  # iter 0: loglik=nan, shared_param1=log(2), shared_param2=log(3)
                [-10.5, 2.5, 3.5],
            ],  # iter 1: loglik=-10.5, shared_param1=log(2.5), shared_param2=log(3.5)
        ]
    )

    shared_param_names = ["shared_param1", "shared_param2"]
    unit_param_names = []

    shared_out, unit_out = par_trans._transform_panel_traces(
        shared_traces=shared_traces,
        unit_traces=None,
        shared_param_names=shared_param_names,
        unit_param_names=unit_param_names,
        direction="from_est",
    )

    assert shared_out is not None
    assert unit_out is None
    assert shared_out.shape == shared_traces.shape

    # Check logliks are unchanged
    assert np.isnan(shared_out[0, 0, 0])
    assert np.abs(shared_out[0, 1, 0] - (-10.5)) < 1e-6

    # Check parameters are transformed: exp(log(x)) = x
    assert np.abs(shared_out[0, 0, 1] - np.exp(2.0)) < 1e-6
    assert np.abs(shared_out[0, 0, 2] - np.exp(3.0)) < 1e-6
    assert np.abs(shared_out[0, 1, 1] - np.exp(2.5)) < 1e-6
    assert np.abs(shared_out[0, 1, 2] - np.exp(3.5)) < 1e-6


def test_transform_panel_traces_unit_only():
    """Test transform_panel_traces with unit-specific parameters only."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Shape: (n_reps, n_iters, n_units, n_spec+1)
    unit_traces = np.array(
        [
            [
                [
                    [np.nan, 1.0],  # iter 0, unit 1: loglik, unit_param1
                    [np.nan, 1.5],  # iter 0, unit 2: loglik, unit_param1
                ],
                [
                    [-5.0, 2.0],  # iter 1, unit 1: loglik, unit_param1
                    [-6.0, 2.5],  # iter 1, unit 2: loglik, unit_param1
                ],
            ],
        ]
    )

    shared_param_names = []
    unit_param_names = ["unit_param1"]

    shared_out, unit_out = par_trans._transform_panel_traces(
        shared_traces=None,
        unit_traces=unit_traces,
        shared_param_names=shared_param_names,
        unit_param_names=unit_param_names,
        direction="from_est",
    )

    assert shared_out is None
    assert unit_out is not None
    assert unit_out.shape == unit_traces.shape

    # Check logliks are unchanged
    assert np.isnan(unit_out[0, 0, 0, 0])
    assert np.abs(unit_out[0, 1, 0, 0] - (-5.0)) < 1e-6

    # Check parameters are transformed
    assert np.abs(unit_out[0, 0, 0, 1] - np.exp(1.0)) < 1e-6
    assert np.abs(unit_out[0, 0, 1, 1] - np.exp(1.5)) < 1e-6
    assert np.abs(unit_out[0, 1, 0, 1] - np.exp(2.0)) < 1e-6
    assert np.abs(unit_out[0, 1, 1, 1] - np.exp(2.5)) < 1e-6


def test_transform_panel_traces_both():
    """Test transform_panel_traces with both shared and unit-specific parameters."""

    def to_est(theta: ParamDict) -> ParamDict:
        # Simple log transformation for all params
        return {k: jnp.log(v) if v > 0 else v for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        # Exp transformation for all params
        return {k: jnp.exp(v) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Shape: (n_reps, n_iters, n_shared+1)
    shared_traces = np.array(
        [
            [
                [np.nan, 1.0],  # iter 0
                [-10.0, 1.5],
            ],  # iter 1
        ]
    )

    # Shape: (n_reps, n_iters, n_units, n_spec+1)
    unit_traces = np.array(
        [
            [
                [
                    [np.nan, 2.0],  # iter 0, unit 1: unit_param
                    [np.nan, 2.5],  # iter 0, unit 2: unit_param
                ],
                [
                    [-5.0, 3.0],  # iter 1, unit 1: unit_param
                    [-6.0, 3.5],  # iter 1, unit 2: unit_param
                ],
            ],
        ]
    )

    shared_param_names = ["shared_param"]
    unit_param_names = ["unit_param"]

    shared_out, unit_out = par_trans._transform_panel_traces(
        shared_traces=shared_traces,
        unit_traces=unit_traces,
        shared_param_names=shared_param_names,
        unit_param_names=unit_param_names,
        direction="from_est",
    )

    assert shared_out is not None
    assert unit_out is not None

    # Check shapes
    assert shared_out.shape == shared_traces.shape
    assert unit_out.shape == unit_traces.shape

    # Check logliks unchanged
    assert np.isnan(shared_out[0, 0, 0])
    assert np.abs(shared_out[0, 1, 0] - (-10.0)) < 1e-6

    # Check transformed parameters
    assert np.abs(shared_out[0, 0, 1] - np.exp(1.0)) < 1e-6
    assert np.abs(shared_out[0, 1, 1] - np.exp(1.5)) < 1e-6
    assert np.abs(unit_out[0, 0, 0, 1] - np.exp(2.0)) < 1e-6
    assert np.abs(unit_out[0, 1, 1, 1] - np.exp(3.5)) < 1e-6


def test_transform_roundtrip():
    """Test that transform_array does a proper round-trip transformation."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.log(v) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
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
    est_space = par_trans._transform_array(original, param_names, direction="to_est")
    back_to_nat = par_trans._transform_array(
        est_space, param_names, direction="from_est"
    )

    # Should be very close to original
    assert np.allclose(back_to_nat, original, rtol=1e-6)

    # Verify that est_space is actually different (in log space)
    assert not np.allclose(est_space, original)
    # In log space, values should be different
    assert np.all(est_space < original)  # log(x) < x for x > 1


def test_transform_array_invalid_direction():
    """Test that _transform_array raises ValueError on invalid direction."""
    import pytest

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

    # shape: (2, 3, 2)
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


def test_transform_panel_traces_invalid_direction():
    """Test that _transform_panel_traces raises ValueError on invalid direction."""
    import pytest

    par_trans = pp.ParTrans()
    with pytest.raises(ValueError, match="Invalid direction"):
        par_trans._transform_panel_traces(None, None, [], [], direction="invalid")  # type: ignore


def test_transform_panel_traces_both_none():
    """Test that _transform_panel_traces returns (None, None) when both traces are None."""
    par_trans = pp.ParTrans()
    shared_out, unit_out = par_trans._transform_panel_traces(
        shared_traces=None,
        unit_traces=None,
        shared_param_names=["shared"],
        unit_param_names=["unit"],
        direction="to_est",
    )
    assert shared_out is None
    assert unit_out is None


def test_transform_panel_traces_unit_traces_only_no_spec():
    """Test that unit_traces is simply copied when unit_traces is not None but n_spec == 0."""
    par_trans = pp.ParTrans()

    # unit_traces has shape (n_reps, n_iters, n_units, n_spec+1)
    # Here n_spec is 0, so shape is (1, 2, 3, 1)
    unit_traces = np.array([[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]])

    shared_out, unit_out = par_trans._transform_panel_traces(
        shared_traces=None,
        unit_traces=unit_traces,
        shared_param_names=["shared"],
        unit_param_names=[],  # n_spec = 0
        direction="to_est",
    )

    assert shared_out is None
    assert unit_out is not None
    assert np.allclose(unit_out, unit_traces)


def test_transform_panel_traces_no_shared_context():
    """Test unit trace transformation when shared_traces is None but n_shared > 0."""

    def to_est(theta: ParamDict) -> ParamDict:
        # unit transformation depends on shared value (e.g. unit + shared)
        # but if shared is missing/None in ctx, it will fall back to 0.0 in zeros context
        return {
            "unit": theta["unit"] + theta.get("shared", 0.0),
            "shared": theta.get("shared", 0.0),
        }

    par_trans = pp.ParTrans(to_est=to_est)

    # unit_traces shape (1, 1, 1, 2) -> n_reps=1, n_iters=1, n_units=1, n_spec=1
    unit_traces = np.array([[[[0.5, 10.0]]]])  # loglik=0.5, unit=10.0

    shared_out, unit_out = par_trans._transform_panel_traces(
        shared_traces=None,
        unit_traces=unit_traces,
        shared_param_names=["shared"],  # n_shared = 1
        unit_param_names=["unit"],
        direction="to_est",
    )

    assert shared_out is None
    assert unit_out is not None
    # 10.0 + shared_context (which should be 0.0 because shared_traces is None) = 10.0
    assert abs(unit_out[0, 0, 0, 1] - 10.0) < 1e-6


def test_transform_panel_traces_no_unit_context():
    """Test shared trace transformation when unit_traces is None but n_spec > 0."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {
            "shared": theta["shared"] + theta.get("unit", 0.0),
            "unit": theta.get("unit", 0.0),
        }

    par_trans = pp.ParTrans(to_est=to_est)

    # shared_traces shape (1, 1, 2) -> n_reps=1, n_iters=1, n_shared=1
    shared_traces = np.array([[[0.5, 10.0]]])  # loglik=0.5, shared=10.0

    shared_out, unit_out = par_trans._transform_panel_traces(
        shared_traces=shared_traces,
        unit_traces=None,
        shared_param_names=["shared"],
        unit_param_names=["unit"],  # n_spec = 1
        direction="to_est",
    )

    assert unit_out is None
    assert shared_out is not None
    # 10.0 + unit_context (which should be 0.0 because unit_traces is None) = 10.0
    assert abs(shared_out[0, 0, 1] - 10.0) < 1e-6

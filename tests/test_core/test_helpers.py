import pypomp.core.algorithms.helpers as ifunc
import numpy as np


def test_calc_ys_covars():
    t0 = -1.0
    times = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    ctimes = np.array([0, 1, 2, 3, 4, 5])
    covars = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    order = "linear"
    dt = 0.5

    interp_covars, dt_array_ext, nstep_array, max_steps_per_interval = (
        ifunc._calc_ys_covars(t0, times, ctimes, covars, dt, None, order)
    )

    # Check that the first and last times in dt_array_ext correspond to t0 and times[-1]
    times0 = np.concatenate((np.array([t0]), np.array(times)))
    nstep_array, dt_array = ifunc._calc_steps(times0, dt, None)
    dt_array_ext_expected = np.repeat(dt_array, nstep_array)
    assert np.allclose(np.array(dt_array_ext), dt_array_ext_expected)

    # Check shapes
    assert dt_array_ext.shape[0] == np.sum(nstep_array)
    assert interp_covars is not None
    assert interp_covars.shape[1] == covars.shape[1]


def test_pad_array():
    import jax.numpy as jnp

    # Test padding when pad_width > 0
    arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    # Pad along axis 0 from size 2 to size 4
    padded = ifunc.pad_array(arr, axis=0, padded_size=4, size=2)
    expected = jnp.array([[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    assert jnp.array_equal(padded, expected)

    # Test padding along axis 1 from size 2 to size 3
    padded_axis1 = ifunc.pad_array(arr, axis=1, padded_size=3, size=2)
    expected_axis1 = jnp.array([[1.0, 2.0, 2.0], [3.0, 4.0, 4.0]])
    assert jnp.array_equal(padded_axis1, expected_axis1)

    # Test no padding needed
    no_pad = ifunc.pad_array(arr, axis=0, padded_size=2, size=2)
    assert jnp.array_equal(no_pad, arr)


def test_merge_and_slice_unsharded():
    import jax.numpy as jnp

    # out_axis is None
    # 0D array
    scalar = jnp.array(5.0)
    assert (
        ifunc.merge_and_slice(
            scalar, out_axis=None, size=2, num_batches=2, batch_size=1
        )
        == scalar
    )

    # 1D array, num_batches > 1 (takes first batch)
    arr = jnp.array([10.0, 20.0])
    assert (
        ifunc.merge_and_slice(arr, out_axis=None, size=2, num_batches=2, batch_size=1)
        == 10.0
    )

    # 1D array, num_batches == 1
    assert jnp.array_equal(
        ifunc.merge_and_slice(arr, out_axis=None, size=2, num_batches=1, batch_size=2),
        arr,
    )

    # Non-array input
    assert (
        ifunc.merge_and_slice(
            "not an array", out_axis=None, size=2, num_batches=2, batch_size=1
        )
        == "not an array"
    )


def test_merge_and_slice_sharded_non_array():
    # non-array input
    assert (
        ifunc.merge_and_slice(
            "not an array", out_axis=0, size=2, num_batches=2, batch_size=1
        )
        == "not an array"
    )

    # 0D array
    import jax.numpy as jnp

    scalar = jnp.array(5.0)
    assert (
        ifunc.merge_and_slice(scalar, out_axis=0, size=2, num_batches=2, batch_size=1)
        == scalar
    )


def test_merge_outputs_unsupported_type():
    import pytest

    with pytest.raises(TypeError, match="Unsupported shard_output_axes type"):
        ifunc.merge_outputs(
            scanned_out=None,
            shard_output_axes=set(),  # set is unsupported
            size=2,
            num_batches=2,
            batch_size=1,
        )


def test_interp_covars_linear_and_constant():
    # 1. Direct test of _interp_covars with linear interpolation
    ctimes = np.array([0.0, 1.0, 2.0])
    covars = np.array([10.0, 20.0, 30.0])

    # t inside bounds
    val = ifunc._interp_covars(0.5, ctimes, covars, order="linear")
    assert val is not None
    assert np.allclose(val, 15.0)
    # t outside lower bound (extrapolates)
    val = ifunc._interp_covars(-0.5, ctimes, covars, order="linear")
    assert val is not None
    assert np.allclose(val, 5.0)

    # t outside upper bound (extrapolates)
    val = ifunc._interp_covars(2.5, ctimes, covars, order="linear")
    assert val is not None
    assert np.allclose(val, 35.0)

    # array of t
    t_arr = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    val = ifunc._interp_covars(t_arr, ctimes, covars, order="linear")
    assert val is not None
    expected = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    assert np.allclose(val, expected)
    # 2. Direct test of _interp_covars with constant interpolation
    # t inside bounds (right-continuous step function)
    # [0.0, 1.0) -> 10.0
    # [1.0, 2.0) -> 20.0
    # [2.0, inf) -> 30.0
    # t < 0.0 -> 10.0
    for t_val, expected_val in [
        (-0.5, 10.0),
        (0.0, 10.0),
        (0.5, 10.0),
        (1.0, 20.0),
        (1.5, 20.0),
        (2.0, 30.0),
        (2.5, 30.0),
    ]:
        val = ifunc._interp_covars(t_val, ctimes, covars, order="constant")
        assert val is not None
        assert np.allclose(val, expected_val)

    # array of t for constant interpolation
    val = ifunc._interp_covars(t_arr, ctimes, covars, order="constant")
    assert val is not None
    expected_const = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
    assert np.allclose(val, expected_const)

    # 3. Test multi-dimensional covariates
    covars_2d = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    val_2d = ifunc._interp_covars(t_arr, ctimes, covars_2d, order="constant")
    assert val_2d is not None
    expected_2d = np.array(
        [
            [10.0, 100.0],
            [10.0, 100.0],
            [10.0, 100.0],
            [20.0, 200.0],
            [20.0, 200.0],
            [30.0, 300.0],
            [30.0, 300.0],
        ]
    )
    assert np.allclose(val_2d, expected_2d)

    # 4. Test unsupported interpolation order raises ValueError
    import pytest

    with pytest.raises(ValueError, match="Unsupported interpolation order"):
        ifunc._interp_covars(0.5, ctimes, covars, order="spline")


def test_pomp_constant_interpolation():
    import pandas as pd
    import pypomp as pp

    ys = pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0])
    covars = pd.DataFrame({"cov1": [10.0, 20.0]}, index=[0.0, 2.0])
    theta = pp.PompParameters({"X0": 0.0, "sigma": 0.1})

    def rinit(theta_, key, covars, t0):
        return {"X": theta_["X0"]}

    def rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"]}

    def dmeas(Y_, X_, theta_, covars, t):
        return 0.0

    # Initialize Pomp with order="constant"
    model = pp.Pomp(
        ys=ys,
        theta=theta,
        statenames=["X"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        covars=covars,
        nstep=2,
        order="constant",
    )

    # Check that model._covars_extended is not None and has constant interpolation
    cov_ext = model._covars_extended
    assert cov_ext is not None
    expected_cov = np.array([10.0, 10.0, 10.0, 10.0, 20.0])
    assert np.allclose(cov_ext.ravel(), expected_cov)

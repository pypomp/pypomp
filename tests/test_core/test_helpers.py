import os
from unittest.mock import patch

import numpy as np
import pytest

import pypomp.core.algorithms.helpers as ifunc


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


def test_plan_sharding_does_not_pad_below_the_device_count():
    """Fewer replicates than devices narrows the mesh rather than padding.

    Padding is not free: `pad_array` repeats the last replicate, and those
    duplicates are computed in full before `merge_outputs` slices them off, so
    padding one replicate up to four devices would be four times the work for
    one replicate's worth of output.
    """
    for size in (1, 2, 3):
        plan = ifunc.plan_sharding(size, num_devices=4, is_cpu=True)
        assert plan.num_shard_devices == size
        assert plan.num_batches == 1
        assert plan.padded_size == size, (
            f"{size} replicate(s) on 4 devices would compute {plan.padded_size}; "
            "the surplus is padding that is calculated and then discarded"
        )


def test_plan_sharding_uses_every_device_when_replicates_match():
    plan = ifunc.plan_sharding(4, num_devices=4, is_cpu=True)
    assert plan == ifunc.ShardPlan(num_shard_devices=4, batch_size=4, num_batches=1)
    assert plan.padded_size == 4


def test_plan_sharding_batches_cpu_work_that_exceeds_the_devices():
    # 10 replicates over 4 CPU devices: three sequential batches of four, the
    # last of which is two-thirds padding. Padding is unavoidable here, since
    # the batches have to be a uniform shape to scan over.
    plan = ifunc.plan_sharding(10, num_devices=4, is_cpu=True)
    assert plan == ifunc.ShardPlan(num_shard_devices=4, batch_size=4, num_batches=3)
    assert plan.padded_size == 12


def test_plan_sharding_pads_accelerators_only_above_the_device_count():
    # Accelerators take the single-step path at any size, so more replicates
    # than devices is the one case that still pads.
    assert ifunc.plan_sharding(3, num_devices=8, is_cpu=False).padded_size == 3
    assert ifunc.plan_sharding(12, num_devices=8, is_cpu=False).padded_size == 16


def test_plan_sharding_invariants():
    """Properties that must hold however the replicates fall across devices."""
    for num_devices in range(1, 9):
        for size in range(0, 20):
            for is_cpu in (True, False):
                plan = ifunc.plan_sharding(size, num_devices, is_cpu=is_cpu)
                assert 1 <= plan.num_shard_devices <= num_devices
                # Never drop a replicate, and never compute more than one extra
                # mesh-width worth of padding.
                assert size <= plan.padded_size < size + plan.num_shard_devices
                # The mesh has to divide the work it is handed evenly.
                assert plan.batch_size % plan.num_shard_devices == 0
                # Below the device count there is nothing to pad.
                if size <= num_devices:
                    assert plan.padded_size == size


# `plan_sharding` covers the arithmetic without needing devices; this checks
# that the plan is what actually reaches the sharded function. It needs a
# subprocess because the JAX device count is fixed when JAX is first imported
# and the suite itself runs on a single device.
_SHARD_PROBE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_shard_probe.py"
)


def test_sharding_computes_no_more_replicates_than_it_was_given():
    import json
    import subprocess
    import sys

    sizes = (1, 3, 4)
    # Hand the child this process's import path so that it exercises the same
    # pypomp the test itself imported, rather than whichever copy happens to be
    # first on a default sys.path.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    proc = subprocess.run(
        [sys.executable, _SHARD_PROBE, *(str(s) for s in sizes)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (
        f"shard probe failed\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    report = json.loads(proc.stdout)

    assert report["devices"] == 4
    for size in sizes:
        entry = report["sizes"][str(size)]
        assert entry["computed"] == [size], (
            f"{size} replicate(s) on 4 devices computed {entry['computed']}; "
            "the surplus is padding that is calculated and then discarded"
        )
        assert entry["out"] == [2.0 * i for i in range(size)]


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


def test_interp_covars_returns_none_when_covars_or_ctimes_missing():
    ctimes = np.array([0.0, 1.0, 2.0])
    covars = np.array([10.0, 20.0, 30.0])

    assert ifunc._interp_covars(0.5, None, None) is None
    assert ifunc._interp_covars(0.5, None, covars) is None
    assert ifunc._interp_covars(0.5, ctimes, None) is None


def test_num_fixedstep_steps_requires_nstep():
    with pytest.raises(ValueError, match="nstep must be provided"):
        ifunc._num_fixedstep_steps(0.0, 1.0, None, None)


def test_num_euler_steps_requires_dt():
    with pytest.raises(ValueError, match="dt must be provided"):
        ifunc._num_euler_steps(0.0, 1.0, None, None)


def test_num_euler_steps_non_positive_interval_returns_zero():
    # t1 >= t2: no steps needed for a zero/negative-length interval.
    assert ifunc._num_euler_steps(1.0, 1.0, 0.1, None) == (0, 0.0)
    assert ifunc._num_euler_steps(2.0, 1.0, 0.1, None) == (0, 0.0)


def test_calc_steps_argument_validation():
    times0 = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Only nstep or dt can be provided, not both"):
        ifunc._calc_steps(times0, dt=0.1, nstep=5)

    with pytest.raises(ValueError, match="Either dt or nstep must be provided"):
        ifunc._calc_steps(times0, dt=None, nstep=None)


def test_is_dynamic_exception_fallback():
    """Exercise the ``except`` branch of ``is_dynamic`` directly.

    ``jax.tree_util.tree_leaves`` essentially never raises for ordinary
    inputs (unregistered objects are just treated as opaque leaves), so the
    fallback logic is only reachable by forcing the primary path to fail.
    """
    import jax.numpy as jnp

    with patch("jax.tree_util.tree_leaves", side_effect=RuntimeError("boom")):
        assert ifunc.is_dynamic(jnp.array([1.0])) is True
        assert ifunc.is_dynamic(np.array([1.0])) is True
        assert ifunc.is_dynamic([jnp.array([1.0]), 2]) is True
        assert ifunc.is_dynamic((1, jnp.array([1.0]))) is True
        assert ifunc.is_dynamic({"a": jnp.array([1.0])}) is True
        assert ifunc.is_dynamic({"a": 2}) is False
        assert ifunc.is_dynamic([1, 2]) is False
        assert ifunc.is_dynamic(5) is False


def test_run_jax_batch_sharded_kwargs_dynamic_and_static():
    """Direct kwargs (dynamic array-valued and static) reach the sharded func.

    In this single-CPU-device test environment, any ``size > 1`` call takes
    the CPU sequential-batching path (``num_batches > 1``); none of the
    library's internal callers pass kwargs through ``run_jax_batch_sharded``,
    so this is the only way to exercise the kwargs-splitting loop.
    """
    import jax.numpy as jnp

    assert len(__import__("jax").devices()) == 1

    def f(x, c, *, bias, label):
        assert label == "add"
        return x + c + bias

    x = jnp.arange(4.0).reshape(4, 1)
    out = ifunc.run_jax_batch_sharded(
        f, {0: 0}, 0, x, 1.0, bias=jnp.array(2.0), label="add"
    )
    expected = x + 1.0 + 2.0
    assert np.allclose(np.asarray(out), np.asarray(expected))


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

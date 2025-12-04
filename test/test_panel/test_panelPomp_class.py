import pandas as pd
import xarray as xr
import jax
import pypomp as pp
import numpy as np
import pickle
import time


def test_get_unit_parameters(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    params = panel.get_unit_parameters(unit="London")
    assert isinstance(params, list)
    assert isinstance(params[0], dict)
    assert len(params) == panel.theta.num_replicates()


def test_results(measles_panel_mp):
    panel, *_ = measles_panel_mp
    results0 = panel.results(0)
    results1 = panel.results(1)
    # Check expected columns for results0 (from mif, index=0)
    expected_cols = {
        "replicate",
        "unit",
        "shared logLik",
        "unit logLik",
    }
    # Add dynamic parameter columns (shared/unit) if present.
    results0_cols = set(results0.columns)
    assert expected_cols.issubset(results0_cols)
    # It should have one row for each replicate/unit combination
    n_units = len(panel.unit_objects)
    n_reps = results0["replicate"].nunique()
    assert len(results0) == n_units * n_reps

    # Check expected columns for results1 (from pfilter, index=1)
    results1_cols = set(results1.columns)
    assert expected_cols.issubset(results1_cols)
    n_units1 = len(panel.unit_objects)
    n_reps1 = results1["replicate"].nunique()
    assert len(results1) == n_units1 * n_reps1


def test_fresh_key(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    J = 2
    panel.pfilter(J=J, key=key)
    assert not np.array_equal(
        jax.random.key_data(panel.fresh_key), jax.random.key_data(key)
    )
    assert np.array_equal(
        jax.random.key_data(panel.results_history[0].key),
        jax.random.key_data(key),
    )


def test_traces(measles_panel_mp):
    panel, *_ = measles_panel_mp
    traces = panel.traces()
    assert isinstance(traces, pd.DataFrame)
    expected_column_order = ["replicate", "unit", "iteration", "method", "logLik"]
    assert list(traces.columns[:5]) == expected_column_order, (
        f"First five columns are {list(traces.columns[:5])}, expected {expected_column_order}"
    )

    traces_sorted = traces.sort_values(
        ["replicate", "unit", "iteration"], kind="stable"
    ).reset_index(drop=True)
    assert traces.equals(traces_sorted)

    grouped = traces.groupby(["replicate", "unit"], sort=False)
    for (rep, unit), df in grouped:  # type: ignore[misc]
        if len(df) == 0:
            continue

        # For non-pfilter methods (e.g., MIF, train), iterations should increase
        # from 0 up to the number of such entries minus one.
        non_pfilter = df[df["method"] != "pfilter"]
        if len(non_pfilter) > 0:
            non_pfilter_iters = np.sort(np.asarray(non_pfilter["iteration"]))
            expected_seq = np.arange(len(non_pfilter_iters))
            assert bool(np.array_equal(non_pfilter_iters, expected_seq)), (
                f"Iteration numbers for replicate={rep}, unit={unit} and "
                f"method!=pfilter are {non_pfilter_iters}, expected {expected_seq}"
            )

        # For pfilter entries, the iteration counter should not advance;
        # they are allowed to repeat the last non-pfilter iteration.
        pfilter = df[df["method"] == "pfilter"]
        if len(pfilter) > 0 and len(non_pfilter) > 0:
            pfilter_iters = np.asarray(pfilter["iteration"])
            last_non_pfilter_iter = non_pfilter["iteration"].max()
            assert bool(np.all(pfilter_iters == last_non_pfilter_iter)), (
                f"Pfilter iteration numbers for replicate={rep}, unit={unit} "
                f"are {pfilter_iters}, expected all {last_non_pfilter_iter}"
            )


def test_time(measles_panel_mp):
    panel, *_ = measles_panel_mp
    time_df = panel.time()
    assert isinstance(time_df, pd.DataFrame)
    assert "method" in time_df.columns
    assert "time" in time_df.columns
    assert time_df.index.name == "history_index"


def test_pickle_panelpomp(measles_panel_mp):
    panel, *_ = measles_panel_mp

    pickled = pickle.dumps(panel)
    unpickled_panel = pickle.loads(pickled)

    # check equality of panel object and unpickled object
    assert panel == unpickled_panel

    # check that the unpickled panel can be used for filtering
    unpickled_panel.pfilter(J=2)


def test_sample_params(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    param_bounds = {
        "R0": [10.0, 60.0],
        "sigma": [25.0, 100.0],
        "gamma": [25.0, 320.0],
        "iota": [0.004, 3.0],
        "rho": [0.1, 0.9],
        "sigmaSE": [0.04, 0.1],
        "psi": [0.05, 3.0],
        "cohort": [0.1, 0.7],
        "amplitude": [0.1, 0.6],
        "S_0": [0.01, 0.07],
        "E_0": [0.000004, 0.0001],
        "I_0": [0.000003, 0.001],
        "R_0": [0.9, 0.99],
    }
    shared_names = ["gamma", "cohort"]
    param_sets = panel.sample_params(
        param_bounds=param_bounds,
        units=list(panel.unit_objects.keys()),
        n=2,
        key=key,
        shared_names=shared_names,
    )
    assert isinstance(param_sets, list)
    assert len(param_sets) == 2

    # Check that shared_param_sets DataFrames have correct index and column in correct order
    for param_set in param_sets:
        shared_df = param_set["shared"]
        # Index should be shared_names, in order
        assert list(shared_df.index) == shared_names
        # Only one column which is exactly ["shared"]
        assert list(shared_df.columns) == ["shared"]

    # For unit_specific_param_sets, check columns and index are correct
    units = list(panel.unit_objects.keys())
    unit_specific_names = [
        name for name in param_bounds if name not in set(shared_names)
    ]
    for param_set in param_sets:
        unit_df = param_set["unit_specific"]
        # Columns (units) are in the correct order
        assert list(unit_df.columns) == units
        # Index should be unit_specific_names, in order
        assert list(unit_df.index) == unit_specific_names

    # Also check that each value is within the specified bounds
    for param_set in param_sets:
        shared_df = param_set["shared"]
        for name in shared_names:
            val = shared_df.loc[name, "shared"]
            lower, upper = param_bounds[name]
            assert lower <= val <= upper

    for param_set in param_sets:
        unit_df = param_set["unit_specific"]
        for param_name in unit_specific_names:
            lower, upper = param_bounds[param_name]
            for unit in units:
                val = unit_df.loc[param_name, unit]
                assert lower <= val <= upper


def test_performance_comprehensive():
    """Test that results() and traces() run in under 2 seconds with many units, replications, and iterations."""
    # Create a comprehensive panel with many units, replications, and iterations
    units = [f"unit_{i}" for i in range(40)]  # 40 units
    pomp_objects = {}

    # Create minimal pomp objects
    for unit in units:
        times = np.linspace(0, 90, 10)  # Use numeric times with larger spacing
        ys = pd.DataFrame({"cases": np.random.poisson(10, 10)}, index=times)
        pomp_obj = pp.Pomp(
            ys=ys,
            theta={
                "param1": 1.0,
                "param2": 2.0,
                "unit_param1": 0.5,
                "unit_param2": 0.5,
            },
            statenames=["S", "I"],
            t0=float(times[0]),
            rinit=lambda theta_, key, covars, t0: {"S": 1000, "I": 1},
            rproc=lambda X_, theta_, key, covars, t, dt: X_,
            dmeas=lambda Y_, X_, theta_, covars, t: 0.0,
            nstep=1,
        )
        pomp_objects[unit] = pomp_obj

    # Create shared and unit-specific parameters
    shared_params = pd.DataFrame(
        {"shared": [1.0, 2.0]}, index=pd.Index(["param1", "param2"])
    )
    unit_specific_params = pd.DataFrame(
        {
            unit: [np.random.uniform(0.1, 1.0), np.random.uniform(0.1, 1.0)]
            for unit in units
        },
        index=pd.Index(["unit_param1", "unit_param2"]),
    )

    # Create panel
    panel = pp.PanelPomp(
        Pomp_dict=pomp_objects,
        theta=[{"shared": shared_params, "unit_specific": unit_specific_params}],
    )

    # Create comprehensive dummy results to stress test
    n_reps = 30  # 30 replicates
    n_iter = 15  # 15 iterations per MIF
    n_units = len(units)

    # Create dummy shared traces
    shared_traces = xr.DataArray(
        np.random.randn(n_reps, n_iter + 1, 3),  # +1 for initial values
        dims=["replicate", "iteration", "variable"],
        coords={
            "replicate": range(n_reps),
            "iteration": range(n_iter + 1),
            "variable": ["logLik", "param1", "param2"],
        },
    )

    # Create dummy unit traces
    unit_traces = xr.DataArray(
        np.random.randn(n_reps, n_iter + 1, 3, n_units),  # +1 for initial values
        dims=["replicate", "iteration", "variable", "unit"],
        coords={
            "replicate": range(n_reps),
            "iteration": range(n_iter + 1),
            "variable": ["unitLogLik", "unit_param1", "unit_param2"],
            "unit": units,
        },
    )

    # Create dummy loglikelihoods
    logLiks = xr.DataArray(
        np.random.randn(n_reps, n_units + 1),  # +1 for shared
        dims=["replicate", "unit"],
        coords={"replicate": range(n_reps), "unit": ["shared"] + units},
    )

    # Add multiple MIF results to history (stress test with many results)
    from pypomp.results import PanelPompMIFResult, PanelPompPFilterResult

    for i in range(6):  # 6 MIF runs
        result = PanelPompMIFResult(
            method="mif",
            execution_time=1.0,
            key=jax.random.key(42),
            theta=pp.PanelParameters(
                [{"shared": shared_params, "unit_specific": unit_specific_params}]
            )
            * n_reps,  # type: ignore[reportArgumentType]
            shared_traces=shared_traces,
            unit_traces=unit_traces,
            logLiks=logLiks,
            J=100,
            M=n_iter,
            rw_sd=pp.RWSigma(
                {
                    "param1": 0.1,
                    "param2": 0.1,
                    "unit_param1": 0.1,
                    "unit_param2": 0.1,
                }
            ),
            a=0.1,
            thresh=0.0,
            block=True,
        )
        panel.results_history.add(result)

    # Add some pfilter results too (stress test with mixed result types)
    for i in range(4):  # 4 pfilter runs
        pfilter_logLiks = xr.DataArray(
            np.random.randn(n_reps, n_units, 3),  # 3 replicates per pfilter
            dims=["theta", "unit", "replicate"],
            coords={"theta": range(n_reps), "unit": units, "replicate": range(3)},
        )
        result = PanelPompPFilterResult(
            method="pfilter",
            execution_time=1.0,
            key=jax.random.key(42),
            theta=pp.PanelParameters(
                [{"shared": shared_params, "unit_specific": unit_specific_params}]
            )
            * n_reps,  # type: ignore[reportArgumentType]
            logLiks=pfilter_logLiks,
            J=100,
            reps=3,
            thresh=0.0,
        )
        panel.results_history.add(result)

    # Test results() performance
    start_time = time.time()
    results_df = panel.results()
    end_time = time.time()

    assert end_time - start_time < 2.0, (
        f"results() took {end_time - start_time:.2f} seconds, expected < 2.0"
    )
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) > 0

    # Test traces() performance
    start_time = time.time()
    traces_df = panel.traces()
    end_time = time.time()

    assert end_time - start_time < 2.0, (
        f"traces() took {end_time - start_time:.2f} seconds, expected < 2.0"
    )
    assert isinstance(traces_df, pd.DataFrame)
    assert len(traces_df) > 0


def test_prune(measles_panel_mp):
    panel, rw_sd, key, J, M, a = measles_panel_mp

    # Get initial state from the last result
    results_df = panel.results()
    initial_n_reps = results_df["replicate"].nunique()
    assert initial_n_reps > 1, "Need multiple replicates to test prune"

    # Store initial parameter lists
    initial_shared_len = len(panel.theta._theta) if panel.theta._theta else 0

    # Store original parameters from results_history for comparison
    original_results = panel.results_history[-1]
    original_theta = original_results.theta
    if original_theta is not None:
        original_shared = [t.get("shared") for t in original_theta._theta]
        original_unit_specific = [t.get("unit_specific") for t in original_theta._theta]
    else:
        original_shared = None
        original_unit_specific = None

    # Test pruning to top 1 replicate without refill
    panel.prune(n=1, refill=False)

    # Check that theta has been updated to length 1
    assert len(panel.theta._theta) == 1

    # Verify that the pruned parameters match the top replicate
    # Get the top replicate index from original results
    if original_results.method == "mif":
        logLiks = original_results.logLiks
        # Get shared log-likelihoods (first column is "shared")
        shared_lls = logLiks[:, 0].values  # shape: (n_reps,)
        top_idx = int(np.argmax(shared_lls))

        # Check that pruned parameters match the top replicate
        if (
            original_shared is not None
            and panel.theta._theta[0].get("shared") is not None
        ):
            pd.testing.assert_frame_equal(
                panel.theta._theta[0]["shared"], original_shared[top_idx]
            )
        if (
            original_unit_specific is not None
            and panel.theta._theta[0].get("unit_specific") is not None
        ):
            pd.testing.assert_frame_equal(
                panel.theta._theta[0]["unit_specific"], original_unit_specific[top_idx]
            )

    # Test refill functionality
    # Restore to initial state from results_history
    panel.theta._theta = (
        [
            {
                "shared": s.copy() if s is not None else None,
                "unit_specific": u.copy() if u is not None else None,
            }
            for s, u in zip(original_shared, original_unit_specific)  # type: ignore[reportArgumentType]
        ]
        if original_shared is not None
        else []
    )

    # Prune with refill=True
    panel.prune(n=1, refill=True)

    # Check that lists are refilled to original length
    assert len(panel.theta._theta) == initial_shared_len
    # All entries should be the same (repeated top replicate)
    if initial_shared_len > 1:
        first_shared = panel.theta._theta[0].get("shared")
        first_unit_specific = panel.theta._theta[0].get("unit_specific")
        for i in range(1, initial_shared_len):
            if first_shared is not None:
                pd.testing.assert_frame_equal(
                    panel.theta._theta[i]["shared"], first_shared
                )
            if first_unit_specific is not None:
                pd.testing.assert_frame_equal(
                    panel.theta._theta[i]["unit_specific"], first_unit_specific
                )


def test_mix_and_match(measles_panel_mp):
    panel, rw_sd, key, J, M, a = measles_panel_mp

    results_df = panel.results()
    initial_n_reps = results_df["replicate"].nunique()
    assert initial_n_reps > 1, "Need multiple replicates to test mix_and_match"

    original_theta = panel.results_history[-1].theta
    if original_theta is None:
        return

    original_shared = [t.get("shared") for t in original_theta._theta]
    original_unit_specific = [t.get("unit_specific") for t in original_theta._theta]
    unit_names = list(panel.unit_objects.keys())

    panel.mix_and_match()
    assert len(panel.theta._theta) == initial_n_reps

    # Compute rankings (same logic as PanelParameters.mix_and_match)
    shared_ranks = original_theta.logLik.argsort()[::-1].tolist()
    unit_name_to_idx = {
        name: idx for idx, name in enumerate(original_theta.get_unit_names())
    }
    unit_ranks = {
        unit: original_theta.logLik_unit[:, unit_name_to_idx[unit]]
        .argsort()[::-1]
        .tolist()
        for unit in unit_names
    }

    # Helper to verify a replicate has correct mixed parameters
    def verify_replicate(rep_idx, rank_idx):
        new_shared = panel.theta.theta[rep_idx]["shared"]
        orig_shared = original_shared[shared_ranks[rank_idx]]
        if new_shared is not None and orig_shared is not None:
            pd.testing.assert_frame_equal(new_shared, orig_shared)
        elif new_shared != orig_shared:
            raise AssertionError(f"Mismatch: {type(new_shared)} vs {type(orig_shared)}")

        if original_unit_specific:
            new_spec = panel.theta.theta[rep_idx]["unit_specific"]
            if new_spec is not None:
                for unit in unit_names:
                    orig_col = original_unit_specific[unit_ranks[unit][rank_idx]].get(
                        unit
                    )
                    new_col = new_spec.get(unit)
                    if orig_col is not None and new_col is not None:
                        pd.testing.assert_series_equal(
                            orig_col, new_col, check_dtype=False
                        )

    verify_replicate(0, 0)
    if initial_n_reps >= 2:
        verify_replicate(1, 1)

    # Sanity check: verify structure (mixing may or may not have occurred)
    if (
        original_unit_specific
        and panel.theta._theta[0].get("unit_specific") is not None
    ):
        orig_cols = {unit: original_unit_specific[0][unit] for unit in unit_names}
        new_cols = {
            unit: panel.theta._theta[0]["unit_specific"][unit] for unit in unit_names
        }
        # Verify comparison runs (mixing may or may not have occurred)
        _ = any(not orig_cols[u].equals(new_cols[u]) for u in unit_names)


def test_print_summary(measles_panel_mp):
    panel, rw_sd, key, J, M, a = measles_panel_mp
    panel.print_summary()

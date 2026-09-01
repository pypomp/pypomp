"""Wall-clock behavior of PanelPomp accessors on a wide panel.

Marked heavy: it asserts elapsed time, so it needs the machine to itself rather
than competing with the rest of the suite under xdist.
"""

import time

import jax
import numpy as np
import pandas as pd
import xarray as xr

import pypomp as pp


def test_performance_comprehensive():
    """Test that results() and traces() run in under 5 seconds with many units, replications, and iterations."""
    # Create a comprehensive panel with many units, replications, and iterations
    units = [f"unit_{i}" for i in range(40)]  # 40 units
    pomp_objects = {}

    def rinit_fn(theta_, key, covars, t0):
        return {"S": 1000, "I": 1}

    def rproc_fn(X_, theta_, key, covars, t, dt):
        return X_

    def dmeas_fn(Y_, X_, theta_, covars, t):
        return 0.0

    # Create minimal pomp objects
    for unit in units:
        times = np.linspace(0, 90, 10)  # Use numeric times with larger spacing
        ys = pd.DataFrame({"cases": np.random.poisson(10, 10)}, index=times)
        pomp_obj = pp.Pomp(
            ys=ys,
            theta=pp.PompParameters(
                {
                    "param1": 1.0,
                    "param2": 2.0,
                    "unit_param1": 0.5,
                    "unit_param2": 0.5,
                }
            ),
            statenames=["S", "I"],
            t0=float(times[0]),
            rinit=rinit_fn,
            rproc=rproc_fn,
            dmeas=dmeas_fn,
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
        pomp_dict=pomp_objects,
        theta=pp.PanelParameters(
            [{"shared": shared_params, "unit_specific": unit_specific_params}]
        ),
    )

    # Create comprehensive dummy results to stress test
    n_reps = 30  # 30 replicates
    n_iter = 15  # 15 iterations per MIF
    n_units = len(units)

    # Create dummy shared traces
    shared_traces = xr.DataArray(
        np.random.randn(n_reps, n_iter + 1, 3),  # +1 for initial values
        dims=["theta_idx", "iteration", "variable"],
        coords={
            "theta_idx": range(n_reps),
            "iteration": range(n_iter + 1),
            "variable": ["logLik", "param1", "param2"],
        },
    )

    # Create dummy unit traces
    unit_traces = xr.DataArray(
        np.random.randn(n_reps, n_iter + 1, n_units, 3),  # +1 for initial values
        dims=["theta_idx", "iteration", "unit", "variable"],
        coords={
            "theta_idx": range(n_reps),
            "iteration": range(n_iter + 1),
            "unit": units,
            "variable": ["unitLogLik", "unit_param1", "unit_param2"],
        },
    )

    # Create dummy loglikelihoods
    logLiks = xr.DataArray(
        np.random.randn(n_reps, n_units + 1),  # +1 for shared
        dims=["theta_idx", "unit"],
        coords={"theta_idx": range(n_reps), "unit": ["shared"] + units},
    )

    # Add multiple MIF results to history (stress test with many results)
    from pypomp.core.results import (
        build_panel_mif_result,
        build_panel_pfilter_result,
    )

    for _i in range(6):  # 6 MIF runs
        result = build_panel_mif_result(
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
            ).geometric_cooling(0.1),
            thresh=0.0,
            n_monitors=0,
            block=True,
        )
        panel.results_history.add(result)

    # Add some pfilter results too (stress test with mixed result types)
    for _i in range(4):  # 4 pfilter runs
        pfilter_logLiks = xr.DataArray(
            np.random.randn(n_reps, n_units, 3),  # 3 replicates per pfilter
            dims=["theta_idx", "unit", "rep"],
            coords={"theta_idx": range(n_reps), "unit": units, "rep": range(3)},
        )
        result = build_panel_pfilter_result(
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

    assert end_time - start_time < 5.0, (
        f"results() took {end_time - start_time:.2f} seconds, expected < 5.0"
    )
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) > 0

    # Test traces() performance
    start_time = time.time()
    traces_df = panel.traces()
    end_time = time.time()

    assert end_time - start_time < 5.0, (
        f"traces() took {end_time - start_time:.2f} seconds, expected < 5.0"
    )
    assert isinstance(traces_df, pd.DataFrame)
    assert len(traces_df) > 0

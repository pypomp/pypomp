"""Builders producing concrete Result objects via the public builder API."""

import jax
import numpy as np
import xarray as xr

from pypomp.core.parameters import PompParameters
from pypomp.core.results import build_mif_result, build_pfilter_result

KEY = jax.random.key(0)


def pomp_pfilter_result(execution_time=1.5, with_diag=True):
    theta = PompParameters({"param1": 1.0, "param2": 2.0})
    logLiks = xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"])
    cll = ess = None
    if with_diag:
        cll = xr.DataArray(
            [[[0.5, 0.6], [0.7, 0.8]]], dims=["theta_idx", "rep", "time"]
        )
        ess = xr.DataArray(
            [[[10.0, 20.0], [30.0, 40.0]]], dims=["theta_idx", "rep", "time"]
        )
    return build_pfilter_result(
        key=KEY,
        execution_time=execution_time,
        theta=theta,
        logLiks=logLiks,
        J=1000,
        reps=2,
        thresh=0.5,
        CLL=cll,
        ESS=ess,
    )


def pomp_mif_result(execution_time=1.0, rw_sd=None):
    theta = PompParameters({"param1": 1.0})
    traces = xr.DataArray(
        [[[1.5, 1.0], [2.5, 1.1]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0, 1],
            "variable": ["logLik", "param1"],
        },
    )
    return build_mif_result(
        key=KEY,
        execution_time=execution_time,
        theta=theta,
        traces=traces,
        J=100,
        M=2,
        rw_sd=rw_sd,
        thresh=0.8,
        n_monitors=5,
    )


def panel_shared_unit(shared_val=1.0, unit_val=2.0):
    shared_traces = xr.DataArray(
        [[[shared_val]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik"]},
    )
    unit_traces = xr.DataArray(
        [[[[unit_val]]]],
        dims=["theta_idx", "iteration", "unit", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0],
            "unit": ["u1"],
            "variable": ["unitLogLik"],
        },
    )
    logLiks = xr.DataArray(
        [[np.nan, np.nan]],
        dims=["theta_idx", "unit"],
        coords={"theta_idx": [0], "unit": ["shared", "u1"]},
    )
    return shared_traces, unit_traces, logLiks

"""Panel (:class:`~pypomp.panel.panel.PanelPomp`) result builders.

Panel "trace" results hold three arrays (``shared_traces``, ``unit_traces``,
``logLiks``) whose ``variable``/``unit`` dims would collide inside a single
:class:`xarray.Dataset`. The builders store them under collision-free internal
dim names (``unit_variable``, ``ll_unit``); the accessor on
:class:`~pypomp.core.results.result.Result` restores the public names on read
(see ``_DIM_UNALIAS`` there).
"""

from __future__ import annotations

from typing import Any

import jax
import xarray as xr

from .result import Result


def _dataset(**data_vars: xr.DataArray | None) -> xr.Dataset:
    return xr.Dataset({k: v for k, v in data_vars.items() if v is not None})


def _trace_payload(
    shared_traces: xr.DataArray,
    unit_traces: xr.DataArray,
    logLiks: xr.DataArray,
) -> xr.Dataset:
    """Store the three panel-trace arrays under collision-free dim names."""
    return _dataset(
        shared_traces=shared_traces,
        unit_traces=unit_traces.rename({"variable": "unit_variable"}),
        logLiks=logLiks.rename({"unit": "ll_unit"}),
    )


def build_panel_pfilter_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    logLiks: xr.DataArray,
    J: int,
    reps: int,
    thresh: float,
    CLL: xr.DataArray | None = None,
    ESS: xr.DataArray | None = None,
    filter_mean: xr.DataArray | None = None,
    prediction_mean: xr.DataArray | None = None,
) -> Result:
    return Result(
        method="pfilter",
        kind="table",
        panel=True,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={"J": J, "reps": reps, "thresh": thresh},
        payload=_dataset(
            logLiks=logLiks,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
        ),
    )


def build_panel_mif_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    shared_traces: xr.DataArray,
    unit_traces: xr.DataArray,
    logLiks: xr.DataArray,
    J: int,
    M: int,
    rw_sd: Any,
    thresh: float,
    n_monitors: int,
    block: bool,
) -> Result:
    return Result(
        method="mif",
        kind="trace",
        panel=True,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={
            "J": J,
            "M": M,
            "rw_sd": rw_sd,
            "thresh": thresh,
            "n_monitors": n_monitors,
            "block": block,
        },
        payload=_trace_payload(shared_traces, unit_traces, logLiks),
    )


def build_panel_train_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    shared_traces: xr.DataArray,
    unit_traces: xr.DataArray,
    logLiks: xr.DataArray,
    optimizer: Any,
    J: int,
    M: int,
    eta: Any,
    alpha: float,
    alpha_cooling: float,
) -> Result:
    return Result(
        method="train",
        kind="trace",
        panel=True,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={
            "optimizer": optimizer,
            "J": J,
            "M": M,
            "eta": eta,
            "alpha": alpha,
            "alpha_cooling": alpha_cooling,
        },
        payload=_trace_payload(shared_traces, unit_traces, logLiks),
    )


def build_panel_dpop_train_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    shared_traces: xr.DataArray,
    unit_traces: xr.DataArray,
    logLiks: xr.DataArray,
    optimizer: Any,
    J: int,
    M: int,
    eta: Any,
    alpha: float,
    alpha_cooling: float,
    process_weight_state: str | None,
    decay: float,
) -> Result:
    return Result(
        method="dpop_train",
        kind="trace",
        panel=True,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={
            "optimizer": optimizer,
            "J": J,
            "M": M,
            "eta": eta,
            "alpha": alpha,
            "alpha_cooling": alpha_cooling,
            "process_weight_state": process_weight_state,
            "decay": decay,
        },
        payload=_trace_payload(shared_traces, unit_traces, logLiks),
    )

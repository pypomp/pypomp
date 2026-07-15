"""Single-unit (:class:`~pypomp.core.pomp.Pomp`) result builders.

All behaviour lives on :class:`~pypomp.core.results.result.Result`.
Construct them through the ``build_*`` helpers, which assemble the
:class:`xarray.Dataset` payload and ``config`` mapping.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import xarray as xr

from .result import Result


def _dataset(**data_vars: xr.DataArray | None) -> xr.Dataset:
    return xr.Dataset({k: v for k, v in data_vars.items() if v is not None})


def build_pfilter_result(
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
        panel=False,
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


def build_mif_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    traces: xr.DataArray,
    J: int,
    M: int,
    rw_sd: Any,
    thresh: float,
    n_monitors: int,
) -> Result:
    return Result(
        method="mif",
        kind="trace",
        panel=False,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={
            "J": J,
            "M": M,
            "rw_sd": rw_sd,
            "thresh": thresh,
            "n_monitors": n_monitors,
        },
        payload=_dataset(traces=traces),
    )


def build_train_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    traces: xr.DataArray,
    optimizer: Any,
    J: int,
    M: int,
    eta: Any,
    alpha: float,
    thresh: float,
    alpha_cooling: float,
) -> Result:
    return Result(
        method="train",
        kind="trace",
        panel=False,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={
            "optimizer": optimizer,
            "J": J,
            "M": M,
            "eta": eta,
            "alpha": alpha,
            "thresh": thresh,
            "alpha_cooling": alpha_cooling,
        },
        payload=_dataset(traces=traces),
    )


def build_pmcmc_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    traces: xr.DataArray,
    Nmcmc: int,
    J: int,
    accepts: np.ndarray,
) -> Result:
    return Result(
        method="pmcmc",
        kind="trace",
        panel=False,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={"Nmcmc": Nmcmc, "J": J},
        payload=_dataset(
            traces=traces,
            accepts=xr.DataArray(
                np.asarray(accepts, dtype=np.int32), dims=["theta_idx"]
            ),
        ),
    )


def build_abc_result(
    *,
    key: jax.Array,
    execution_time: float | None,
    theta: Any,
    traces: xr.DataArray,
    Nabc: int,
    epsilon: float,
    accepts: np.ndarray,
) -> Result:
    return Result(
        method="abc",
        kind="trace",
        panel=False,
        execution_time=execution_time,
        key=key,
        theta=theta,
        config={"Nabc": Nabc, "epsilon": float(epsilon)},
        payload=_dataset(
            traces=traces,
            accepts=xr.DataArray(
                np.asarray(accepts, dtype=np.int32), dims=["theta_idx"]
            ),
        ),
    )

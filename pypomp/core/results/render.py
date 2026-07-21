"""Rendering for :class:`~pypomp.core.results.result.Result`.

Every result — regardless of method — is a single :class:`Result`. The
tidy-DataFrame / summary logic that used to be split across two mixins and
several per-class overrides lives here as free functions that dispatch on
``result.kind`` (``"table"`` vs ``"trace"``), ``result.panel``, and — for the
MCMC family — ``result.method``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from ...maths import logmeanexp, logmeanexp_se
from .result import _unalias

if TYPE_CHECKING:
    from .result import Result


def _var(result: "Result", name: str) -> xr.DataArray | None:
    """Return a payload variable with public dim names, or ``None`` if absent."""
    payload = result.payload
    if name not in payload.data_vars:
        return None
    return _unalias(payload[name])


_MCMC_METHODS = {"pmcmc", "abc"}

# config-key -> human label, for print_summary.
_SUMMARY_LABELS = {
    "optimizer": "Optimizer",
    "J": "Number of particles (J)",
    "reps": "Number of replicates",
    "M": "Number of iterations (M)",
    "eta": "Learning rate (eta)",
    "alpha": "Discount factor (alpha)",
    "thresh": "Resampling threshold",
    "alpha_cooling": "Cooling factor for alpha",
    "n_monitors": "Number of monitors",
    "block": "Block",
    "process_weight_state": "Process weight state",
    "decay": "Decay",
}


# ======================================================================
# Public dispatch entry points (called by Result.<method>).
# ======================================================================
def to_dataframe(result: "Result", ignore_nan: bool = False) -> pd.DataFrame:
    if result.kind == "table":
        return (
            _panel_table_to_df(result, ignore_nan)
            if result.panel
            else _pomp_table_to_df(result, ignore_nan)
        )
    if result.method in _MCMC_METHODS:
        return _mcmc_to_df(result, ignore_nan)
    return _panel_trace_to_df(result) if result.panel else _pomp_trace_to_df(result)


def traces(result: "Result") -> pd.DataFrame:
    if result.kind == "table":
        return (
            _panel_table_traces(result) if result.panel else _pomp_table_traces(result)
        )
    if result.method in _MCMC_METHODS:
        return _mcmc_traces(result)
    return _panel_trace_traces(result) if result.panel else _pomp_trace_traces(result)


def CLL(result: "Result", average: bool = False) -> pd.DataFrame:
    cll_da = result.payload["CLL"] if "CLL" in result.payload else None
    if cll_da is None or cll_da.size == 0:
        return pd.DataFrame()
    if not average:
        return cll_da.to_dataframe(name="CLL").reset_index()
    try:
        axis = cll_da.dims.index("rep")
    except ValueError:
        axis = 1
    avg = logmeanexp(np.asarray(cll_da.values), axis=axis)
    dims = [d for d in cll_da.dims if d != "rep"]
    coords = {d: cll_da.coords[d].values for d in dims}
    return (
        xr.DataArray(avg, dims=dims, coords=coords)
        .to_dataframe(name="CLL")
        .reset_index()
    )


def ESS(result: "Result", average: bool = False) -> pd.DataFrame:
    ess_da = result.payload["ESS"] if "ESS" in result.payload else None
    if ess_da is None or ess_da.size == 0:
        return pd.DataFrame()
    ess = ess_da.mean(dim="rep") if average else ess_da
    return ess.to_dataframe(name="ESS").reset_index()


def print_summary(result: "Result", n: int = 5) -> None:
    if result.method in _MCMC_METHODS:
        _print_mcmc_summary(result, n)
        return

    print(f"Method: {result.method}")
    print(f"Number of parameter sets: {_theta_count(result.theta)}")
    for key, val in result.config.items():
        if key == "rw_sd":
            continue
        label = _SUMMARY_LABELS.get(key)
        if label is None:
            continue
        print(f"{label}: {val}")

    rw_sd = result.config.get("rw_sd")
    if rw_sd is not None:
        ctype = getattr(rw_sd, "cooling_type", "none")
        if ctype == "geometric":
            print(f"Cooling fraction (a): {rw_sd.a}")
        elif ctype == "hyperbolic":
            print(f"Cooling rate (s): {rw_sd.s}")
        elif ctype == "cosine":
            print(f"Cosine min cooling (c): {rw_sd.c}")
            print(f"Cosine duration (M): {rw_sd.M}")
        elif ctype == "custom":
            fn = rw_sd.cooling_fn
            print(f"Cooling function: {getattr(fn, '__name__', str(fn))}")

    print(f"Execution time: {result.execution_time} seconds")
    df = to_dataframe(result)
    if not df.empty:
        print(f"\nTop {n} Results:")
        sort_col = "shared logLik" if "shared logLik" in df.columns else "logLik"
        print(df.sort_values(sort_col, ascending=False).head(n).to_string())


# ======================================================================
# pfilter ("table") — single-unit.
# ======================================================================
def _pomp_table_to_df(result: "Result", ignore_nan: bool) -> pd.DataFrame:
    theta = result.theta
    logLiks = result.payload["logLiks"] if "logLiks" in result.payload else None
    if not theta or logLiks is None or logLiks.size == 0:
        return pd.DataFrame()
    arr = np.asarray(logLiks.values)
    logLik = np.atleast_1d(logmeanexp(arr, axis=-1, ignore_nan=ignore_nan))
    se = (
        logmeanexp_se(arr, axis=-1, ignore_nan=ignore_nan)
        if arr.shape[-1] > 1
        else np.full_like(logLik, np.nan)
    )
    se = np.atleast_1d(se)
    theta_df = pd.DataFrame(theta.params(as_list=True))
    df = pd.DataFrame(
        {"theta_idx": np.arange(len(theta_df)), "logLik": logLik, "se": se}
    )
    return pd.concat([df, theta_df], axis=1)


def _pomp_table_traces(result: "Result") -> pd.DataFrame:
    df = _pomp_table_to_df(result, ignore_nan=False)
    if df.empty:
        return df
    df.insert(1, "iteration", 0)
    df.insert(2, "method", result.method)
    cols = ["theta_idx", "iteration", "method", "logLik", "se"]
    other_cols = [c for c in df.columns if c not in cols]
    return df[cols + other_cols]


# ======================================================================
# pfilter ("table") — panel.
# ======================================================================
def _attach_panel_params(theta, df_s, df_u):
    """Join theta values onto the shared/unit DataFrames."""
    if theta is None or theta.num_replicates() == 0:
        return df_s, df_u

    shared_names = theta.get_shared_param_names()
    if shared_names and "shared" in theta._data:
        s_vals = theta._data["shared"].sel(parameter=shared_names).values
        p_s = pd.DataFrame(s_vals, columns=shared_names)
        if df_s is not None:
            df_s = df_s.join(p_s, on="theta_idx")
        df_u = df_u.join(p_s, on="theta_idx")

    specific_names = theta.get_unit_param_names()
    if specific_names and "unit_specific" in theta._data:
        p_u = (
            theta._data["unit_specific"]
            .sel(parameter=specific_names)
            .to_dataset(dim="parameter")
            .to_dataframe()
            .reset_index()
        )
        df_u = df_u.merge(p_u, on=["theta_idx", "unit"], how="left")

    return df_s, df_u


def _panel_table_to_df(result: "Result", ignore_nan: bool) -> pd.DataFrame:
    logLiks = result.payload["logLiks"]
    ll = logmeanexp(logLiks.values, axis=-1, ignore_nan=ignore_nan)
    unit_names = logLiks.coords["unit"].values
    se_unit = (
        logmeanexp_se(logLiks.values, axis=-1, ignore_nan=ignore_nan)
        if logLiks.shape[-1] > 1
        else np.full_like(ll, np.nan)
    )
    se_shared = np.sqrt(np.sum(se_unit**2, axis=1))

    df_ll = (
        pd.DataFrame(ll, columns=unit_names)
        .assign(
            theta_idx=lambda x: range(len(x)),
            **{"shared logLik": lambda x: x.loc[:, unit_names].sum(axis=1)},
        )
        .melt(
            id_vars=["theta_idx", "shared logLik"],
            var_name="unit",
            value_name="unit logLik",
        )
    )
    df_se = (
        pd.DataFrame(se_unit, columns=logLiks.coords["unit"].values)
        .assign(
            theta_idx=lambda x: range(len(x)),
            **{"shared logLik se": se_shared},
        )
        .melt(
            id_vars=["theta_idx", "shared logLik se"],
            var_name="unit",
            value_name="unit logLik se",
        )
    )
    df = pd.merge(df_ll, df_se, on=["theta_idx", "unit"])
    cols = [
        "theta_idx",
        "shared logLik",
        "shared logLik se",
        "unit",
        "unit logLik",
        "unit logLik se",
    ]
    df = df[cols]

    _, df = _attach_panel_params(result.theta, None, df)
    return df


def _panel_table_traces(result: "Result") -> pd.DataFrame:
    logLiks = result.payload["logLiks"]
    ll = logmeanexp(logLiks.values, axis=-1)
    se_unit = (
        logmeanexp_se(logLiks.values, axis=-1)
        if logLiks.shape[-1] > 1
        else np.full_like(ll, np.nan)
    )
    se_shared = np.sqrt(np.sum(se_unit**2, axis=1))

    reps = np.arange(len(ll))
    df_s = pd.DataFrame(
        {
            "theta_idx": reps,
            "unit": "shared",
            "logLik": ll.sum(axis=1),
            "se": se_shared,
        }
    )
    df_u = (
        pd.DataFrame(ll, columns=logLiks.coords["unit"].values, index=reps)
        .melt(ignore_index=False, var_name="unit", value_name="logLik")
        .reset_index()
        .rename(columns={"index": "theta_idx"})
    )
    df_se_u = (
        pd.DataFrame(se_unit, columns=logLiks.coords["unit"].values, index=reps)
        .melt(ignore_index=False, var_name="unit", value_name="se")
        .reset_index()
        .rename(columns={"index": "theta_idx"})
    )
    df_u = pd.merge(df_u, df_se_u, on=["theta_idx", "unit"], how="left")

    df_s, df_u = _attach_panel_params(result.theta, df_s, df_u)
    dfs_to_concat = [df for df in [df_s, df_u] if not df.empty]
    if not dfs_to_concat:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df = pd.concat(dfs_to_concat, ignore_index=True)
    df = df.assign(method="pfilter", iteration=0)
    cols = ["theta_idx", "unit", "iteration", "method", "logLik", "se"]
    other_cols = [c for c in df.columns if c not in cols]
    return df.loc[:, cols + other_cols].copy()


# ======================================================================
# Estimation ("trace") — single-unit (mif / train).
# ======================================================================
def _pomp_trace_to_df(result: "Result") -> pd.DataFrame:
    traces_da = result.payload["traces"] if "traces" in result.payload else None
    if traces_da is None or not traces_da.sizes:
        return pd.DataFrame()
    df = (
        traces_da.isel(iteration=-1)
        .to_dataset(dim="variable")
        .to_dataframe()
        .reset_index()
    )
    param_names = result.theta.get_param_names() if result.theta is not None else []
    df = df[["theta_idx", "logLik"] + param_names]
    df.insert(2, "se", np.nan)
    return df


def _pomp_trace_traces(result: "Result") -> pd.DataFrame:
    traces_da = result.payload["traces"] if "traces" in result.payload else None
    if traces_da is None or traces_da.size == 0:
        return pd.DataFrame()
    df = (
        traces_da.to_dataset(dim="variable")
        .to_dataframe()
        .reset_index()
        .assign(method=result.method, se=np.nan)
    )
    cols = ["theta_idx", "iteration", "method", "logLik", "se"]
    other_cols = [c for c in df.columns if c not in cols]
    return df[cols + other_cols]


# ======================================================================
# Estimation ("trace") — panel (mif / train / dpop_train).
# ======================================================================
def _panel_trace_to_df(result: "Result") -> pd.DataFrame:
    shared_traces = _var(result, "shared_traces")
    unit_traces = _var(result, "unit_traces")
    if shared_traces is None or unit_traces is None or shared_traces.size == 0:
        return pd.DataFrame()
    s_df = (
        shared_traces.isel(iteration=-1)
        .to_dataset(dim="variable")
        .to_dataframe()
        .rename(columns={"logLik": "shared logLik"})
    )
    u_df = (
        unit_traces.isel(iteration=-1)
        .to_dataset(dim="variable")
        .to_dataframe()
        .rename(columns={"unitLogLik": "unit logLik"})
    )
    if "iteration" in s_df.columns:
        s_df = s_df.drop(columns=["iteration"])

    u_df = u_df.join(s_df, on="theta_idx").reset_index()

    u_df["shared logLik se"] = np.nan
    u_df["unit logLik se"] = np.nan
    cols = [
        "theta_idx",
        "iteration",
        "shared logLik",
        "shared logLik se",
        "unit",
        "unit logLik",
        "unit logLik se",
    ]
    return u_df[cols + [c for c in u_df.columns if c not in cols]]


def _panel_trace_traces(result: "Result") -> pd.DataFrame:
    shared_traces = _var(result, "shared_traces")
    unit_traces = _var(result, "unit_traces")
    if shared_traces is None or unit_traces is None or shared_traces.size == 0:
        return pd.DataFrame()
    df_s = (
        shared_traces.to_dataset(dim="variable")
        .to_dataframe()
        .reset_index()
        .assign(unit="shared")
    )
    df_u = (
        unit_traces.to_dataset(dim="variable")
        .to_dataframe()
        .reset_index()
        .rename(columns={"unitLogLik": "logLik"})
    )

    shared_params = [
        c for c in df_s.columns if c not in {"theta_idx", "iteration", "logLik", "unit"}
    ]
    if shared_params:
        df_u = df_u.merge(
            df_s[["theta_idx", "iteration"] + shared_params],
            on=["theta_idx", "iteration"],
            how="left",
        )
    dfs_to_concat = [df for df in [df_s, df_u] if not df.empty]
    if not dfs_to_concat:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df = pd.concat(dfs_to_concat, ignore_index=True)

    df = df.assign(method=result.method, se=np.nan)

    cols = ["theta_idx", "unit", "iteration", "method", "logLik", "se"]
    other_cols = [c for c in df.columns if c not in cols]
    return df[cols + other_cols]


# ======================================================================
# MCMC family ("trace") — pmcmc / abc.
# ======================================================================
def _mcmc_to_df(result: "Result", ignore_nan: bool) -> pd.DataFrame:
    traces_da = result.payload["traces"] if "traces" in result.payload else None
    if traces_da is None or traces_da.size == 0:
        return pd.DataFrame()
    df = traces_da.to_dataset(dim="variable").to_dataframe().reset_index()
    var_order = list(traces_da.coords["variable"].values)
    cols = ["theta_idx", "iteration"] + [c for c in var_order if c in df.columns]
    df = df[cols]
    if ignore_nan:
        df = df.dropna()
    return df


def _mcmc_traces(result: "Result") -> pd.DataFrame:
    df = _mcmc_to_df(result, ignore_nan=False)
    if df.empty:
        return df
    df.insert(2, "method", result.method)
    if result.method == "abc":
        df.insert(3, "logLik", np.nan)
    df.insert(4, "se", np.nan)
    return df


def _print_mcmc_summary(result: "Result", n: int) -> None:
    print(f"Method: {result.method}")
    print(f"Number of chains: {result.n_chains}")
    if result.method == "pmcmc":
        print(f"Number of particles (J): {result.config.get('J')}")
        print(f"MCMC iterations (M): {result.config.get('M')}")
    else:
        print(f"ABC iterations (M): {result.config.get('M')}")
        print(f"Tolerance (epsilon): {result.config.get('epsilon')}")

    accepts = (
        np.asarray(result.payload["accepts"].values)
        if "accepts" in result.payload
        else np.zeros(0)
    )
    if accepts.size > 0:
        rates = result.acceptance_rate
        for chain_idx in range(int(accepts.size)):
            print(
                f"  chain {chain_idx}: accepts={int(accepts[chain_idx])}, "
                f"rate={float(rates[chain_idx]):.3f}"
            )
    print(f"Execution time: {result.execution_time} seconds")

    traces_da = result.payload["traces"] if "traces" in result.payload else None
    if traces_da is not None and traces_da.size > 0:
        var = "logLik" if result.method == "pmcmc" else "distance"
        if var in list(traces_da.coords["variable"].values):
            last = traces_da.isel(iteration=-1).sel(variable=var).values
            print(f"\nFinal {var} per chain: {np.asarray(last)}")


def _theta_count(theta) -> int:
    if isinstance(theta, list):
        return len(theta)
    return theta.num_replicates() if theta else 0

from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar, Type, cast
import pandas as pd
import xarray as xr
import numpy as np
import jax
import warnings

T = TypeVar("T", bound="BaseResult")


def _merge_results(
    cls: Type[T],
    results: Sequence[T],
    scalar_fields: list[str],
    array_fields: list[str],
) -> T:
    """Helper to merge multiple result objects into one."""
    if not results:
        raise ValueError(f"At least one {cls.__name__} object must be provided.")
    first = results[0]

    for result in results:
        if not isinstance(result, cls):
            raise TypeError(f"All merged objects must be of type {cls.__name__}.")
        for field_name in scalar_fields:
            if getattr(result, field_name) != getattr(first, field_name):
                raise ValueError(
                    f"All {cls.__name__} objects must have the same {field_name}."
                )

    # Merge theta
    first_theta = getattr(first, "theta", None)
    if first_theta is not None:
        if isinstance(first_theta, list):
            merged_theta: Any = []
            for r in results:
                merged_theta.extend(getattr(r, "theta", []))
        elif hasattr(type(first_theta), "merge"):
            merged_theta = type(first_theta).merge(
                *[
                    getattr(r, "theta", None)
                    for r in results
                    if getattr(r, "theta", None) is not None
                ]
            )
        else:
            merged_theta = first_theta
    else:
        merged_theta = None

    # Merge DataArrays
    merged_arrays = {}
    for name in array_fields:
        arrays = [
            getattr(r, name)
            for r in results
            if getattr(r, name) is not None and getattr(r, name).size > 0
        ]
        merged_arrays[name] = xr.concat(arrays, dim="theta_idx") if arrays else None

    # Max execution time
    times = [r.execution_time for r in results if r.execution_time is not None]
    max_time = max(times) if times else None

    kwargs = {
        f.name: getattr(first, f.name)
        for f in fields(cls)
        if f.name
        not in (scalar_fields + array_fields + ["theta", "execution_time", "timestamp"])
    }
    kwargs.update({f: getattr(first, f) for f in scalar_fields})
    kwargs.update(merged_arrays)
    kwargs["execution_time"] = max_time
    if "theta" in [f.name for f in fields(cls)]:
        kwargs["theta"] = merged_theta

    return cls(**kwargs)


@dataclass(eq=False)
class BaseResult(ABC):
    """Base class for all result types."""

    method: str
    """The name of the method that produced this result (e.g., 'pfilter', 'mif')."""
    execution_time: float | None
    """Total execution time in seconds."""
    key: jax.Array
    """The JAX random key used for this execution."""
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    """The date and time when the result was created."""

    def __post_init__(self):
        """Post-initialization hook."""
        pass

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Structural equality for all result types using dataclass fields."""
        if type(self) is not type(other):
            return False

        for f in fields(self):
            v1, v2 = getattr(self, f.name), getattr(other, f.name)
            if f.name == "key":
                if not jax.numpy.array_equal(
                    jax.random.key_data(v1), jax.random.key_data(v2)
                ):
                    return False
            elif isinstance(v1, xr.DataArray) or isinstance(v2, xr.DataArray):
                if (v1 is None) != (v2 is None) or (
                    v1 is not None and not v1.equals(v2)
                ):
                    return False
            elif isinstance(v1, (np.ndarray, jax.Array)) or isinstance(
                v2, (np.ndarray, jax.Array)
            ):
                if not np.array_equal(v1, v2, equal_nan=True):
                    return False
            elif v1 != v2:
                return False
        return True

    def __getstate__(self):
        """Custom pickling: store JAX key as raw bits."""
        state = vars(self).copy()
        if self.key is not None:
            state["_key_data"] = jax.random.key_data(self.key)
        state.pop("key", None)
        return state

    def __setstate__(self, state):
        """Custom unpickling: reconstruct JAX key from raw bits."""
        vars(self).update(state)
        if "_key_data" in state:
            self.key = cast(jax.Array, jax.random.wrap_key_data(state["_key_data"]))
        vars(self).pop("_key_data", None)

    @abstractmethod
    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert result to DataFrame."""
        pass

    @staticmethod
    @abstractmethod
    def merge(*results: Any) -> "BaseResult":
        """Merge multiple result objects of the same type."""
        pass

    def CLL(self, average: bool = False) -> pd.DataFrame:
        """Return conditional log-likelihoods as a DataFrame."""
        return pd.DataFrame()

    def ESS(self, average: bool = False) -> pd.DataFrame:
        """Return Effective Sample Size as a DataFrame."""
        return pd.DataFrame()

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this result."""
        return pd.DataFrame()

    def print_summary(self, n: int = 5):
        """Print a summary of this result."""
        print(f"Method: {self.method}")
        for label, attr in self._summary_config:
            val = getattr(self, attr)
            if attr == "theta":
                print(
                    f"{label}: {len(val) if isinstance(val, list) else (val.num_replicates() if val else 0)}"
                )
            else:
                print(f"{label}: {val}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print(f"\nTop {n} Results:")
            sort_col = "shared logLik" if "shared logLik" in df.columns else "logLik"
            print(df.sort_values(sort_col, ascending=False).head(n).to_string())

    @property
    @abstractmethod
    def _summary_config(self) -> list[tuple[str, str]]:
        """List of (label, attribute_name) to print in summary."""
        pass


class PompEstimationTracesMixin:
    """Mixin for estimation results (MIF, Train) using traces_da pattern."""

    def to_dataframe(self: Any, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert result to DataFrame using traces_da."""
        traces_da = getattr(self, "traces_da", None)
        if traces_da is None or not hasattr(traces_da, "sizes") or not traces_da.sizes:
            return pd.DataFrame()
        df = (
            traces_da.isel(iteration=-1)
            .to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
        )
        param_names = list(self.theta[0].keys())
        df = df[["theta_idx", "logLik"] + param_names]
        df.insert(2, "se", np.nan)
        return df

    def traces(self: Any) -> pd.DataFrame:
        """Return traces DataFrame using traces_da."""
        traces_da = getattr(self, "traces_da", None)
        if traces_da is None or traces_da.size == 0:
            return pd.DataFrame()
        df = (
            traces_da.to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
            .assign(method=self.method)
        )
        cols = ["theta_idx", "iteration", "method", "logLik"]
        other_cols = [c for c in df.columns if c not in cols]
        return df.loc[:, cols + other_cols].copy()


class PanelPompEstimationTracesMixin:
    """Mixin for panel estimation results using shared_traces/unit_traces pattern."""

    def to_dataframe(self: Any, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert panel result to DataFrame."""
        shared_traces = getattr(self, "shared_traces", None)
        unit_traces = getattr(self, "unit_traces", None)
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
        cols = ["theta_idx", "iteration", "shared logLik", "unit", "unit logLik"]
        return u_df.loc[:, cols + [c for c in u_df.columns if c not in cols]].copy()

    def traces(self: Any) -> pd.DataFrame:
        """Return panel result formatted as traces (long format)."""
        shared_traces = getattr(self, "shared_traces", None)
        unit_traces = getattr(self, "unit_traces", None)
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
            c
            for c in df_s.columns
            if c not in {"theta_idx", "iteration", "logLik", "unit"}
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

        df = df.assign(method=self.method)

        cols = ["theta_idx", "unit", "iteration", "method", "logLik"]
        other_cols = [c for c in df.columns if c not in cols]
        return df.loc[:, cols + other_cols].copy()

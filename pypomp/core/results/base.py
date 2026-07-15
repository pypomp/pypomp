from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar, Type
import pandas as pd
import xarray as xr
import numpy as np
import jax
import warnings

T = TypeVar("T", bound="BaseResult")


def _merge_results(
    cls: Type[T],
    results: Sequence[T],
    scalar_fields: list[str] | None = None,
    array_fields: list[str] | None = None,
) -> T:
    """Helper to merge multiple result objects into one."""
    if not results:
        raise ValueError(f"At least one {cls.__name__} object must be provided.")
    first = results[0]

    for result in results:
        if not isinstance(result, cls):
            raise TypeError(f"All merged objects must be of type {cls.__name__}.")

    special_fields = {"theta", "execution_time", "timestamp", "key"}

    if array_fields is None:
        array_fields_set = set()
        for f in fields(cls):
            if f.name in special_fields:
                continue
            for r in results:
                val = getattr(r, f.name, None)
                if isinstance(val, xr.DataArray):
                    array_fields_set.add(f.name)
                    break
        array_fields_list = list(array_fields_set)
    else:
        array_fields_list = list(array_fields)

    if scalar_fields is None:
        scalar_fields_list = [
            f.name
            for f in fields(cls)
            if f.name not in special_fields and f.name not in array_fields_list
        ]
    else:
        scalar_fields_list = list(scalar_fields)

    for result in results:
        for field_name in scalar_fields_list:
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
    for name in array_fields_list:
        arrays = []
        for r in results:
            arr = getattr(r, name, None)
            if arr is not None and arr.size > 0:
                arrays.append(arr)
        if arrays:
            merged_arr = xr.concat(arrays, dim="theta_idx")
            if "theta_idx" in merged_arr.dims:
                merged_arr = merged_arr.assign_coords(
                    theta_idx=np.arange(merged_arr.sizes["theta_idx"])
                )
            merged_arrays[name] = merged_arr
        else:
            merged_arrays[name] = None

    # Max execution time
    times = [r.execution_time for r in results if r.execution_time is not None]
    max_time = max(times) if times else None

    kwargs = {
        f.name: getattr(first, f.name)
        for f in fields(cls)
        if f.name
        not in (
            scalar_fields_list
            + array_fields_list
            + ["theta", "execution_time", "timestamp"]
        )
    }
    kwargs.update({f: getattr(first, f) for f in scalar_fields_list})
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
            self.key = jax.random.wrap_key_data(state["_key_data"])
        vars(self).pop("_key_data", None)

    @abstractmethod
    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Parameters
        ----------
        ignore_nan : bool, optional
            Whether to ignore NaNs when computing log-likelihoods and standard errors.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            DataFrame representation of the results.
        """
        pass

    @classmethod
    def merge(cls: Type[T], *results: T) -> T:
        """Merge multiple result objects of the same type."""
        return _merge_results(cls, results)

    def CLL(self, average: bool = False) -> pd.DataFrame:
        """Return conditional log-likelihoods as a DataFrame.

        Parameters
        ----------
        average : bool, optional
            If ``True``, average the conditional log-likelihoods over the
            replicates scaled in likelihood space.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of conditional log-likelihoods.

            For single-unit models:
                - ``theta_idx``: Index of the parameter set.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``CLL``: Conditional log-likelihood value.

            For panel models:
                - ``theta_idx``: Index of the parameter set.
                - ``unit``: Unit name/identifier.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``CLL``: Conditional log-likelihood value.
        """
        cll_da = getattr(self, "CLL_da", None)
        if cll_da is None or cll_da.size == 0:
            return pd.DataFrame()
        if not average:
            return cll_da.to_dataframe(name="CLL").reset_index()
        try:
            axis = cll_da.dims.index("rep")
        except ValueError:
            axis = 1
        from ...maths import logmeanexp

        avg = logmeanexp(np.asarray(cll_da.values), axis=axis)
        dims = [d for d in cll_da.dims if d != "rep"]
        coords = {d: cll_da.coords[d].values for d in dims}
        return (
            xr.DataArray(avg, dims=dims, coords=coords)
            .to_dataframe(name="CLL")
            .reset_index()
        )

    def ESS(self, average: bool = False) -> pd.DataFrame:
        """Return Effective Sample Size as a DataFrame.

        Parameters
        ----------
        average : bool, optional
            If ``True``, average the ESS values over replicates.  Defaults to
            ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of ESS values.

            For single-unit models:
                - ``theta_idx``: Index of the parameter set.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``ESS``: Effective Sample Size value.

            For panel models:
                - ``theta_idx``: Index of the parameter set.
                - ``unit``: Unit name/identifier.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``ESS``: Effective Sample Size value.
        """
        ess_da = getattr(self, "ESS_da", None)
        if ess_da is None or ess_da.size == 0:
            return pd.DataFrame()
        ess = ess_da.mean(dim="rep") if average else ess_da
        return ess.to_dataframe(name="ESS").reset_index()

    def traces(self) -> pd.DataFrame:
        """Return parameter and likelihood trace history.

        Returns
        -------
        pd.DataFrame
            DataFrame of the traces.
        """
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

        rw_sd = getattr(self, "rw_sd", None)
        if rw_sd is not None:
            info = getattr(rw_sd, "_cooling_info", ("none",))
            ctype = info[0]
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
        """Convert the estimation result to a pandas DataFrame.

        Parameters
        ----------
        ignore_nan : bool, optional
            Whether to ignore NaNs when computing log-likelihoods and standard errors.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame representation of the estimation results. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``logLik``: The estimated log-likelihood at the final iteration.
            3. ``se``: The standard error of the log-likelihood (retains NaN for estimation runs).
            4. Parameter columns: One column per model parameter in their defined order.
        """
        traces_da = getattr(self, "traces_da", None)
        if traces_da is None or not hasattr(traces_da, "sizes") or not traces_da.sizes:
            return pd.DataFrame()
        df = (
            traces_da.isel(iteration=-1)
            .to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
        )
        param_names = self.theta.get_param_names() if self.theta is not None else []
        df = df[["theta_idx", "logLik"] + param_names]
        df.insert(2, "se", np.nan)
        return df

    def traces(self: Any) -> pd.DataFrame:
        """Return parameter and likelihood trace history across all iterations.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of iteration-by-iteration traces. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``iteration``: The iteration counter.
            3. ``method``: The name of the estimation method (e.g., ``'mif'`` or ``'train'``).
            4. ``logLik``: The estimated log-likelihood at each iteration.
            5. ``se``: The standard error of the log-likelihood (typically NaN).
            6. Parameter columns: One column per model parameter in their defined order.
        """
        traces_da = getattr(self, "traces_da", None)
        if traces_da is None or traces_da.size == 0:
            return pd.DataFrame()
        df = (
            traces_da.to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
            .assign(method=self.method, se=np.nan)
        )
        cols = ["theta_idx", "iteration", "method", "logLik", "se"]
        other_cols = [c for c in df.columns if c not in cols]
        return df[cols + other_cols]


class PanelPompEstimationTracesMixin:
    """Mixin for panel estimation results using shared_traces/unit_traces pattern."""

    def to_dataframe(self: Any, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert the panel estimation result to a pandas DataFrame.

        Parameters
        ----------
        ignore_nan : bool, optional
            Whether to ignore NaNs when computing log-likelihoods and standard errors.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame representation of the panel estimation results. The columns
            appear in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``iteration``: The iteration number.
            3. ``shared logLik``: The estimated shared log-likelihood across all units.
            4. ``shared logLik se``: The standard error of the shared log-likelihood (typically NaN).
            5. ``unit``: The unit name/identifier.
            6. ``unit logLik``: The unit-specific log-likelihood.
            7. ``unit logLik se``: The standard error of the unit-specific log-likelihood (typically NaN).
            8. Parameter columns: Shared and unit-specific parameters sharded by unit.
        """
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

    def traces(self: Any) -> pd.DataFrame:
        """Return panel result formatted as traces (long format).

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of iteration-by-iteration panel traces. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``unit``: The unit identifier (or ``'shared'`` for shared parameter rows).
            3. ``iteration``: The iteration counter.
            4. ``method``: The name of the estimation method.
            5. ``logLik``: The estimated log-likelihood (shared log-likelihood for shared rows, unit-specific log-likelihood for unit rows).
            6. ``se``: The standard error of the log-likelihood (typically NaN).
            7. Parameter columns: Shared and unit-specific parameters sharded by unit.
        """
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

        df = df.assign(method=self.method, se=np.nan)

        cols = ["theta_idx", "unit", "iteration", "method", "logLik", "se"]
        other_cols = [c for c in df.columns if c not in cols]
        return df[cols + other_cols]

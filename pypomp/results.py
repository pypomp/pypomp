from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
import xarray as xr
import numpy as np
import jax
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .RWSigma_class import RWSigma
else:
    RWSigma = object

from .util import logmeanexp, logmeanexp_se


@dataclass
class BaseResult(ABC):
    """Base class for all result types."""

    method: str
    execution_time: float | None
    key: jax.Array
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    def __post_init__(self):
        """Post-initialization hook."""
        pass

    @abstractmethod
    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert result to DataFrame."""
        pass

    @abstractmethod
    def print_summary(self):
        """Print a summary of this result."""
        pass


@dataclass
class PompBaseResult(BaseResult):
    """Base class for Pomp results."""

    theta: list[dict] = field(default_factory=list)


@dataclass
class PanelPompBaseResult(BaseResult):
    """Base class for PanelPomp results."""

    shared: list[pd.DataFrame] | None = None
    unit_specific: list[pd.DataFrame] | None = None


@dataclass
class PompPFilterResult(PompBaseResult):
    """Result from Pomp.pfilter() method."""

    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    J: int = 0
    reps: int = 1
    thresh: float = 0.0
    CLL: xr.DataArray | None = None
    ESS: xr.DataArray | None = None
    filter_mean: xr.DataArray | None = None
    prediction_mean: xr.DataArray | None = None

    def __post_init__(self):
        """Set method to pfilter."""
        self.method = "pfilter"

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert pfilter result to DataFrame."""
        rows = []
        param_names = list(self.theta[0].keys())
        for param_idx, (logLik_arr, theta_dict) in enumerate(
            zip(self.logLiks, self.theta)
        ):
            # Use underlying NumPy array if available to avoid copies
            arr = getattr(logLik_arr, "values", logLik_arr)
            logLik_arr_np = np.asarray(arr)
            logLik = float(logmeanexp(logLik_arr_np, ignore_nan=ignore_nan))
            se = (
                float(logmeanexp_se(logLik_arr_np, ignore_nan=ignore_nan))
                if len(logLik_arr_np) > 1
                else np.nan
            )
            row = {"logLik": logLik, "se": se}
            row.update({param: float(theta_dict[param]) for param in param_names})
            rows.append(row)
        return pd.DataFrame(rows)

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this pfilter result."""
        if not self.theta or not len(self.logLiks):
            return pd.DataFrame()

        param_names = list(self.theta[0].keys())
        n_reps = len(self.theta)

        # Vectorize loglik computation
        logliks = []
        for logLik_arr in self.logLiks:
            arr = getattr(logLik_arr, "values", logLik_arr)
            logLik_arr_np = np.asarray(arr)
            logliks.append(float(logmeanexp(logLik_arr_np)))

        # Vectorize replicate, iteration, and method lists
        replicate_list = np.arange(n_reps).tolist()
        iteration_list = [0] * n_reps  # Local iteration for pfilter
        method_list = ["pfilter"] * n_reps
        loglik_list = logliks

        # Vectorize parameter extraction
        param_columns = {}
        for p in param_names:
            param_columns[p] = [
                float(self.theta[rep_idx][p]) for rep_idx in range(n_reps)
            ]

        data = {
            "replicate": replicate_list,
            "iteration": iteration_list,
            "method": method_list,
            "loglik": loglik_list,
        }
        data.update(param_columns)

        return pd.DataFrame(data)

    def print_summary(self):
        """Print summary of pfilter result."""
        print(f"Method: {self.method}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of replicates: {self.reps}")
        print(f"Resampling threshold: {self.thresh}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print("\nResults:")
            print(df.to_string())


@dataclass
class PompMIFResult(PompBaseResult):
    """Result from Pomp.mif() method."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))  # type: ignore[assignment]
    J: int = 0
    M: int = 0
    rw_sd: RWSigma | None = None
    a: float = 0.0
    thresh: float = 0.0

    def __post_init__(self):
        """Set method to mif."""
        self.method = "mif"

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert mif result to DataFrame."""
        rows = []
        param_names = list(self.theta[0].keys())
        # traces_da is an xarray.DataArray with dims: (replicate, iteration, variable)
        traces_da: xr.DataArray = self.traces_da
        if traces_da is None or not hasattr(traces_da, "sizes"):
            return pd.DataFrame()
        n_reps = traces_da.sizes["replicate"]
        last_idx = traces_da.sizes["iteration"] - 1
        for rep in range(n_reps):
            last_row = traces_da.sel(replicate=rep, iteration=last_idx)
            logLik_val = float(last_row.sel(variable="logLik").values)
            row = {"logLik": logLik_val, "se": np.nan}
            for param in param_names:
                row[param] = float(last_row.sel(variable=param).values)
            rows.append(row)
        return pd.DataFrame(rows)

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this mif result."""
        traces_da: xr.DataArray = self.traces_da
        if traces_da is None or not hasattr(traces_da, "sizes"):
            return pd.DataFrame()

        n_rep = traces_da.sizes["replicate"]
        n_iter = traces_da.sizes["iteration"]
        variable_names = list(traces_da.coords["variable"].values)

        traces_array = traces_da.values
        loglik_idx = variable_names.index("logLik")

        param_names = list(self.theta[0].keys())
        param_indices = np.array([variable_names.index(p) for p in param_names])

        # Vectorize index creation using meshgrid
        rep_indices, iter_indices = np.meshgrid(
            np.arange(n_rep), np.arange(n_iter), indexing="ij"
        )
        replicate_list = rep_indices.flatten().tolist()
        iteration_list = iter_indices.flatten().tolist()
        method_list = ["mif"] * (n_rep * n_iter)

        # Vectorize loglik extraction
        loglik_list = traces_array[:, :, loglik_idx].flatten().astype(float).tolist()

        # Vectorize parameter extraction
        param_columns = {}
        for i, p in enumerate(param_names):
            param_columns[p] = (
                traces_array[:, :, param_indices[i]].flatten().astype(float).tolist()
            )

        data = {
            "replicate": replicate_list,
            "iteration": iteration_list,
            "method": method_list,
            "loglik": loglik_list,
        }
        data.update(param_columns)

        return pd.DataFrame(data)

    def print_summary(self):
        """Print summary of mif result."""
        print(f"Method: {self.method}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of iterations (M): {self.M}")
        print(f"Cooling fraction (a): {self.a}")
        print(f"Resampling threshold: {self.thresh}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print("\nResults:")
            print(df.to_string())


@dataclass
class PompTrainResult(PompBaseResult):
    """Result from Pomp.train() method."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))  # type: ignore[assignment]
    optimizer: str = "SGD"
    J: int = 0
    M: int = 0
    eta: float = 0.0
    alpha: float = 0.97
    thresh: int = 0
    ls: bool = False
    c: float = 0.1
    max_ls_itn: int = 10

    def __post_init__(self):
        """Set method to train."""
        self.method = "train"

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert train result to DataFrame."""
        rows = []
        param_names = list(self.theta[0].keys())
        # traces_da is an xarray.DataArray with dims: (replicate, iteration, variable)
        traces_da: xr.DataArray = self.traces_da
        if traces_da is None or not hasattr(traces_da, "sizes"):
            return pd.DataFrame()
        n_reps = traces_da.sizes["replicate"]
        last_idx = traces_da.sizes["iteration"] - 1
        for rep in range(n_reps):
            last_row = traces_da.sel(replicate=rep, iteration=last_idx)
            logLik_val = float(last_row.sel(variable="logLik").values)
            row = {"logLik": logLik_val, "se": np.nan}
            for param in param_names:
                row[param] = float(last_row.sel(variable=param).values)
            rows.append(row)
        return pd.DataFrame(rows)

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this train result."""
        traces_da: xr.DataArray = self.traces_da
        if traces_da is None or not hasattr(traces_da, "sizes"):
            return pd.DataFrame()

        n_rep = traces_da.sizes["replicate"]
        n_iter = traces_da.sizes["iteration"]
        variable_names = list(traces_da.coords["variable"].values)

        traces_array = traces_da.values
        loglik_idx = variable_names.index("logLik")

        param_names = list(self.theta[0].keys())
        param_indices = np.array([variable_names.index(p) for p in param_names])

        # Vectorize index creation using meshgrid
        rep_indices, iter_indices = np.meshgrid(
            np.arange(n_rep), np.arange(n_iter), indexing="ij"
        )
        replicate_list = rep_indices.flatten().tolist()
        iteration_list = iter_indices.flatten().tolist()
        method_list = ["train"] * (n_rep * n_iter)

        # Vectorize loglik extraction
        loglik_list = traces_array[:, :, loglik_idx].flatten().astype(float).tolist()

        # Vectorize parameter extraction
        param_columns = {}
        for i, p in enumerate(param_names):
            param_columns[p] = (
                traces_array[:, :, param_indices[i]].flatten().astype(float).tolist()
            )

        data = {
            "replicate": replicate_list,
            "iteration": iteration_list,
            "method": method_list,
            "loglik": loglik_list,
        }
        data.update(param_columns)

        return pd.DataFrame(data)

    def print_summary(self):
        """Print summary of train result."""
        print(f"Method: {self.method}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of iterations (M): {self.M}")
        print(f"Learning rate (eta): {self.eta}")
        print(f"Discount factor (alpha): {self.alpha}")
        print(f"Resampling threshold: {self.thresh}")
        print(f"Line search: {self.ls}")
        if self.ls:
            print(f"Armijo constant (c): {self.c}")
            print(f"Max line search iterations: {self.max_ls_itn}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print("\nResults:")
            print(df.to_string())


@dataclass
class PanelPompPFilterResult(PanelPompBaseResult):
    """Result from PanelPomp.pfilter() method."""

    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    J: int = 0
    reps: int = 1
    thresh: float = 0.0

    def __post_init__(self):
        """Set method to pfilter."""
        self.method = "pfilter"

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert panel pfilter result to DataFrame."""
        unit_names = list(self.logLiks.coords["unit"].values)
        n_reps = self.logLiks.sizes["theta"]

        logliks_array = self.logLiks.values

        unit_logliks = np.array(
            [
                [
                    logmeanexp(logliks_array[rep, unit_idx, :], ignore_nan=ignore_nan)
                    for unit_idx in range(len(unit_names))
                ]
                for rep in range(n_reps)
            ]
        )

        shared_logliks = np.sum(unit_logliks, axis=1)

        rep_indices = np.repeat(np.arange(n_reps), len(unit_names))
        unit_indices = np.tile(np.arange(len(unit_names)), n_reps)

        data = {
            "replicate": rep_indices,
            "unit": [unit_names[i] for i in unit_indices],
            "shared logLik": shared_logliks[rep_indices],
            "unit logLik": unit_logliks[rep_indices, unit_indices],
        }

        if self.shared and len(self.shared) > 0:
            shared_param_data = {}
            for rep in range(min(n_reps, len(self.shared))):
                shared_df = self.shared[rep]
                if hasattr(shared_df, "values") and shared_df.shape[1] >= 1:
                    shared_vals = shared_df.iloc[:, 0].values
                    shared_names = (
                        list(shared_df.columns) if hasattr(shared_df, "columns") else []
                    )
                    for i, param_name in enumerate(shared_names):
                        if param_name not in shared_param_data:
                            shared_param_data[param_name] = np.full(n_reps, np.nan)
                        shared_param_data[param_name][rep] = (
                            shared_vals[i] if i < len(shared_vals) else np.nan
                        )

            for param_name, values in shared_param_data.items():
                data[param_name] = values[rep_indices]

        if self.unit_specific and len(self.unit_specific) > 0:
            unit_param_data = {}
            for rep in range(min(n_reps, len(self.unit_specific))):
                unit_df = self.unit_specific[rep]
                if hasattr(unit_df, "columns"):
                    unit_param_names = (
                        list(unit_df.index) if hasattr(unit_df, "index") else []
                    )
                    for param_name in unit_param_names:
                        if param_name not in unit_param_data:
                            unit_param_data[param_name] = np.full(
                                (n_reps, len(unit_names)), np.nan
                            )
                        for unit_idx, unit in enumerate(unit_names):
                            if unit in unit_df.columns:
                                unit_param_data[param_name][rep, unit_idx] = (
                                    unit_df.loc[param_name, unit]
                                )

            for param_name, values in unit_param_data.items():
                data[param_name] = values[rep_indices, unit_indices]

        return pd.DataFrame(data)

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this panel pfilter result."""
        unit_names = list(self.logLiks.coords["unit"].values)
        shared_list = self.shared
        unit_list = self.unit_specific

        n_theta = self.logLiks.sizes["theta"]
        n_units = len(unit_names)

        if n_theta == 0:
            return pd.DataFrame()

        logliks_array = self.logLiks.values
        # Vectorize unit_avgs calculation
        unit_avgs = np.array(
            [
                [logmeanexp(logliks_array[rep_idx, u_i, :]) for u_i in range(n_units)]
                for rep_idx in range(n_theta)
            ]
        )
        shared_totals = np.sum(unit_avgs, axis=1)

        # Get param names
        shared_param_names = []
        unit_param_names = []
        if shared_list:
            shared_df = shared_list[0]
            if hasattr(shared_df, "index"):
                shared_param_names = list(shared_df.index)
        if unit_list:
            unit_df = unit_list[0]
            if hasattr(unit_df, "index"):
                unit_param_names = list(unit_df.index)

        all_param_names = shared_param_names + unit_param_names

        shared_param_data = {}
        if isinstance(shared_list, list):
            for rep_idx in range(min(n_theta, len(shared_list))):
                df = shared_list[rep_idx]
                if hasattr(df, "index") and df.shape[1] >= 1:
                    for name in df.index:
                        if name not in shared_param_data:
                            shared_param_data[name] = np.full(n_theta, np.nan)
                        shared_param_data[name][rep_idx] = float(
                            df.loc[name, df.columns[0]]
                        )

        unit_param_data = {}
        if isinstance(unit_list, list):
            for rep_idx in range(min(n_theta, len(unit_list))):
                df = unit_list[rep_idx]
                if hasattr(df, "columns"):
                    for name in df.index:
                        if name not in unit_param_data:
                            unit_param_data[name] = np.full((n_theta, n_units), np.nan)
                        for unit_idx, unit in enumerate(unit_names):
                            if str(unit) in df.columns:
                                unit_param_data[name][rep_idx, unit_idx] = float(
                                    df.loc[name, str(unit)]
                                )

        # Vectorize row creation
        # Create shared rows: one per replicate
        rep_indices_shared = np.arange(n_theta)
        # Create unit rows: n_units per replicate
        rep_indices_units = np.repeat(np.arange(n_theta), n_units)
        unit_indices = np.tile(np.arange(n_units), n_theta)

        # Build shared rows data
        shared_data = {
            "replicate": rep_indices_shared.tolist(),
            "unit": ["shared"] * n_theta,
            "iteration": [0] * n_theta,
            "method": ["pfilter"] * n_theta,
            "logLik": shared_totals.astype(float).tolist(),
        }
        for name in all_param_names:
            if name in shared_param_data:
                shared_data[name] = shared_param_data[name].astype(float).tolist()
            else:
                shared_data[name] = [float("nan")] * n_theta

        # Build unit rows data
        unit_data = {
            "replicate": rep_indices_units.tolist(),
            "unit": [str(unit_names[i]) for i in unit_indices],
            "iteration": [0] * (n_theta * n_units),
            "method": ["pfilter"] * (n_theta * n_units),
            "logLik": unit_avgs[rep_indices_units, unit_indices].astype(float).tolist(),
        }
        for name in all_param_names:
            if name in unit_param_data:
                unit_data[name] = (
                    unit_param_data[name][rep_indices_units, unit_indices]
                    .astype(float)
                    .tolist()
                )
            elif name in shared_param_data:
                unit_data[name] = (
                    shared_param_data[name][rep_indices_units].astype(float).tolist()
                )
            else:
                unit_data[name] = [float("nan")] * (n_theta * n_units)

        # Combine shared and unit rows, interleaving by replicate
        # Build interleaved structure directly using numpy for efficiency
        n_total = n_theta * (1 + n_units)  # 1 shared + n_units per replicate

        # Create interleaved structure: for each rep, [shared, unit1, unit2, ...]
        # Shared positions: 0, 1+n_units, 2*(1+n_units), ...
        shared_positions = np.arange(0, n_total, 1 + n_units)
        # Unit positions: everything else
        unit_positions = np.setdiff1d(np.arange(n_total), shared_positions)

        # Build combined data dict directly
        combined_data = {}

        # Handle "unit" column specially
        shared_units = np.array(["shared"] * n_theta, dtype=object)
        unit_units = np.array([str(unit_names[i]) for i in unit_indices], dtype=object)
        combined_units = np.empty(n_total, dtype=object)
        combined_units[shared_positions] = shared_units
        combined_units[unit_positions] = unit_units
        combined_data["unit"] = combined_units.tolist()

        # Handle other columns
        for col in ["replicate", "iteration", "method", "logLik"]:
            shared_vals = np.array(shared_data[col])
            unit_vals = np.array(unit_data[col])
            combined_vals = np.empty(n_total, dtype=shared_vals.dtype)
            combined_vals[shared_positions] = shared_vals
            combined_vals[unit_positions] = unit_vals
            combined_data[col] = combined_vals.tolist()

        # Handle parameter columns
        for name in all_param_names:
            if name in shared_param_data:
                shared_vals = shared_param_data[name]
            else:
                shared_vals = np.full(n_theta, np.nan)

            if name in unit_param_data:
                unit_vals = unit_param_data[name][rep_indices_units, unit_indices]
            elif name in shared_param_data:
                unit_vals = shared_param_data[name][rep_indices_units]
            else:
                unit_vals = np.full(n_theta * n_units, np.nan)

            combined_vals = np.empty(n_total, dtype=float)
            combined_vals[shared_positions] = shared_vals.astype(float)
            combined_vals[unit_positions] = unit_vals.astype(float)
            combined_data[name] = combined_vals.tolist()

        df = pd.DataFrame(combined_data)
        return df

    def print_summary(self):
        """Print summary of panel pfilter result."""
        print(f"Method: {self.method}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of replicates: {self.reps}")
        print(f"Resampling threshold: {self.thresh}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print("\nResults:")
            print(df.to_string())


@dataclass
class PanelPompMIFResult(PanelPompBaseResult):
    """Result from PanelPomp.mif() method."""

    shared_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    unit_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    J: int = 0
    M: int = 0
    rw_sd: RWSigma | None = None
    a: float = 0.0
    thresh: float = 0.0
    block: bool = True

    def __post_init__(self):
        """Set method to mif."""
        self.method = "mif"

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert panel mif result to DataFrame."""
        shared_da = self.shared_traces
        unit_da = self.unit_traces

        all_shared_vars = list(shared_da.coords["variable"].values)
        shared_names = all_shared_vars[1:] if len(all_shared_vars) > 1 else []

        all_unit_vars = list(unit_da.coords["variable"].values)
        unit_names = list(unit_da.coords["unit"].values)
        n_reps = shared_da.sizes["replicate"]

        shared_final_values = shared_da.isel(iteration=-1).values
        unit_final_values = unit_da.isel(iteration=-1).values

        shared_logliks = self.logLiks[:, 0].values
        unit_logliks = self.logLiks[:, 1:].values

        rep_indices = np.repeat(np.arange(n_reps), len(unit_names))
        unit_indices = np.tile(np.arange(len(unit_names)), n_reps)

        data = {
            "replicate": rep_indices,
            "unit": [unit_names[i] for i in unit_indices],
            "shared logLik": shared_logliks[rep_indices],
            "unit logLik": unit_logliks[rep_indices, unit_indices],
        }

        if shared_names:
            shared_param_values = shared_final_values[:, 1:]
            for i, param_name in enumerate(shared_names):
                data[param_name] = shared_param_values[rep_indices, i]

        unit_param_values = unit_final_values[:, 1:, :]
        for i, param_name in enumerate(all_unit_vars[1:]):
            data[param_name] = unit_param_values[rep_indices, i, unit_indices]

        return pd.DataFrame(data)

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this panel mif result."""
        shared_da = self.shared_traces
        unit_da = self.unit_traces
        unit_names = list(unit_da.coords["unit"].values)
        shared_vars = list(shared_da.coords["variable"].values)
        unit_vars = list(unit_da.coords["variable"].values)

        n_rep = shared_da.sizes["replicate"]
        n_iter = shared_da.sizes["iteration"]
        n_units = len(unit_names)

        if n_rep == 0 or n_iter == 0:
            return pd.DataFrame()

        shared_values = shared_da.values
        unit_values = unit_da.values

        shared_param_indices = {name: i for i, name in enumerate(shared_vars[1:])}
        unit_param_indices = {name: i for i, name in enumerate(unit_vars[1:])}

        shared_param_names = list(shared_param_indices.keys())
        unit_param_names = list(unit_param_indices.keys())
        all_param_names = shared_param_names + unit_param_names

        # Vectorize index creation
        rep_indices, iter_indices = np.meshgrid(
            np.arange(n_rep), np.arange(n_iter), indexing="ij"
        )
        rep_indices_flat = rep_indices.flatten()
        iter_indices_flat = iter_indices.flatten()

        # Extract shared logliks and params vectorized
        shared_logliks = shared_values[:, :, 0].flatten()
        shared_param_arrays = {}
        for name in shared_param_names:
            idx = shared_param_indices[name] + 1
            shared_param_arrays[name] = shared_values[:, :, idx].flatten()

        # Extract unit logliks and params vectorized
        unit_logliks = unit_values[:, :, 0, :].reshape(n_rep * n_iter, n_units)
        unit_param_arrays = {}
        for name in unit_param_names:
            idx = unit_param_indices[name] + 1
            unit_param_arrays[name] = unit_values[:, :, idx, :].reshape(
                n_rep * n_iter, n_units
            )

        # Build shared rows
        shared_data = {
            "replicate": rep_indices_flat.tolist(),
            "unit": ["shared"] * (n_rep * n_iter),
            "iteration": iter_indices_flat.tolist(),
            "method": ["mif"] * (n_rep * n_iter),
            "logLik": shared_logliks.astype(float).tolist(),
        }
        for name in all_param_names:
            if name in shared_param_arrays:
                shared_data[name] = shared_param_arrays[name].astype(float).tolist()
            else:
                shared_data[name] = [float("nan")] * (n_rep * n_iter)

        # Build unit rows
        # Repeat each (rep, iter) combination for each unit
        rep_indices_units = np.repeat(rep_indices_flat, n_units)
        iter_indices_units = np.repeat(iter_indices_flat, n_units)
        unit_indices_flat = np.tile(np.arange(n_units), n_rep * n_iter)

        unit_data = {
            "replicate": rep_indices_units.tolist(),
            "unit": [str(unit_names[i]) for i in unit_indices_flat],
            "iteration": iter_indices_units.tolist(),
            "method": ["mif"] * (n_rep * n_iter * n_units),
            "logLik": unit_logliks.flatten().astype(float).tolist(),
        }

        # Map flat indices back to (rep*iter, unit) for unit params
        flat_to_repiter = np.repeat(np.arange(n_rep * n_iter), n_units)

        for name in all_param_names:
            if name in unit_param_arrays:
                unit_data[name] = (
                    unit_param_arrays[name].flatten().astype(float).tolist()
                )
            elif name in shared_param_arrays:
                # Broadcast shared params to units
                unit_data[name] = (
                    shared_param_arrays[name][flat_to_repiter].astype(float).tolist()
                )
            else:
                unit_data[name] = [float("nan")] * (n_rep * n_iter * n_units)

        # Combine shared and unit rows, interleaving by (rep, iter)
        # Build interleaved structure directly using numpy for efficiency
        n_total = n_rep * n_iter * (1 + n_units)  # 1 shared + n_units per (rep, iter)

        # Create interleaved structure: for each (rep, iter), [shared, unit1, unit2, ...]
        # Shared positions: 0, 1+n_units, 2*(1+n_units), ...
        shared_positions = np.arange(0, n_total, 1 + n_units)
        # Unit positions: everything else
        unit_positions = np.setdiff1d(np.arange(n_total), shared_positions)

        # Build combined data dict directly
        combined_data = {}

        # Handle "unit" column specially
        shared_units = np.array(["shared"] * (n_rep * n_iter), dtype=object)
        unit_units = np.array(
            [str(unit_names[i]) for i in unit_indices_flat], dtype=object
        )
        combined_units = np.empty(n_total, dtype=object)
        combined_units[shared_positions] = shared_units
        combined_units[unit_positions] = unit_units
        combined_data["unit"] = combined_units.tolist()

        # Handle other columns
        for col in ["replicate", "iteration", "method", "logLik"]:
            shared_vals = np.array(shared_data[col])
            unit_vals = np.array(unit_data[col])
            combined_vals = np.empty(n_total, dtype=shared_vals.dtype)
            combined_vals[shared_positions] = shared_vals
            combined_vals[unit_positions] = unit_vals
            combined_data[col] = combined_vals.tolist()

        # Handle parameter columns
        for name in all_param_names:
            if name in shared_param_arrays:
                shared_vals = shared_param_arrays[name]
            else:
                shared_vals = np.full(n_rep * n_iter, np.nan)

            if name in unit_param_arrays:
                unit_vals = unit_param_arrays[name].flatten()
            elif name in shared_param_arrays:
                unit_vals = shared_param_arrays[name][flat_to_repiter]
            else:
                unit_vals = np.full(n_rep * n_iter * n_units, np.nan)

            combined_vals = np.empty(n_total, dtype=float)
            combined_vals[shared_positions] = shared_vals.astype(float)
            combined_vals[unit_positions] = unit_vals.astype(float)
            combined_data[name] = combined_vals.tolist()

        df = pd.DataFrame(combined_data)
        return df

    def print_summary(self):
        """Print summary of panel mif result."""
        print(f"Method: {self.method}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of iterations (M): {self.M}")
        print(f"Cooling fraction (a): {self.a}")
        print(f"Resampling threshold: {self.thresh}")
        print(f"Block: {self.block}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print("\nResults:")
            print(df.to_string())


class ResultsHistory:
    """Container class for managing result history."""

    def __init__(self):
        self._entries: list[BaseResult] = []

    def add(self, result: BaseResult):
        """Add a result entry."""
        self._entries.append(result)

    def __getitem__(self, index):
        """Get result by index."""
        return self._entries[index]

    def __len__(self):
        """Get number of entries."""
        return len(self._entries)

    def clear(self):
        """Clear all entries from the history."""
        self._entries.clear()

    def last(self) -> BaseResult:
        """Get last entry."""
        if not self._entries:
            raise ValueError("History is empty")
        return self._entries[-1]

    def get_best_run(self) -> BaseResult | None:
        """Find run with highest log-likelihood."""
        if not self._entries:
            return None
        return max(
            self._entries,
            key=lambda x: getattr(
                x, "total_log_lik", getattr(x, "best_log_lik", -float("inf"))
            ),
        )

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """Get results DataFrame for entry at index."""
        if not self._entries:
            return pd.DataFrame()
        result = self._entries[index]
        return result.to_dataframe(ignore_nan=ignore_nan)

    def time(self) -> pd.DataFrame:
        """Return execution times DataFrame."""
        rows = []
        for idx, res in enumerate(self._entries):
            method = res.method
            exec_time = res.execution_time
            rows.append({"method": method, "time": exec_time})
        df = pd.DataFrame(rows)
        df.index.name = "history_index"
        return df

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame from entire result history."""
        if not self._entries:
            return pd.DataFrame()

        all_dfs = []
        global_iters: dict[int, int] = {}  # replicate -> current global iteration

        for res in self._entries:
            # Check if the result class has a traces() method (not just a traces attribute)
            traces_method = getattr(type(res), "traces", None)
            if not callable(traces_method):
                continue
            df = res.traces()  # type: ignore[attr-defined]
            if df.empty:
                continue

            # Adjust iteration numbers to be global (vectorized)
            rep_indices = df["replicate"].values.astype(int)
            local_iters = df["iteration"].values.astype(int)

            # Initialize global_iters for any new replicates
            unique_reps = np.unique(rep_indices)
            for rep_idx in unique_reps:
                if rep_idx not in global_iters:
                    global_iters[rep_idx] = 0

            # Vectorized processing based on result type
            if isinstance(res, (PompMIFResult, PompTrainResult, PanelPompMIFResult)):
                # For mif/train: skip starting parameters beyond the first round
                keep_mask = np.ones(len(df), dtype=bool)
                new_iterations = np.zeros(len(df), dtype=int)

                for rep_idx in unique_reps:
                    rep_mask = rep_indices == rep_idx
                    local_iters_rep = local_iters[rep_mask]
                    rep_indices_in_df = np.where(rep_mask)[0]

                    # Skip starting parameters beyond first round
                    if global_iters[rep_idx] == 0:
                        # Keep all rows for first round
                        # Update iterations sequentially
                        for i, idx_in_df in enumerate(rep_indices_in_df):
                            new_iterations[idx_in_df] = global_iters[rep_idx]
                            global_iters[rep_idx] += 1
                    else:
                        # Skip rows with local_iter == 0 (starting params)
                        skip_mask = local_iters_rep == 0
                        keep_mask[rep_indices_in_df[skip_mask]] = False
                        # Update iterations for kept rows
                        kept_mask = ~skip_mask
                        for i, idx_in_df in enumerate(rep_indices_in_df[kept_mask]):
                            new_iterations[idx_in_df] = global_iters[rep_idx]
                            global_iters[rep_idx] += 1

                df = df[keep_mask].copy()
                df["iteration"] = new_iterations[keep_mask]

            elif isinstance(res, (PompPFilterResult, PanelPompPFilterResult)):
                # For pfilter, use the last global iteration (or 1 if first)
                new_iterations = np.zeros(len(df), dtype=int)
                for rep_idx in unique_reps:
                    rep_mask = rep_indices == rep_idx
                    last_iter = global_iters.get(rep_idx, 1) - 1
                    new_iter = last_iter if last_iter > 0 else 1
                    new_iterations[rep_mask] = new_iter

                df = df.copy()
                df["iteration"] = new_iterations

            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # Concatenate all DataFrames
        result_df = pd.concat(all_dfs, ignore_index=True)

        # Sort by iteration and replicate
        if "unit" in result_df.columns:
            # Panel results: sort by replicate, unit, iteration
            result_df = result_df.sort_values(
                ["replicate", "unit", "iteration"]
            ).reset_index(drop=True)
        else:
            # Regular results: sort by iteration, replicate
            result_df = result_df.sort_values(["iteration", "replicate"]).reset_index(
                drop=True
            )

        return result_df

    def print_summary(self):
        """Print summary of all entries."""
        if not self._entries:
            print("No results history.")
            return

        print("Results history:")
        print("----------------")
        for idx, entry in enumerate(self._entries, 1):
            print(f"Results entry {idx}:")
            entry.print_summary()
            print()

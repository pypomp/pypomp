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
        if not self.theta or self.logLiks.size == 0:
            return pd.DataFrame()

        arr = getattr(self.logLiks, "values", self.logLiks)
        logLik_arr_np = np.asarray(arr)

        logLik = np.apply_along_axis(
            logmeanexp, -1, logLik_arr_np, ignore_nan=ignore_nan
        )

        if logLik_arr_np.shape[-1] > 1:
            se = np.apply_along_axis(
                logmeanexp_se, -1, logLik_arr_np, ignore_nan=ignore_nan
            )
        else:
            se = np.full_like(logLik, np.nan, dtype=float)

        theta_df = pd.DataFrame(self.theta)

        df = pd.DataFrame(
            {
                "logLik": logLik.astype(float),
                "se": se.astype(float),
            }
        )

        return pd.concat(
            [df.reset_index(drop=True), theta_df.reset_index(drop=True)], axis=1
        )

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this pfilter result."""
        if not self.theta or not len(self.logLiks):
            return pd.DataFrame()

        arr = getattr(self.logLiks, "values", self.logLiks)
        logLik_arr_np = np.asarray(arr)
        logliks = np.apply_along_axis(logmeanexp, -1, logLik_arr_np)

        n_reps = len(self.theta)

        base_df = pd.DataFrame(
            {
                "replicate": np.arange(n_reps, dtype=int),
                "iteration": np.zeros(n_reps, dtype=int),
                "method": self.method,
                "logLik": logliks.astype(float),
            }
        )

        theta_df = pd.DataFrame(self.theta).reset_index(drop=True)

        return pd.concat([base_df.reset_index(drop=True), theta_df], axis=1)

    def print_summary(self):
        """Print summary of pfilter result."""
        print(f"Method: {self.method}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of replicates: {self.reps}")
        print(f"Resampling threshold: {self.thresh}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print("\nTop 5 Results:")
            df_sorted = df.sort_values("logLik", ascending=False).head(5)
            print(df_sorted.to_string())


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
        traces_da: xr.DataArray = self.traces_da
        if traces_da is None or not hasattr(traces_da, "sizes") or not traces_da.sizes:
            return pd.DataFrame()

        df = (
            traces_da.isel(iteration=-1)
            .to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
        )

        param_names = list(self.theta[0].keys())
        cols = ["logLik"] + param_names
        df = pd.DataFrame(df[cols])
        df.insert(1, "se", np.nan)

        return df

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this mif result."""
        if self.traces_da is None:
            return pd.DataFrame()

        return (
            self.traces_da.to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
            .assign(method="mif")
        )

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
            print("\nTop 5 Results:")
            df_sorted = df.sort_values("logLik", ascending=False).head(5)
            print(df_sorted.to_string())


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
        traces_da: xr.DataArray = self.traces_da
        if traces_da is None or not hasattr(traces_da, "sizes") or not traces_da.sizes:
            return pd.DataFrame()

        df = (
            traces_da.isel(iteration=-1)
            .to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
        )

        param_names = list(self.theta[0].keys())
        cols = ["logLik"] + param_names
        df = pd.DataFrame(df[cols])
        df.insert(1, "se", np.nan)

        return df

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this train result."""
        if self.traces_da is None:
            return pd.DataFrame()

        return (
            self.traces_da.to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
            .assign(method="train")
        )

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
            print("\nTop 5 Results:")
            df_sorted = df.sort_values("logLik", ascending=False).head(5)
            print(df_sorted.to_string())


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
        ll = np.apply_along_axis(
            logmeanexp, -1, self.logLiks.values, ignore_nan=ignore_nan
        )

        df = (
            pd.DataFrame(ll, columns=self.logLiks.coords["unit"].values)
            .assign(
                replicate=lambda x: range(len(x)),
                **{"shared logLik": lambda x: x.sum(axis=1)},
            )
            .melt(
                id_vars=["replicate", "shared logLik"],
                var_name="unit",
                value_name="unit logLik",
            )
        )

        if self.shared:
            s_params = pd.concat(self.shared, axis=1).T.reset_index(drop=True)
            df = df.join(s_params, on="replicate")

        if self.unit_specific:
            u_params = (
                pd.concat(self.unit_specific, keys=range(len(self.unit_specific)))
                .stack()
                .unstack(level=1)
                .reset_index()
            )
            col_names = list(u_params.columns)
            u_params.rename(
                columns={col_names[0]: "replicate", col_names[1]: "unit"}, inplace=True
            )
            df = df.merge(u_params, on=["replicate", "unit"], how="left")

        return df

    def traces(self) -> pd.DataFrame:
        """Return pfilter results formatted as traces (long format)."""
        ll = np.apply_along_axis(logmeanexp, -1, self.logLiks.values)
        reps = np.arange(len(ll))

        df_s = pd.DataFrame(
            {"replicate": reps, "unit": "shared", "logLik": ll.sum(axis=1)}
        )

        df_u = (
            pd.DataFrame(ll, columns=self.logLiks.coords["unit"].values, index=reps)
            .melt(ignore_index=False, var_name="unit", value_name="logLik")
            .reset_index()
            .rename(columns={"index": "replicate"})
        )

        if self.shared:
            p_s = pd.concat(self.shared, axis=1).T.set_axis(reps, axis=0)
            df_s = df_s.join(p_s, on="replicate")
            df_u = df_u.join(p_s, on="replicate")

        if self.unit_specific:
            p_u = pd.concat(self.unit_specific, keys=reps).stack().unstack(level=1)
            p_u.index.names = ["replicate", "unit"]
            df_u = df_u.join(p_u, on=["replicate", "unit"])

        return pd.concat([df_s, df_u], ignore_index=True).assign(
            method="pfilter", iteration=1
        )

    def print_summary(self):
        """Print summary of panel pfilter result."""
        print(f"Method: {self.method}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of replicates: {self.reps}")
        print(f"Resampling threshold: {self.thresh}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print("\nTop 5 Results:")
            df_sorted = df.sort_values("shared logLik", ascending=False).head(5)
            print(df_sorted.to_string())


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
        s_df = (
            self.shared_traces.isel(iteration=-1)
            .to_dataset(dim="variable")
            .to_dataframe()
            .rename(columns={"logLik": "shared logLik"})
        )

        u_df = (
            self.unit_traces.isel(iteration=-1)
            .to_dataset(dim="variable")
            .to_dataframe()
            .rename(columns={"unitLogLik": "unit logLik"})
        )

        # Avoid duplicate "iteration" column on join; keep the one from u_df
        if "iteration" in s_df.columns:
            s_df = s_df.drop(columns=["iteration"])

        return u_df.join(s_df, on="replicate").reset_index()

    def traces(self) -> pd.DataFrame:
        """Return panel mif results formatted as traces (long format)."""
        if self.shared_traces.size == 0:
            return pd.DataFrame()

        df_s = (
            self.shared_traces.to_dataset(dim="variable").to_dataframe().reset_index()
        )
        df_s["unit"] = "shared"

        df_u = self.unit_traces.to_dataset(dim="variable").to_dataframe().reset_index()
        df_u = df_u.rename(columns={"unitLogLik": "logLik"})

        meta_cols = {"replicate", "iteration", "logLik", "unit"}
        shared_params = [c for c in df_s.columns if c not in meta_cols]

        if shared_params:
            df_u = df_u.merge(
                df_s[["replicate", "iteration"] + shared_params],
                on=["replicate", "iteration"],
                how="left",
            )

        return pd.concat([df_s, df_u], ignore_index=True).assign(method="mif")

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
            print("\nTop 5 Results:")
            df_sorted = df.sort_values("shared logLik", ascending=False).head(5)
            print(df_sorted.to_string())


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

    def __iter__(self):
        """Iterate over entries."""
        return iter(self._entries)

    def clear(self):
        """Clear all entries from the history."""
        self._entries.clear()

    def last(self) -> BaseResult:
        """Get last entry."""
        if not self._entries:
            raise ValueError("History is empty")
        return self._entries[-1]

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
        """
        Return traces DataFrame from entire result history.

        Handles continuous iteration counting across chained runs
        (e.g., MIF -> MIF) and aligns checkpoints (PFilter).
        """
        if not self._entries:
            return pd.DataFrame()

        all_dfs = []
        global_iter_counters: dict[int, int] = {}

        for res in self._entries:
            if not hasattr(res, "traces"):
                continue
            df = res.traces()  # pyright: ignore[reportAttributeAccessIssue]
            if df.empty:
                continue

            is_estimation = res.method in ["mif", "train"]

            unique_reps = df["replicate"].unique()
            offsets_map = {r: global_iter_counters.get(r, 0) for r in unique_reps}

            row_offsets = df["replicate"].map(offsets_map)

            if is_estimation:
                mask = (df["iteration"] > 0) | (row_offsets == 0)
                df = df.loc[mask].copy()

                row_offsets = df["replicate"].map(offsets_map)

                df["iteration"] = df["iteration"] + row_offsets

                new_maxes = df.groupby("replicate")["iteration"].max()
                for r, mx in new_maxes.items():
                    global_iter_counters[r] = int(mx)

            else:
                # LOGIC: PFilter is a snapshot.
                # Plot it at the current "end" of the timeline.
                df = df.copy()
                df["iteration"] = row_offsets
                # We do NOT increment the global_iter_counters here

            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        result_df = pd.concat(all_dfs, ignore_index=True)

        sort_cols = ["replicate", "iteration"]
        if "unit" in result_df.columns:
            sort_cols.insert(1, "unit")

        result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

        canonical_first = ["replicate", "unit", "iteration", "method", "logLik"]
        existing_first = [c for c in canonical_first if c in result_df.columns]
        remaining = [c for c in result_df.columns if c not in existing_first]
        result_df = result_df[existing_first + remaining]

        assert isinstance(result_df, pd.DataFrame), (
            "result_df is not a DataFrame; something went wrong"
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

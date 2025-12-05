from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd
import xarray as xr
import numpy as np
import jax
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .RWSigma_class import RWSigma
    from .parameters import PanelParameters
else:
    RWSigma = object
    PanelParameters = object

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

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """
        Structural equality for all result types.

        Compares:
        - type
        - method string
        - execution_time
        - timestamp
        - JAX key contents (via key_data)
        """
        if not isinstance(other, type(self)):
            return False

        if self.method != other.method:
            return False

        if self.execution_time != other.execution_time:
            return False

        if self.timestamp != other.timestamp:
            return False

        # Compare JAX keys by underlying data
        if not jax.numpy.array_equal(
            jax.random.key_data(self.key), jax.random.key_data(other.key)
        ):
            return False

        return True

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

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """
        Structural equality for Pomp result types.

        Extends BaseResult equality by comparing theta.
        """
        if not super().__eq__(other):
            return False

        # theta is a list of plain dicts; rely on Python's structural equality
        if self.theta != other.theta:
            return False

        return True


@dataclass
class PanelPompBaseResult(BaseResult):
    """Base class for PanelPomp results."""

    theta: "PanelParameters | None" = None

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """
        Structural equality for PanelPomp result types.

        Extends BaseResult equality by comparing PanelParameters.
        """
        if not super().__eq__(other):
            return False

        if (self.theta is None) != (other.theta is None):
            return False

        if self.theta is not None and self.theta != other.theta:
            return False

        return True


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

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Structural equality including log-likelihoods and diagnostics."""
        if not super().__eq__(other):
            return False

        if self.J != other.J or self.reps != other.reps or self.thresh != other.thresh:
            return False

        # logLiks
        if isinstance(self.logLiks, xr.DataArray) and isinstance(
            other.logLiks, xr.DataArray
        ):
            if not self.logLiks.equals(other.logLiks):
                return False
        else:
            if not np.array_equal(
                np.asarray(self.logLiks), np.asarray(other.logLiks), equal_nan=True
            ):
                return False

        # Optional diagnostics
        for name in ["CLL", "ESS", "filter_mean", "prediction_mean"]:
            a = getattr(self, name)
            b = getattr(other, name)
            if (a is None) != (b is None):
                return False
            if a is not None and b is not None:
                if isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray):
                    if not a.equals(b):
                        return False
                else:
                    if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
                        return False

        return True

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

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Structural equality including traces and algorithmic settings."""
        if not super().__eq__(other):
            return False

        if (
            self.J != other.J
            or self.M != other.M
            or self.a != other.a
            or self.thresh != other.thresh
        ):
            return False

        # rw_sd comparison: rely on its own __eq__ if present
        if (self.rw_sd is None) != (other.rw_sd is None):
            return False
        if self.rw_sd is not None and self.rw_sd != other.rw_sd:
            return False

        # traces_da
        if isinstance(self.traces_da, xr.DataArray) and isinstance(
            other.traces_da, xr.DataArray
        ):
            if not self.traces_da.equals(other.traces_da):
                return False
        else:
            if not np.array_equal(
                np.asarray(self.traces_da),
                np.asarray(other.traces_da),
                equal_nan=True,
            ):
                return False

        return True

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

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Structural equality including traces and optimizer settings."""
        if not super().__eq__(other):
            return False

        scalar_fields = [
            "optimizer",
            "J",
            "M",
            "eta",
            "alpha",
            "thresh",
            "ls",
            "c",
            "max_ls_itn",
        ]
        for name in scalar_fields:
            if getattr(self, name) != getattr(other, name):
                return False

        # traces_da
        if isinstance(self.traces_da, xr.DataArray) and isinstance(
            other.traces_da, xr.DataArray
        ):
            if not self.traces_da.equals(other.traces_da):
                return False
        else:
            if not np.array_equal(
                np.asarray(self.traces_da),
                np.asarray(other.traces_da),
                equal_nan=True,
            ):
                return False

        return True

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
    theta: "PanelParameters | None" = None

    def __post_init__(self):
        """Set method to pfilter."""
        self.method = "pfilter"

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Structural equality including panel log-likelihoods."""
        if not super().__eq__(other):
            return False

        if self.J != other.J or self.reps != other.reps or self.thresh != other.thresh:
            return False

        if isinstance(self.logLiks, xr.DataArray) and isinstance(
            other.logLiks, xr.DataArray
        ):
            if not self.logLiks.equals(other.logLiks):
                return False
        else:
            if not np.array_equal(
                np.asarray(self.logLiks),
                np.asarray(other.logLiks),
                equal_nan=True,
            ):
                return False

        return True

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

        # Extract shared/unit_specific from theta
        if self.theta is not None:
            shared_list: list[pd.DataFrame] = []
            unit_specific_list: list[pd.DataFrame] = []
            for i in range(len(self.theta._theta)):
                shared_df = self.theta._theta[i].get("shared")
                unit_specific_df = self.theta._theta[i].get("unit_specific")
                if shared_df is not None:
                    shared_list.append(shared_df)
                if unit_specific_df is not None:
                    unit_specific_list.append(unit_specific_df)

            if shared_list:
                s_params = pd.concat(shared_list, axis=1).T.reset_index(drop=True)
                df = df.join(s_params, on="replicate")

            if unit_specific_list:
                u_params = (
                    pd.concat(unit_specific_list, keys=range(len(unit_specific_list)))
                    .stack()
                    .unstack(level=1)
                    .reset_index()
                )
                col_names = list(u_params.columns)
                u_params.rename(
                    columns={col_names[0]: "replicate", col_names[1]: "unit"},
                    inplace=True,
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

        if self.theta is not None:
            shared_list: list[pd.DataFrame] = []
            unit_specific_list: list[pd.DataFrame] = []
            for i in range(len(self.theta._theta)):
                shared_df = self.theta._theta[i].get("shared")
                unit_specific_df = self.theta._theta[i].get("unit_specific")
                if shared_df is not None:
                    shared_list.append(shared_df)
                if unit_specific_df is not None:
                    unit_specific_list.append(unit_specific_df)

            if shared_list:
                p_s = pd.concat(shared_list, axis=1).T.set_axis(reps, axis=0)
                df_s = df_s.join(p_s, on="replicate")
                df_u = df_u.join(p_s, on="replicate")

            if unit_specific_list:
                p_u = pd.concat(unit_specific_list, keys=reps).stack().unstack(level=1)
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
    theta: "PanelParameters | None" = None
    J: int = 0
    M: int = 0
    rw_sd: RWSigma | None = None
    a: float = 0.0
    thresh: float = 0.0
    block: bool = True

    def __post_init__(self):
        """Set method to mif."""
        self.method = "mif"

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Structural equality including traces, log-likelihoods, and settings."""
        if not super().__eq__(other):
            return False

        if (
            self.J != other.J
            or self.M != other.M
            or self.a != other.a
            or self.thresh != other.thresh
            or self.block != other.block
        ):
            return False

        # rw_sd comparison
        if (self.rw_sd is None) != (other.rw_sd is None):
            return False
        if self.rw_sd is not None and self.rw_sd != other.rw_sd:
            return False

        # shared_traces, unit_traces, logLiks
        for name in ["shared_traces", "unit_traces", "logLiks"]:
            a = getattr(self, name)
            b = getattr(other, name)
            if isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray):
                if not a.equals(b):
                    return False
            else:
                if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
                    return False

        return True

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

        u_df = u_df.join(s_df, on="replicate").reset_index()

        cols = ["replicate", "shared logLik", "unit", "unit logLik"] + [
            c
            for c in u_df.columns
            if c not in {"replicate", "shared logLik", "unit", "unit logLik"}
        ]
        u_df = u_df[cols]

        assert isinstance(u_df, pd.DataFrame)

        return u_df

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

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """
        Structural equality for ResultsHistory.

        Two histories are equal if they contain the same sequence of result
        objects (compared via their own __eq__ implementations).
        """
        if not isinstance(other, type(self)):
            return False

        if len(self._entries) != len(other._entries):
            return False

        for a, b in zip(self._entries, other._entries):
            if a != b:
                return False

        return True

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

from dataclasses import dataclass, field
import pandas as pd
import xarray as xr
import numpy as np
import warnings

from .base import BaseResult, PanelPompEstimationTracesMixin, _merge_results
from ...maths import logmeanexp, logmeanexp_se
from ..rw_sigma import RWSigma
from ..learning_rate import LearningRate
from ..parameters import PanelParameters
from ..optimizer import Optimizer, Adam


@dataclass(eq=False)
class PanelPompBaseResult(BaseResult):
    """Base class for PanelPomp results."""

    theta: PanelParameters | None = None
    """The panel parameter object used for the computation."""


@dataclass(eq=False)
class PanelPompPFilterResult(PanelPompBaseResult):
    """Result from PanelPomp.pfilter() method."""

    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each parameter set, replicate, and unit."""
    J: int = 0
    """The number of particles used for filtering."""
    reps: int = 1
    """The number of replicates for each parameter set."""
    thresh: float = 0.0
    """The resampling threshold used."""
    CLL_da: xr.DataArray | None = None
    """Conditional log-likelihoods for each unit and time point."""
    ESS_da: xr.DataArray | None = None
    """Effective Sample Size for each unit and time point."""
    filter_mean: xr.DataArray | None = None
    """The mean of the filtering distribution for each state variable."""
    prediction_mean: xr.DataArray | None = None
    """The mean of the predictive distribution for each state variable."""

    def __post_init__(self):
        self.method = "pfilter"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of particles (J)", "J"),
            ("Number of replicates", "reps"),
            ("Resampling threshold", "thresh"),
        ]

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert panel pfilter result to DataFrame."""
        ll = logmeanexp(self.logLiks.values, axis=-1, ignore_nan=ignore_nan)
        unit_names = self.logLiks.coords["unit"].values
        se_unit = (
            logmeanexp_se(self.logLiks.values, axis=-1, ignore_nan=ignore_nan)
            if self.logLiks.shape[-1] > 1
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
            pd.DataFrame(se_unit, columns=self.logLiks.coords["unit"].values)
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

        if self.theta is not None and self.theta.num_replicates() > 0:
            shared_names = self.theta.get_shared_param_names()
            if shared_names and "shared" in self.theta._data:
                s_vals = self.theta._data["shared"].sel(parameter=shared_names).values
                p_s = pd.DataFrame(s_vals, columns=shared_names)
                df = df.join(p_s, on="theta_idx")

            specific_names = self.theta.get_unit_param_names()
            if specific_names and "unit_specific" in self.theta._data:
                p_u = (
                    self.theta._data["unit_specific"]
                    .sel(parameter=specific_names)
                    .to_dataset(dim="parameter")
                    .to_dataframe()
                    .reset_index()
                )
                df = df.merge(p_u, on=["theta_idx", "unit"], how="left")
        return df

    def CLL(self, average: bool = False) -> pd.DataFrame:
        """Return conditional log-likelihoods as a DataFrame."""
        if self.CLL_da is None or self.CLL_da.size == 0:
            return pd.DataFrame()
        if not average:
            return self.CLL_da.to_dataframe(name="CLL").reset_index()
        avg = logmeanexp(np.asarray(self.CLL_da.values), axis=2)
        return (
            xr.DataArray(
                avg,
                dims=["theta_idx", "unit", "time"],
                coords={
                    "theta_idx": self.CLL_da.coords.get(
                        "theta_idx", np.arange(avg.shape[0])
                    ),
                    "unit": self.CLL_da.coords["unit"].values,
                    "time": self.CLL_da.coords.get("time", np.arange(avg.shape[2])),
                },
            )
            .to_dataframe(name="CLL")
            .reset_index()
        )

    def ESS(self, average: bool = False) -> pd.DataFrame:
        """Return Effective Sample Size as a DataFrame."""
        if self.ESS_da is None or self.ESS_da.size == 0:
            return pd.DataFrame()
        ess = self.ESS_da.mean(dim="rep") if average else self.ESS_da
        return ess.to_dataframe(name="ESS").reset_index()

    def traces(self) -> pd.DataFrame:
        """Return pfilter results formatted as traces (long format)."""
        ll = logmeanexp(self.logLiks.values, axis=-1)
        se_unit = (
            logmeanexp_se(self.logLiks.values, axis=-1)
            if self.logLiks.shape[-1] > 1
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
            pd.DataFrame(ll, columns=self.logLiks.coords["unit"].values, index=reps)
            .melt(ignore_index=False, var_name="unit", value_name="logLik")
            .reset_index()
            .rename(columns={"index": "theta_idx"})
        )
        df_se_u = (
            pd.DataFrame(
                se_unit, columns=self.logLiks.coords["unit"].values, index=reps
            )
            .melt(ignore_index=False, var_name="unit", value_name="se")
            .reset_index()
            .rename(columns={"index": "theta_idx"})
        )
        df_u = pd.merge(df_u, df_se_u, on=["theta_idx", "unit"], how="left")

        if self.theta is not None and self.theta.num_replicates() > 0:
            shared_names = self.theta.get_shared_param_names()
            if shared_names and "shared" in self.theta._data:
                s_vals = self.theta._data["shared"].sel(parameter=shared_names).values
                p_s = pd.DataFrame(s_vals, columns=shared_names)
                df_s, df_u = (
                    df_s.join(p_s, on="theta_idx"),
                    df_u.join(p_s, on="theta_idx"),
                )

            specific_names = self.theta.get_unit_param_names()
            if specific_names and "unit_specific" in self.theta._data:
                p_u = (
                    self.theta._data["unit_specific"]
                    .sel(parameter=specific_names)
                    .to_dataset(dim="parameter")
                    .to_dataframe()
                    .reset_index()
                )
                df_u = df_u.merge(p_u, on=["theta_idx", "unit"], how="left")
        dfs_to_concat = [df for df in [df_s, df_u] if not df.empty]
        if not dfs_to_concat:
            return pd.DataFrame()

        all_cols = dfs_to_concat[0].columns
        for df in dfs_to_concat[1:]:
            all_cols = all_cols.union(df.columns)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = pd.concat(dfs_to_concat, ignore_index=True)
        df = df.assign(method="pfilter", iteration=0)
        cols = ["theta_idx", "unit", "iteration", "method", "logLik", "se"]
        other_cols = [c for c in df.columns if c not in cols]
        return df.loc[:, cols + other_cols].copy()

    @staticmethod
    def merge(*results: "PanelPompPFilterResult") -> "PanelPompPFilterResult":
        return _merge_results(
            PanelPompPFilterResult,
            results,
            ["J", "reps", "thresh", "method"],
            ["logLiks", "CLL_da", "ESS_da", "filter_mean", "prediction_mean"],
        )


@dataclass(eq=False)
class PanelPompMIFResult(PanelPompEstimationTracesMixin, PanelPompBaseResult):
    """Result from PanelPomp.mif() method."""

    shared_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Shared parameter traces across iterations."""
    unit_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Unit-specific parameter traces across iterations."""
    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each unit across iterations."""
    J: int = 0
    """The number of particles used for filtering."""
    M: int = 0
    """The number of iterations performed."""
    rw_sd: RWSigma | None = None
    """The random walk standard deviations for parameter perturbation."""
    thresh: float = 0.0
    """The resampling threshold used."""
    n_monitors: int = 0
    """The number of particle filters used to estimate log-likelihoods at each iteration."""
    block: bool = False
    """Whether block-style filtering was used."""

    def __post_init__(self):
        self.method = "mif"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of particles (J)", "J"),
            ("Number of iterations (M)", "M"),
            ("Resampling threshold", "thresh"),
            ("Number of monitors", "n_monitors"),
            ("Block", "block"),
        ]

    @staticmethod
    def merge(*results: "PanelPompMIFResult") -> "PanelPompMIFResult":
        return _merge_results(
            PanelPompMIFResult,
            results,
            ["J", "M", "thresh", "n_monitors", "block", "rw_sd", "method"],
            ["shared_traces", "unit_traces", "logLiks"],
        )


@dataclass(eq=False)
class PanelPompTrainResult(PanelPompEstimationTracesMixin, PanelPompBaseResult):
    """Result from PanelPomp.train() method."""

    shared_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Shared parameter traces across iterations."""
    unit_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Unit-specific parameter traces across iterations."""
    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each unit across iterations."""
    optimizer: Optimizer = field(default_factory=Adam)
    """The optimizer used for training."""
    J: int = 0
    """The number of particles used for filtering."""
    M: int = 0
    """The number of iterations performed."""
    eta: LearningRate | None = None
    """The learning rate object."""
    alpha: float = 0.97
    """The discount factor for the gradient moving average."""
    alpha_cooling: float = 1.0
    """The cooling factor for the discount factor."""

    def __post_init__(self):
        self.method = "train"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Optimizer", "optimizer"),
            ("Number of particles (J)", "J"),
            ("Number of iterations (M)", "M"),
            ("Learning rate (eta)", "eta"),
            ("Discount factor (alpha)", "alpha"),
            ("Cooling factor for alpha", "alpha_cooling"),
        ]

    @staticmethod
    def merge(*results: "PanelPompTrainResult") -> "PanelPompTrainResult":
        return _merge_results(
            PanelPompTrainResult,
            results,
            ["optimizer", "J", "M", "eta", "alpha", "alpha_cooling", "method"],
            ["shared_traces", "unit_traces", "logLiks"],
        )


@dataclass(eq=False)
class PanelPompDpopTrainResult(PanelPompBaseResult):
    """Result from PanelPomp.dpop_train() method."""

    shared_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    unit_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    theta: "PanelParameters | None" = None
    optimizer: Optimizer = field(default_factory=Adam)
    J: int = 0
    M: int = 0
    eta: LearningRate | dict[str, float] | float = field(default_factory=lambda: {})
    alpha: float = 0.97
    alpha_cooling: float = 1.0
    process_weight_state: str | None = None
    decay: float = 0.0

    def __post_init__(self):
        self.method = "dpop_train"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Optimizer", "optimizer"),
            ("Number of particles (J)", "J"),
            ("Number of iterations (M)", "M"),
            ("Learning rate (eta)", "eta"),
            ("Discount factor (alpha)", "alpha"),
            ("Cooling factor for alpha", "alpha_cooling"),
            ("Process weight state", "process_weight_state"),
            ("Decay", "decay"),
        ]

    def __eq__(self, other) -> bool:  # type: ignore[override]
        """Structural equality including traces, log-likelihoods, and settings."""
        if not super().__eq__(other):
            return False

        if (
            self.optimizer != other.optimizer
            or self.J != other.J
            or self.M != other.M
            or self.eta != other.eta
            or self.alpha != other.alpha
            or self.alpha_cooling != other.alpha_cooling
            or self.process_weight_state != other.process_weight_state
            or self.decay != other.decay
        ):
            return False

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
        rep_dim = (
            "theta_idx" if "theta_idx" in self.shared_traces.dims else "replicate"
        )
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

        if "iteration" in s_df.columns:
            s_df = s_df.drop(columns=["iteration"])

        u_df = u_df.join(s_df, on=rep_dim).reset_index()
        if rep_dim != "theta_idx":
            u_df = u_df.rename(columns={rep_dim: "theta_idx"})

        cols = ["theta_idx", "iteration", "shared logLik", "unit", "unit logLik"] + [
            c
            for c in u_df.columns
            if c
            not in {"theta_idx", "iteration", "shared logLik", "unit", "unit logLik"}
        ]
        u_df = u_df[cols]

        assert isinstance(u_df, pd.DataFrame)

        return u_df

    def traces(self) -> pd.DataFrame:
        """Return panel dpop_train results formatted as traces (long format)."""
        if self.shared_traces.size == 0:
            return pd.DataFrame()

        df_s = (
            self.shared_traces.to_dataset(dim="variable").to_dataframe().reset_index()
        )
        if "replicate" in df_s.columns and "theta_idx" not in df_s.columns:
            df_s = df_s.rename(columns={"replicate": "theta_idx"})
        df_s["unit"] = "shared"

        df_u = self.unit_traces.to_dataset(dim="variable").to_dataframe().reset_index()
        if "replicate" in df_u.columns and "theta_idx" not in df_u.columns:
            df_u = df_u.rename(columns={"replicate": "theta_idx"})
        df_u = df_u.rename(columns={"unitLogLik": "logLik"})

        meta_cols = {"theta_idx", "iteration", "logLik", "unit"}
        shared_params = [c for c in df_s.columns if c not in meta_cols]

        if shared_params:
            df_u = df_u.merge(
                df_s[["theta_idx", "iteration"] + shared_params],
                on=["theta_idx", "iteration"],
                how="left",
            )

        df = pd.concat([df_s, df_u], ignore_index=True).assign(method="dpop_train")
        cols = ["theta_idx", "unit", "iteration", "method", "logLik"]
        return df.loc[:, cols + [c for c in df.columns if c not in cols]].copy()

    def CLL(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def ESS(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def print_summary(self, n: int = 5):
        """Print summary of panel dpop_train result."""
        print(f"Method: {self.method}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of iterations (M): {self.M}")
        print(f"Learning rate (eta): {self.eta}")
        print(f"Discount factor (alpha): {self.alpha}")
        print(f"Cooling factor for alpha: {self.alpha_cooling}")
        print(f"Process weight state: {self.process_weight_state}")
        print(f"Decay: {self.decay}")
        print(f"Execution time: {self.execution_time} seconds")
        df = self.to_dataframe()
        if not df.empty:
            print(f"\nTop {n} Results:")
            df_sorted = df.sort_values("shared logLik", ascending=False).head(n)
            print(df_sorted.to_string())

    @staticmethod
    def merge(*results: "PanelPompDpopTrainResult") -> "PanelPompDpopTrainResult":
        """Merge parameter sets from multiple PanelPompDpopTrainResult objects."""
        if len(results) == 0:
            raise ValueError(
                "At least one PanelPompDpopTrainResult object must be provided."
            )
        first = results[0]

        for result in results:
            if not isinstance(result, type(first)):
                raise TypeError(
                    "All merged objects must be of type PanelPompDpopTrainResult."
                )
            if (
                result.optimizer != first.optimizer
                or result.J != first.J
                or result.M != first.M
                or result.eta != first.eta
                or result.alpha != first.alpha
                or result.alpha_cooling != first.alpha_cooling
            ):
                raise ValueError(
                    "All PanelPompDpopTrainResult objects must have the same optimizer, J, M, eta, alpha, and alpha_cooling."
                )

        merged_theta = (
            PanelParameters.merge(*[r.theta for r in results if r.theta is not None])
            if any(r.theta is not None for r in results)
            else None
        )

        def _theta_idx_array(array: xr.DataArray) -> xr.DataArray:
            if "replicate" in array.dims and "theta_idx" not in array.dims:
                array = array.rename({"replicate": "theta_idx"})
            return array

        shared_trace_arrays = [
            _theta_idx_array(r.shared_traces)
            for r in results
            if r.shared_traces.size > 0
        ]
        merged_shared_traces = (
            xr.concat(shared_trace_arrays, dim="theta_idx").assign_coords(
                theta_idx=lambda x: np.arange(x.sizes["theta_idx"])
            )
            if shared_trace_arrays
            else xr.DataArray([])
        )  # type: ignore[assignment]

        unit_trace_arrays = [
            _theta_idx_array(r.unit_traces) for r in results if r.unit_traces.size > 0
        ]
        merged_unit_traces = (
            xr.concat(unit_trace_arrays, dim="theta_idx").assign_coords(
                theta_idx=lambda x: np.arange(x.sizes["theta_idx"])
            )
            if unit_trace_arrays
            else xr.DataArray([])
        )  # type: ignore[assignment]

        logLik_arrays = [
            _theta_idx_array(r.logLiks) for r in results if r.logLiks.size > 0
        ]
        merged_logLiks = (
            xr.concat(logLik_arrays, dim="theta_idx").assign_coords(
                theta_idx=lambda x: np.arange(x.sizes["theta_idx"])
            )
            if logLik_arrays
            else xr.DataArray([])
        )  # type: ignore[assignment]

        execution_times = [
            r.execution_time for r in results if r.execution_time is not None
        ]
        max_execution_time = max(execution_times) if execution_times else None

        return PanelPompDpopTrainResult(
            method=first.method,
            execution_time=max_execution_time,
            key=first.key,
            theta=merged_theta,
            shared_traces=merged_shared_traces,
            unit_traces=merged_unit_traces,
            logLiks=merged_logLiks,
            optimizer=first.optimizer,
            J=first.J,
            M=first.M,
            eta=first.eta,
            alpha=first.alpha,
            alpha_cooling=first.alpha_cooling,
            process_weight_state=first.process_weight_state,
            decay=first.decay,
        )

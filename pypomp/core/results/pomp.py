from dataclasses import dataclass, field
import pandas as pd
import xarray as xr
import numpy as np

from .base import BaseResult, PompEstimationTracesMixin, _merge_results
from ...maths import logmeanexp, logmeanexp_se
from ..rw_sigma import RWSigma
from ..learning_rate import LearningRate
from ..optimizer import Optimizer, Adam


@dataclass(eq=False)
class PompBaseResult(BaseResult):
    """Base class for Pomp results."""

    theta: list[dict] = field(default_factory=list)
    """The list of parameter sets used for the computation."""


@dataclass(eq=False)
class PompPFilterResult(PompBaseResult):
    """Result from Pomp.pfilter() method."""

    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each parameter set and replicate."""
    J: int = 0
    """The number of particles used for filtering."""
    reps: int = 1
    """The number of replicates for each parameter set."""
    thresh: float = 0.0
    """The resampling threshold used by the filter."""
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
        """Convert pfilter result to DataFrame."""
        if not self.theta or self.logLiks.size == 0:
            return pd.DataFrame()
        arr = np.asarray(getattr(self.logLiks, "values", self.logLiks))
        logLik = logmeanexp(arr, axis=-1, ignore_nan=ignore_nan)
        se = (
            logmeanexp_se(arr, axis=-1, ignore_nan=ignore_nan)
            if arr.shape[-1] > 1
            else np.full_like(logLik, np.nan)
        )
        theta_df = pd.DataFrame(self.theta)
        df = pd.DataFrame(
            {"theta_idx": np.arange(len(theta_df)), "logLik": logLik, "se": se}
        )
        return pd.concat([df, theta_df], axis=1)

    def CLL(self, average: bool = False) -> pd.DataFrame:
        """Return conditional log-likelihoods as a DataFrame."""
        if self.CLL_da is None or self.CLL_da.size == 0:
            return pd.DataFrame()
        if not average:
            return self.CLL_da.to_dataframe(name="CLL").reset_index()
        avg = logmeanexp(np.asarray(self.CLL_da.values), axis=1)
        return (
            xr.DataArray(
                avg,
                dims=["theta_idx", "time"],
                coords={
                    "theta_idx": self.CLL_da.coords.get(
                        "theta_idx", np.arange(avg.shape[0])
                    ),
                    "time": self.CLL_da.coords.get("time", np.arange(avg.shape[1])),
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
        """Return traces DataFrame for this pfilter result."""
        if not self.theta or not len(self.logLiks):
            return pd.DataFrame()
        logliks = logmeanexp(
            np.asarray(getattr(self.logLiks, "values", self.logLiks)), axis=-1
        )
        base_df = pd.DataFrame(
            {
                "theta_idx": np.arange(len(self.theta)),
                "iteration": 0,
                "method": self.method,
                "logLik": logliks,
            }
        )
        if not self.theta:
            return base_df
        return pd.concat([base_df, pd.DataFrame(self.theta)], axis=1)

    @staticmethod
    def merge(*results: "PompPFilterResult") -> "PompPFilterResult":
        return _merge_results(
            PompPFilterResult,
            results,
            ["J", "reps", "thresh", "method"],
            ["logLiks", "CLL_da", "ESS_da", "filter_mean", "prediction_mean"],
        )


@dataclass(eq=False)
class PompMIFResult(PompEstimationTracesMixin, PompBaseResult):
    """Result from Pomp.mif() method."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Parameter traces and log-likelihoods across iterations."""
    J: int = 0
    """The number of particles used for filtering."""
    M: int = 0
    """The number of iterations performed."""
    rw_sd: RWSigma | None = None
    """The random walk standard deviations for parameter perturbation."""
    a: float = 0.0
    """The cooling fraction used."""
    thresh: float = 0.0
    """The resampling threshold used."""
    n_monitors: int = 0
    """The number of particle filters used to estimate log-likelihoods at each iteration."""

    def __post_init__(self):
        self.method = "mif"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of particles (J)", "J"),
            ("Number of iterations (M)", "M"),
            ("Cooling fraction (a)", "a"),
            ("Resampling threshold", "thresh"),
            ("Number of monitors", "n_monitors"),
        ]

    @staticmethod
    def merge(*results: "PompMIFResult") -> "PompMIFResult":
        return _merge_results(
            PompMIFResult,
            results,
            ["J", "M", "a", "thresh", "n_monitors", "rw_sd", "method"],
            ["traces_da"],
        )


@dataclass(eq=False)
class PompTrainResult(PompEstimationTracesMixin, PompBaseResult):
    """Result from Pomp.train() method."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))  # type: ignore[assignment]
    """Parameter traces and log-likelihoods across iterations."""
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
    thresh: int = 0
    """The resampling threshold used."""
    alpha_cooling: float = 1.0
    """The cooling factor for the discount factor."""

    def __post_init__(self):
        self.method = "train"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        config = [
            ("Number of parameter sets", "theta"),
            ("Optimizer", "optimizer"),
            ("Number of particles (J)", "J"),
            ("Number of iterations (M)", "M"),
            ("Learning rate (eta)", "eta"),
            ("Discount factor (alpha)", "alpha"),
            ("Resampling threshold", "thresh"),
            ("Cooling factor for alpha", "alpha_cooling"),
        ]
        return config

    @staticmethod
    def merge(*results: "PompTrainResult") -> "PompTrainResult":
        return _merge_results(
            PompTrainResult,
            results,
            [
                "optimizer",
                "J",
                "M",
                "eta",
                "alpha",
                "thresh",
                "alpha_cooling",
                "method",
            ],
            ["traces_da"],
        )


@dataclass(eq=False)
class PompPMCMCResult(PompBaseResult):
    """Result from Pomp.pmcmc() method.

    Stores the full MCMC trace (log-likelihood, log-prior, and all
    parameter values at each iteration), the number of accepted
    proposals, and algorithmic settings.
    """

    traces_arr: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    """Array of shape ``(Nmcmc + 1, 2 + n_params)`` with columns
    ``[loglik, log_prior, param_1, ..., param_p]``."""

    trace_names: list[str] = field(default_factory=list)
    """Column names for *traces_arr*: ``["loglik", "log_prior", p1, ...]``."""

    Nmcmc: int = 0
    J: int = 0
    reps: int = 1
    accepts: int = 0

    def __post_init__(self):
        self.method = "pmcmc"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of MCMC iterations (Nmcmc)", "Nmcmc"),
            ("Number of particles (J)", "J"),
            ("Number of replicates", "reps"),
            ("Number of accepted proposals", "accepts"),
        ]

    def __eq__(self, other) -> bool:  # type: ignore[override]
        if not super().__eq__(other):
            return False
        if (
            self.Nmcmc != other.Nmcmc
            or self.J != other.J
            or self.reps != other.reps
            or self.accepts != other.accepts
        ):
            return False
        if self.trace_names != other.trace_names:
            return False
        if not np.array_equal(
            np.asarray(self.traces_arr), np.asarray(other.traces_arr), equal_nan=True
        ):
            return False
        return True

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / max(self.Nmcmc, 1)

    def traces(self) -> pd.DataFrame:
        """Return traces as a DataFrame compatible with ResultsHistory."""
        if self.traces_arr.size == 0:
            return pd.DataFrame()
        df = pd.DataFrame(self.traces_arr, columns=self.trace_names)
        # Use embedded _chain column from merge(), or default to 0
        if "_chain" in df.columns:
            df["chain"] = df.pop("_chain").astype(int)
        else:
            df.insert(0, "chain", 0)
        df.insert(0, "iteration", np.arange(len(df)))
        df.insert(0, "replicate", 0)
        return df.assign(method="pmcmc")

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert traces to a DataFrame."""
        df = pd.DataFrame(self.traces_arr, columns=self.trace_names)
        # Use embedded _chain column from merge(), or default to 0
        if "_chain" in df.columns:
            df["chain"] = df.pop("_chain").astype(int)
        else:
            df.insert(0, "chain", 0)
        df.insert(0, "iteration", np.arange(len(df)))
        if ignore_nan:
            df = df.dropna()
        return df

    def CLL(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def ESS(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def print_summary(self, n: int = 5):
        """Print a summary of the PMCMC result."""
        print(f"Method: {self.method}")
        print(f"Number of particles (J): {self.J}")
        print(f"MCMC iterations (Nmcmc): {self.Nmcmc}")
        print(f"PFilter replicates per iteration: {self.reps}")
        print(f"Accepted proposals: {self.accepts}")
        print(f"Acceptance rate: {self.acceptance_rate:.3f}")
        print(f"Execution time: {self.execution_time} seconds")
        if self.traces_arr.size > 0:
            print(f"\nFinal loglik: {self.traces_arr[-1, 0]:.4f}")
            print(f"Final log_prior: {self.traces_arr[-1, 1]:.4f}")

    @staticmethod
    def merge(*results: "PompPMCMCResult") -> "PompPMCMCResult":
        """Concatenate traces from multiple PMCMC runs (e.g. multiple chains).

        Each input result is treated as a separate chain.  The merged
        ``traces()`` and ``to_dataframe()`` DataFrames include a ``chain``
        column so individual chains remain distinguishable.
        """
        if len(results) == 0:
            raise ValueError("At least one PompPMCMCResult must be provided.")
        first = results[0]
        for r in results:
            if not isinstance(r, type(first)):
                raise TypeError("All results must be PompPMCMCResult.")
            if r.J != first.J or r.reps != first.reps:
                raise ValueError("All results must have the same J and reps.")

        # Add a chain-id column (last column) before concatenating
        parts = []
        for chain_id, r in enumerate(results):
            chain_col = np.full((r.traces_arr.shape[0], 1), chain_id, dtype=float)
            parts.append(np.concatenate([r.traces_arr, chain_col], axis=1))
        merged_traces = np.concatenate(parts, axis=0)
        merged_trace_names = first.trace_names + ["_chain"]

        merged_theta = sum((r.theta for r in results), [])
        total_accepts = sum(r.accepts for r in results)
        total_Nmcmc = sum(r.Nmcmc for r in results)
        execution_times = [r.execution_time for r in results if r.execution_time is not None]
        max_time = max(execution_times) if execution_times else None

        return PompPMCMCResult(
            method=first.method,
            execution_time=max_time,
            key=first.key,
            theta=merged_theta,
            traces_arr=merged_traces,
            trace_names=merged_trace_names,
            Nmcmc=total_Nmcmc,
            J=first.J,
            reps=first.reps,
            accepts=total_accepts,
        )


@dataclass(eq=False)
class PompABCResult(PompBaseResult):
    """Result from Pomp.abc() method.

    Stores the full ABC-MCMC trace (distance, log-prior, and all
    parameter values at each iteration), the number of accepted
    proposals, and algorithmic settings.
    """

    traces_arr: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    """Array of shape ``(Nabc + 1, 2 + n_params)`` with columns
    ``[distance, log_prior, param_1, ..., param_p]``."""

    trace_names: list[str] = field(default_factory=list)
    """Column names for *traces_arr*: ``["distance", "log_prior", p1, ...]``."""

    Nabc: int = 0
    epsilon: float = 0.0
    accepts: int = 0

    def __post_init__(self):
        self.method = "abc"

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of ABC iterations (Nabc)", "Nabc"),
            ("Tolerance (epsilon)", "epsilon"),
            ("Number of accepted proposals", "accepts"),
        ]

    def __eq__(self, other) -> bool:  # type: ignore[override]
        if not super().__eq__(other):
            return False
        if (
            self.Nabc != other.Nabc
            or self.epsilon != other.epsilon
            or self.accepts != other.accepts
        ):
            return False
        if self.trace_names != other.trace_names:
            return False
        if not np.array_equal(
            np.asarray(self.traces_arr), np.asarray(other.traces_arr), equal_nan=True
        ):
            return False
        return True

    @property
    def acceptance_rate(self) -> float:
        return self.accepts / max(self.Nabc, 1)

    def traces(self) -> pd.DataFrame:
        """Return traces as a DataFrame compatible with ResultsHistory."""
        if self.traces_arr.size == 0:
            return pd.DataFrame()
        df = pd.DataFrame(self.traces_arr, columns=self.trace_names)
        if "_chain" in df.columns:
            df["chain"] = df.pop("_chain").astype(int)
        else:
            df.insert(0, "chain", 0)
        df.insert(0, "iteration", np.arange(len(df)))
        df.insert(0, "replicate", 0)
        return df.assign(method="abc")

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert traces to a DataFrame."""
        df = pd.DataFrame(self.traces_arr, columns=self.trace_names)
        if "_chain" in df.columns:
            df["chain"] = df.pop("_chain").astype(int)
        else:
            df.insert(0, "chain", 0)
        df.insert(0, "iteration", np.arange(len(df)))
        if ignore_nan:
            df = df.dropna()
        return df

    def CLL(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def ESS(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def print_summary(self, n: int = 5):
        """Print a summary of the ABC result."""
        print(f"Method: {self.method}")
        print(f"ABC iterations (Nabc): {self.Nabc}")
        print(f"Tolerance (epsilon): {self.epsilon}")
        print(f"Accepted proposals: {self.accepts}")
        print(f"Acceptance rate: {self.acceptance_rate:.3f}")
        print(f"Execution time: {self.execution_time} seconds")
        if self.traces_arr.size > 0:
            print(f"\nFinal distance: {self.traces_arr[-1, 0]:.4f}")
            print(f"Final log_prior: {self.traces_arr[-1, 1]:.4f}")

    @staticmethod
    def merge(*results: "PompABCResult") -> "PompABCResult":
        """Concatenate traces from multiple ABC runs (e.g. multiple chains).

        Each input result is treated as a separate chain.  The merged
        ``traces()`` and ``to_dataframe()`` DataFrames include a ``chain``
        column so individual chains remain distinguishable.
        """
        if len(results) == 0:
            raise ValueError("At least one PompABCResult must be provided.")
        first = results[0]
        for r in results:
            if not isinstance(r, type(first)):
                raise TypeError("All results must be PompABCResult.")
            if r.epsilon != first.epsilon:
                raise ValueError("All results must have the same epsilon.")

        parts = []
        for chain_id, r in enumerate(results):
            chain_col = np.full((r.traces_arr.shape[0], 1), chain_id, dtype=float)
            parts.append(np.concatenate([r.traces_arr, chain_col], axis=1))
        merged_traces = np.concatenate(parts, axis=0)
        merged_trace_names = first.trace_names + ["_chain"]

        merged_theta = sum((r.theta for r in results), [])
        total_accepts = sum(r.accepts for r in results)
        total_Nabc = sum(r.Nabc for r in results)
        execution_times = [r.execution_time for r in results if r.execution_time is not None]
        max_time = max(execution_times) if execution_times else None

        return PompABCResult(
            method=first.method,
            execution_time=max_time,
            key=first.key,
            theta=merged_theta,
            traces_arr=merged_traces,
            trace_names=merged_trace_names,
            Nabc=total_Nabc,
            epsilon=first.epsilon,
            accepts=total_accepts,
        )



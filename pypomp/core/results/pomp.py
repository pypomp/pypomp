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

    Stores per-chain PMCMC traces in an :class:`xarray.DataArray` with
    dimensions ``(theta_idx, iteration, variable)``, where ``variable``
    enumerates ``"logLik"``, ``"log_prior"`` and each parameter name.
    """

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Per-chain MCMC trace array.  Dims ``(theta_idx, iteration, variable)``."""

    Nmcmc: int = 0
    """Number of MCMC iterations per chain."""

    J: int = 0
    """Number of particles per filter evaluation."""

    accepts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    """Per-chain count of accepted proposals.  Shape ``(n_chains,)``."""

    def __post_init__(self):
        self.method = "pmcmc"

    @property
    def n_chains(self) -> int:
        if self.traces_da.size == 0:
            return 0
        return int(self.traces_da.sizes.get("theta_idx", 0))

    @property
    def acceptance_rate(self) -> np.ndarray:
        """Per-chain acceptance rate."""
        if self.Nmcmc <= 0:
            return np.zeros_like(self.accepts, dtype=float)
        return np.asarray(self.accepts, dtype=float) / float(self.Nmcmc)

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of chains", "n_chains"),
            ("Number of MCMC iterations (Nmcmc)", "Nmcmc"),
            ("Number of particles (J)", "J"),
            ("Accepted proposals (per chain)", "accepts"),
        ]

    def __eq__(self, other) -> bool:  # type: ignore[override]
        if not super().__eq__(other):
            return False
        if self.Nmcmc != other.Nmcmc or self.J != other.J:
            return False
        if not np.array_equal(np.asarray(self.accepts), np.asarray(other.accepts)):
            return False
        if self.traces_da.size == 0 and other.traces_da.size == 0:
            return True
        return bool(self.traces_da.equals(other.traces_da))

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert traces to a tidy DataFrame with one row per (chain, iteration)."""
        if self.traces_da.size == 0:
            return pd.DataFrame()
        df = (
            self.traces_da.to_dataframe(name="value")
            .reset_index()
            .pivot_table(
                index=["theta_idx", "iteration"],
                columns="variable",
                values="value",
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .rename(columns={"theta_idx": "chain"})
        )
        # Preserve the original variable ordering from the DataArray's coord.
        var_order = list(self.traces_da.coords["variable"].values)
        cols = ["chain", "iteration"] + [c for c in var_order if c in df.columns]
        df = df[cols]
        if ignore_nan:
            df = df.dropna()
        return df

    def traces(self) -> pd.DataFrame:
        """Tidy DataFrame compatible with :class:`ResultsHistory`."""
        df = self.to_dataframe()
        if df.empty:
            return df
        df.insert(0, "method", self.method)
        df.insert(0, "replicate", 0)
        return df

    def CLL(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def ESS(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def print_summary(self, n: int = 5):
        """Print a summary of the PMCMC result."""
        print(f"Method: {self.method}")
        print(f"Number of chains: {self.n_chains}")
        print(f"Number of particles (J): {self.J}")
        print(f"MCMC iterations (Nmcmc): {self.Nmcmc}")
        if np.asarray(self.accepts).size > 0:
            rates = self.acceptance_rate
            for c in range(int(np.asarray(self.accepts).size)):
                print(
                    f"  chain {c}: accepts={int(self.accepts[c])}, "
                    f"rate={float(rates[c]):.3f}"
                )
        print(f"Execution time: {self.execution_time} seconds")
        if (
            self.traces_da.size > 0
            and "logLik" in list(self.traces_da.coords["variable"].values)
        ):
            last = self.traces_da.isel(iteration=-1).sel(variable="logLik").values
            print(f"\nFinal logLik per chain: {np.asarray(last)}")

    @staticmethod
    def merge(*results: "PompPMCMCResult") -> "PompPMCMCResult":
        """Concatenate per-chain traces along the ``theta_idx`` dimension.

        Each input result may itself contain one or more chains; the merged
        result stacks them together with ``theta_idx`` re-indexed ``0..N-1``.
        All inputs must share the same ``J``, ``Nmcmc`` and ``variable``
        coordinate.
        """
        if len(results) == 0:
            raise ValueError("At least one PompPMCMCResult must be provided.")
        first = results[0]
        for r in results:
            if not isinstance(r, type(first)):
                raise TypeError("All results must be PompPMCMCResult.")
            if r.J != first.J:
                raise ValueError("All results must have the same J.")
            if r.Nmcmc != first.Nmcmc:
                raise ValueError("All results must have the same Nmcmc.")
            if list(r.traces_da.coords["variable"].values) != list(
                first.traces_da.coords["variable"].values
            ):
                raise ValueError(
                    "All results must have the same variable coord ordering."
                )

        merged_da = xr.concat(
            [r.traces_da for r in results], dim="theta_idx"
        ).assign_coords(theta_idx=np.arange(sum(r.n_chains for r in results)))

        merged_theta = sum((r.theta for r in results), [])
        merged_accepts = np.concatenate(
            [np.asarray(r.accepts).ravel() for r in results]
        )
        execution_times = [
            r.execution_time for r in results if r.execution_time is not None
        ]
        max_time = max(execution_times) if execution_times else None

        return PompPMCMCResult(
            method=first.method,
            execution_time=max_time,
            key=first.key,
            theta=merged_theta,
            traces_da=merged_da,
            Nmcmc=first.Nmcmc,
            J=first.J,
            accepts=merged_accepts,
        )


@dataclass(eq=False)
class PompABCResult(PompBaseResult):
    """Result from Pomp.abc() method.

    Stores per-chain ABC-MCMC traces in an :class:`xarray.DataArray` with
    dimensions ``(theta_idx, iteration, variable)``, where ``variable``
    enumerates ``"distance"``, ``"log_prior"`` and each parameter name.
    """

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Per-chain ABC trace array.  Dims ``(theta_idx, iteration, variable)``."""

    Nabc: int = 0
    """Number of ABC iterations per chain."""

    epsilon: float = 0.0
    """Distance threshold (acceptance requires ``distance < epsilon**2``)."""

    accepts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    """Per-chain count of accepted proposals.  Shape ``(n_chains,)``."""

    def __post_init__(self):
        self.method = "abc"

    @property
    def n_chains(self) -> int:
        if self.traces_da.size == 0:
            return 0
        return int(self.traces_da.sizes.get("theta_idx", 0))

    @property
    def acceptance_rate(self) -> np.ndarray:
        """Per-chain acceptance rate."""
        if self.Nabc <= 0:
            return np.zeros_like(self.accepts, dtype=float)
        return np.asarray(self.accepts, dtype=float) / float(self.Nabc)

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of chains", "n_chains"),
            ("Number of ABC iterations (Nabc)", "Nabc"),
            ("Tolerance (epsilon)", "epsilon"),
            ("Accepted proposals (per chain)", "accepts"),
        ]

    def __eq__(self, other) -> bool:  # type: ignore[override]
        if not super().__eq__(other):
            return False
        if self.Nabc != other.Nabc or self.epsilon != other.epsilon:
            return False
        if not np.array_equal(np.asarray(self.accepts), np.asarray(other.accepts)):
            return False
        if self.traces_da.size == 0 and other.traces_da.size == 0:
            return True
        return bool(self.traces_da.equals(other.traces_da))

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert traces to a tidy DataFrame with one row per (chain, iteration)."""
        if self.traces_da.size == 0:
            return pd.DataFrame()
        df = (
            self.traces_da.to_dataframe(name="value")
            .reset_index()
            .pivot_table(
                index=["theta_idx", "iteration"],
                columns="variable",
                values="value",
            )
            .reset_index()
            .rename_axis(None, axis=1)
            .rename(columns={"theta_idx": "chain"})
        )
        var_order = list(self.traces_da.coords["variable"].values)
        cols = ["chain", "iteration"] + [c for c in var_order if c in df.columns]
        df = df[cols]
        if ignore_nan:
            df = df.dropna()
        return df

    def traces(self) -> pd.DataFrame:
        """Tidy DataFrame compatible with :class:`ResultsHistory`."""
        df = self.to_dataframe()
        if df.empty:
            return df
        df.insert(0, "method", self.method)
        df.insert(0, "replicate", 0)
        return df

    def CLL(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def ESS(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def print_summary(self, n: int = 5):
        """Print a summary of the ABC result."""
        print(f"Method: {self.method}")
        print(f"Number of chains: {self.n_chains}")
        print(f"ABC iterations (Nabc): {self.Nabc}")
        print(f"Tolerance (epsilon): {self.epsilon}")
        if np.asarray(self.accepts).size > 0:
            rates = self.acceptance_rate
            for c in range(int(np.asarray(self.accepts).size)):
                print(
                    f"  chain {c}: accepts={int(self.accepts[c])}, "
                    f"rate={float(rates[c]):.3f}"
                )
        print(f"Execution time: {self.execution_time} seconds")
        if (
            self.traces_da.size > 0
            and "distance" in list(self.traces_da.coords["variable"].values)
        ):
            last = self.traces_da.isel(iteration=-1).sel(variable="distance").values
            print(f"\nFinal distance per chain: {np.asarray(last)}")

    @staticmethod
    def merge(*results: "PompABCResult") -> "PompABCResult":
        """Concatenate per-chain traces along ``theta_idx``.

        All inputs must share the same ``Nabc``, ``epsilon`` and ``variable``
        coordinate.
        """
        if len(results) == 0:
            raise ValueError("At least one PompABCResult must be provided.")
        first = results[0]
        for r in results:
            if not isinstance(r, type(first)):
                raise TypeError("All results must be PompABCResult.")
            if r.Nabc != first.Nabc:
                raise ValueError("All results must have the same Nabc.")
            if r.epsilon != first.epsilon:
                raise ValueError("All results must have the same epsilon.")
            if list(r.traces_da.coords["variable"].values) != list(
                first.traces_da.coords["variable"].values
            ):
                raise ValueError(
                    "All results must have the same variable coord ordering."
                )

        merged_da = xr.concat(
            [r.traces_da for r in results], dim="theta_idx"
        ).assign_coords(theta_idx=np.arange(sum(r.n_chains for r in results)))

        merged_theta = sum((r.theta for r in results), [])
        merged_accepts = np.concatenate(
            [np.asarray(r.accepts).ravel() for r in results]
        )
        execution_times = [
            r.execution_time for r in results if r.execution_time is not None
        ]
        max_time = max(execution_times) if execution_times else None

        return PompABCResult(
            method=first.method,
            execution_time=max_time,
            key=first.key,
            theta=merged_theta,
            traces_da=merged_da,
            Nabc=first.Nabc,
            epsilon=first.epsilon,
            accepts=merged_accepts,
        )



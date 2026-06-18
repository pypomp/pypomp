from dataclasses import dataclass, field
import pandas as pd
import xarray as xr
import numpy as np

from .base import BaseResult, PompEstimationTracesMixin, _merge_results
from ...maths import logmeanexp, logmeanexp_se
from ..rw_sigma import RWSigma
from ..learning_rate import LearningRate
from ..optimizer import Optimizer, Adam
from ..parameters import PompParameters


@dataclass(eq=False)
class PompBaseResult(BaseResult):
    """Base class for Pomp results."""

    theta: PompParameters | None = None
    """The parameter object used for the computation."""


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
        logLik = np.atleast_1d(logmeanexp(arr, axis=-1, ignore_nan=ignore_nan))
        se = (
            logmeanexp_se(arr, axis=-1, ignore_nan=ignore_nan)
            if arr.shape[-1] > 1
            else np.full_like(logLik, np.nan)
        )
        se = np.atleast_1d(se)
        theta_df = pd.DataFrame(self.theta.params())
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
        arr = np.asarray(getattr(self.logLiks, "values", self.logLiks))
        logliks = np.atleast_1d(logmeanexp(arr, axis=-1))
        se = (
            logmeanexp_se(arr, axis=-1)
            if arr.shape[-1] > 1
            else np.full_like(logliks, np.nan)
        )
        se = np.atleast_1d(se)
        base_df = pd.DataFrame(
            {
                "theta_idx": np.arange(len(self.theta)),
                "iteration": 0,
                "method": self.method,
                "logLik": logliks,
                "se": se,
            }
        )
        if not self.theta:
            return base_df
        return pd.concat([base_df, pd.DataFrame(self.theta.params())], axis=1)

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
            ("Resampling threshold", "thresh"),
            ("Number of monitors", "n_monitors"),
        ]

    @staticmethod
    def merge(*results: "PompMIFResult") -> "PompMIFResult":
        return _merge_results(
            PompMIFResult,
            results,
            ["J", "M", "thresh", "n_monitors", "rw_sd", "method"],
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


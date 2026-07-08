from dataclasses import dataclass, field
import pandas as pd
import xarray as xr
import numpy as np

from .base import BaseResult, PompEstimationTracesMixin
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
    """Log-likelihoods for each parameter set and replicate.
    Dimensions: ("theta_idx", "rep")
    """
    J: int = 0
    """The number of particles used for filtering."""
    reps: int = 1
    """The number of replicates for each parameter set."""
    thresh: float = 0.0
    """The resampling threshold used by the filter."""
    CLL_da: xr.DataArray | None = None
    """Conditional log-likelihoods for each unit and time point.
    Dimensions: ("theta_idx", "rep", "time")
    """
    ESS_da: xr.DataArray | None = None
    """Effective Sample Size for each unit and time point.
    Dimensions: ("theta_idx", "rep", "time")
    """
    filter_mean: xr.DataArray | None = None
    """The mean of the filtering distribution for each state variable.
    Dimensions: ("theta_idx", "rep", "state", "time")
    """
    prediction_mean: xr.DataArray | None = None
    """The mean of the predictive distribution for each state variable.
    Dimensions: ("theta_idx", "rep", "state", "time")
    """

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

    def traces(self) -> pd.DataFrame:
        """Return traces DataFrame for this pfilter result."""
        df = self.to_dataframe()
        if df.empty:
            return df
        df.insert(1, "iteration", 0)
        df.insert(2, "method", self.method)
        cols = ["theta_idx", "iteration", "method", "logLik", "se"]
        other_cols = [c for c in df.columns if c not in cols]
        return df[cols + other_cols]


@dataclass(eq=False)
class PompMIFResult(PompEstimationTracesMixin, PompBaseResult):
    """Result from Pomp.mif() method."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihood and parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "variable")
    """
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


@dataclass(eq=False)
class PompTrainResult(PompEstimationTracesMixin, PompBaseResult):
    """Result from Pomp.train() method."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihood and parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "variable")
    """
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
    thresh: float = 0.0
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

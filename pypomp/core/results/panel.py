from dataclasses import dataclass, field
import pandas as pd
import xarray as xr
import numpy as np
import warnings

from .base import BaseResult, PanelPompEstimationTracesMixin
from ...maths import logmeanexp, logmeanexp_se
from ..rw_sigma import RWSigma
from ..learning_rate import LearningRate
from ..parameters import PanelParameters
from ..optimizer import Optimizer, Adam


def _attach_panel_params(
    theta: PanelParameters | None, df_s: pd.DataFrame | None, df_u: pd.DataFrame
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Helper to join theta values to df_s and df_u."""
    if theta is None or theta.num_replicates() == 0:
        return df_s, df_u

    shared_names = theta.get_shared_param_names()
    if shared_names and "shared" in theta._data:
        s_vals = theta._data["shared"].sel(parameter=shared_names).values
        p_s = pd.DataFrame(s_vals, columns=shared_names)
        if df_s is not None:
            df_s = df_s.join(p_s, on="theta_idx")
        df_u = df_u.join(p_s, on="theta_idx")

    specific_names = theta.get_unit_param_names()
    if specific_names and "unit_specific" in theta._data:
        p_u = (
            theta._data["unit_specific"]
            .sel(parameter=specific_names)
            .to_dataset(dim="parameter")
            .to_dataframe()
            .reset_index()
        )
        df_u = df_u.merge(p_u, on=["theta_idx", "unit"], how="left")

    return df_s, df_u


@dataclass(eq=False)
class PanelPompBaseResult(BaseResult):
    """Base class for PanelPomp results."""

    theta: PanelParameters | None = None
    """The panel parameter object used for the computation."""


@dataclass(eq=False)
class PanelPompPFilterResult(PanelPompBaseResult):
    """Result from PanelPomp.pfilter() method."""

    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each parameter set, replicate, and unit.
    Dimensions: ("theta_idx", "unit", "rep")
    """
    J: int = 0
    """The number of particles used for filtering."""
    reps: int = 1
    """The number of replicates for each parameter set."""
    thresh: float = 0.0
    """The resampling threshold used."""
    CLL_da: xr.DataArray | None = None
    """Conditional log-likelihoods for each unit and time point.
    Dimensions: ("theta_idx", "unit", "rep", "time")
    """
    ESS_da: xr.DataArray | None = None
    """Effective Sample Size for each unit and time point.
    Dimensions: ("theta_idx", "unit", "rep", "time")
    """
    filter_mean: xr.DataArray | None = None
    """The mean of the filtering distribution for each state variable.
    Dimensions: ("theta_idx", "unit", "rep", "state", "time")
    """
    prediction_mean: xr.DataArray | None = None
    """The mean of the predictive distribution for each state variable.
    Dimensions: ("theta_idx", "unit", "rep", "state", "time")
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

        _, df = _attach_panel_params(self.theta, None, df)
        return df

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

        df_s, df_u = _attach_panel_params(self.theta, df_s, df_u)
        dfs_to_concat = [df for df in [df_s, df_u] if not df.empty]
        if not dfs_to_concat:
            return pd.DataFrame()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = pd.concat(dfs_to_concat, ignore_index=True)
        df = df.assign(method="pfilter", iteration=0)
        cols = ["theta_idx", "unit", "iteration", "method", "logLik", "se"]
        other_cols = [c for c in df.columns if c not in cols]
        return df.loc[:, cols + other_cols].copy()


@dataclass(eq=False)
class PanelPompMIFResult(PanelPompEstimationTracesMixin, PanelPompBaseResult):
    """Result from PanelPomp.mif() method."""

    shared_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Shared parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "variable")
    """
    unit_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Unit-specific parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "unit", "variable")
    """
    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each unit across iterations.
    Dimensions: ("theta_idx", "iteration", "unit")
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


@dataclass(eq=False)
class PanelPompTrainResult(PanelPompEstimationTracesMixin, PanelPompBaseResult):
    """Result from PanelPomp.train() method."""

    shared_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Shared parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "variable")
    """
    unit_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Unit-specific parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "unit", "variable")
    """
    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each unit across iterations.
    Dimensions: ("theta_idx", "iteration", "unit")
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


@dataclass(eq=False)
class PanelPompDpopTrainResult(PanelPompEstimationTracesMixin, PanelPompBaseResult):
    """Result from PanelPomp.dpop_train() method."""

    shared_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Shared parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "variable")
    """
    unit_traces: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Unit-specific parameter traces across iterations.
    Dimensions: ("theta_idx", "iteration", "unit", "variable")
    """
    logLiks: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Log-likelihoods for each unit across iterations.
    Dimensions: ("theta_idx", "iteration", "unit")
    """
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

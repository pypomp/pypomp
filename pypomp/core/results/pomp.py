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
        """Convert results to a pandas DataFrame.

        Parameters
        ----------
        ignore_nan : bool, optional
            Whether to ignore rows containing NaN when computing log-likelihoods and standard errors.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame representation of the results. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``logLik``: The estimated log-likelihood.
            3. ``se``: The standard error of the log-likelihood estimate.
            4. Parameter columns: One column per model parameter in their defined order.
        """
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
        theta_df = pd.DataFrame(self.theta.params(as_list=True))
        df = pd.DataFrame(
            {"theta_idx": np.arange(len(theta_df)), "logLik": logLik, "se": se}
        )
        return pd.concat([df, theta_df], axis=1)

    def traces(self) -> pd.DataFrame:
        """Return parameter and likelihood trace history.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of the traces. The columns appear in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``iteration``: The iteration index (fixed at 0 for ``pfilter``).
            3. ``method``: The name of the method (``'pfilter'``).
            4. ``logLik``: The estimated log-likelihood.
            5. ``se``: The standard error of the log-likelihood estimate.
            6. Parameter columns: One column per model parameter in their defined order.
        """
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


@dataclass(eq=False)
class PompPMCMCResult(PompBaseResult):
    """Result from :meth:`pypomp.core.pomp.Pomp.pmcmc`."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Per-chain trace array with dims ``("theta_idx", "iteration", "variable")``."""
    Nmcmc: int = 0
    """Number of MCMC iterations per chain."""
    J: int = 0
    """Number of particles per particle-filter likelihood evaluation."""
    accepts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    """Per-chain accepted proposal counts."""

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

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert the full PMCMC trace to a tidy DataFrame."""
        if self.traces_da.size == 0:
            return pd.DataFrame()
        df = self.traces_da.to_dataset(dim="variable").to_dataframe().reset_index()
        var_order = list(self.traces_da.coords["variable"].values)
        cols = ["theta_idx", "iteration"] + [c for c in var_order if c in df.columns]
        df = df[cols]
        if ignore_nan:
            df = df.dropna()
        return df

    def traces(self) -> pd.DataFrame:
        """Return a trace DataFrame compatible with :class:`ResultsHistory`."""
        df = self.to_dataframe()
        if df.empty:
            return df
        df.insert(2, "method", self.method)
        df.insert(4, "se", np.nan)
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
            for chain_idx in range(int(np.asarray(self.accepts).size)):
                print(
                    f"  chain {chain_idx}: accepts={int(self.accepts[chain_idx])}, "
                    f"rate={float(rates[chain_idx]):.3f}"
                )
        print(f"Execution time: {self.execution_time} seconds")
        if self.traces_da.size > 0 and "logLik" in list(
            self.traces_da.coords["variable"].values
        ):
            last = self.traces_da.isel(iteration=-1).sel(variable="logLik").values
            print(f"\nFinal logLik per chain: {np.asarray(last)}")

    @classmethod
    def merge(cls, *results: BaseResult) -> "PompPMCMCResult":
        """Concatenate PMCMC chains from multiple results."""
        if len(results) == 0:
            raise ValueError("At least one PompPMCMCResult must be provided.")
        pmcmc_results: list[PompPMCMCResult] = []
        for r in results:
            if not isinstance(r, PompPMCMCResult):
                raise TypeError("All results must be PompPMCMCResult.")
            pmcmc_results.append(r)

        first = pmcmc_results[0]
        for result in pmcmc_results:
            if result.J != first.J:
                raise ValueError("All results must have the same J.")
            if result.Nmcmc != first.Nmcmc:
                raise ValueError("All results must have the same Nmcmc.")
            if list(result.traces_da.coords["variable"].values) != list(
                first.traces_da.coords["variable"].values
            ):
                raise ValueError("All results must have the same variable ordering.")

        merged_da = xr.concat(
            [result.traces_da for result in pmcmc_results], dim="theta_idx"
        ).assign_coords(theta_idx=np.arange(sum(r.n_chains for r in pmcmc_results)))
        theta_objs = [
            result.theta for result in pmcmc_results if result.theta is not None
        ]
        merged_theta = PompParameters.merge(*theta_objs) if theta_objs else None
        merged_accepts = np.concatenate(
            [np.asarray(result.accepts).ravel() for result in pmcmc_results]
        )
        execution_times = [
            result.execution_time
            for result in pmcmc_results
            if result.execution_time is not None
        ]

        return PompPMCMCResult(
            method=first.method,
            execution_time=max(execution_times) if execution_times else None,
            key=first.key,
            theta=merged_theta,
            traces_da=merged_da,
            Nmcmc=first.Nmcmc,
            J=first.J,
            accepts=merged_accepts,
        )


@dataclass(eq=False)
class PompABCResult(PompBaseResult):
    """Result from :meth:`pypomp.core.pomp.Pomp.abc`."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Per-chain trace array with dims ``("theta_idx", "iteration", "variable")``."""
    Nabc: int = 0
    """Number of ABC-MCMC iterations per chain."""
    epsilon: float = 0.0
    """ABC distance threshold."""
    accepts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    """Per-chain accepted proposal counts."""

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

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert the full ABC-MCMC trace to a tidy DataFrame."""
        if self.traces_da.size == 0:
            return pd.DataFrame()
        df = self.traces_da.to_dataset(dim="variable").to_dataframe().reset_index()
        var_order = list(self.traces_da.coords["variable"].values)
        cols = ["theta_idx", "iteration"] + [c for c in var_order if c in df.columns]
        df = df[cols]
        if ignore_nan:
            df = df.dropna()
        return df

    def traces(self) -> pd.DataFrame:
        """Return a trace DataFrame compatible with :class:`ResultsHistory`."""
        df = self.to_dataframe()
        if df.empty:
            return df
        df.insert(2, "method", self.method)
        df.insert(3, "logLik", np.nan)
        df.insert(4, "se", np.nan)
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
            for chain_idx in range(int(np.asarray(self.accepts).size)):
                print(
                    f"  chain {chain_idx}: accepts={int(self.accepts[chain_idx])}, "
                    f"rate={float(rates[chain_idx]):.3f}"
                )
        print(f"Execution time: {self.execution_time} seconds")
        if self.traces_da.size > 0 and "distance" in list(
            self.traces_da.coords["variable"].values
        ):
            last = self.traces_da.isel(iteration=-1).sel(variable="distance").values
            print(f"\nFinal distance per chain: {np.asarray(last)}")

    @classmethod
    def merge(cls, *results: BaseResult) -> "PompABCResult":
        """Concatenate ABC-MCMC chains from multiple results."""
        if len(results) == 0:
            raise ValueError("At least one PompABCResult must be provided.")
        abc_results: list[PompABCResult] = []
        for r in results:
            if not isinstance(r, PompABCResult):
                raise TypeError("All results must be PompABCResult.")
            abc_results.append(r)

        first = abc_results[0]
        for result in abc_results:
            if result.Nabc != first.Nabc:
                raise ValueError("All results must have the same Nabc.")
            if result.epsilon != first.epsilon:
                raise ValueError("All results must have the same epsilon.")
            if list(result.traces_da.coords["variable"].values) != list(
                first.traces_da.coords["variable"].values
            ):
                raise ValueError("All results must have the same variable ordering.")

        merged_da = xr.concat(
            [result.traces_da for result in abc_results], dim="theta_idx"
        ).assign_coords(theta_idx=np.arange(sum(r.n_chains for r in abc_results)))
        theta_objs = [
            result.theta for result in abc_results if result.theta is not None
        ]
        merged_theta = PompParameters.merge(*theta_objs) if theta_objs else None
        merged_accepts = np.concatenate(
            [np.asarray(result.accepts).ravel() for result in abc_results]
        )
        execution_times = [
            result.execution_time
            for result in abc_results
            if result.execution_time is not None
        ]

        return PompABCResult(
            method=first.method,
            execution_time=max(execution_times) if execution_times else None,
            key=first.key,
            theta=merged_theta,
            traces_da=merged_da,
            Nabc=first.Nabc,
            epsilon=first.epsilon,
            accepts=merged_accepts,
        )

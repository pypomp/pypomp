from dataclasses import dataclass, field
import pandas as pd
import xarray as xr
import numpy as np
from typing import cast

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


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cdf = np.cumsum(weights_sorted)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, values_sorted)


@dataclass(eq=False)
class PompBIFResult(PompBaseResult):
    """Result from Pomp.bif()."""

    traces_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Stage 1 parameter traces and log-likelihoods."""
    cloud_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Final Stage 1 cloud on the natural parameter scale."""
    cloud_est_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Final Stage 1 cloud on the estimation parameter scale."""
    posterior_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Flattened weighted cloud on the natural parameter scale."""
    weights_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Stage 2 normalized deconvolution weights."""
    log_Hf_da: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
    """Leave-one-out log-smoothed-cloud density estimates."""
    J: int = 0
    """The number of particles used in Stage 1."""
    M: int = 0
    """The number of Stage 1 iterations."""
    perturb_sd: RWSigma | None = None
    """Fixed initial-perturbation standard deviations defining Q."""
    rw_sd: RWSigma | None = None
    """Within-trajectory random-walk standard deviations."""
    a: float = 0.0
    """Cooling fraction for the within-trajectory random walk."""
    thresh: float = 0.0
    """The resampling threshold used."""
    n_monitors: int = 0
    """The number of particle filters used to monitor likelihood."""
    active_params: list[str] = field(default_factory=list)
    """Parameters with positive fixed perturbation standard deviation."""
    ess: float = 0.0
    """Effective sample size of the Stage 2 deconvolution weights."""

    def __post_init__(self):
        self.method = "bif"

    @property
    def n_samples(self) -> int:
        if self.weights_da.size == 0:
            return 0
        return int(self.weights_da.sizes.get("sample", 0))

    @property
    def _summary_config(self) -> list[tuple[str, str]]:
        return [
            ("Number of parameter sets", "theta"),
            ("Number of particles (J)", "J"),
            ("Number of iterations (M)", "M"),
            ("Number of weighted samples", "n_samples"),
            ("Deconvolution ESS", "ess"),
            ("Active parameters", "active_params"),
        ]

    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Return weighted posterior samples on the natural parameter scale."""
        if self.posterior_da.size == 0 or self.weights_da.size == 0:
            return pd.DataFrame()

        variables = cast(list[str], list(self.posterior_da.coords["variable"].values))
        df = pd.DataFrame(
            np.asarray(self.posterior_da.values),
            columns=pd.Index(variables),
        )
        df.insert(0, "sample", np.arange(len(df)))

        if "theta_idx" in self.weights_da.coords:
            df.insert(1, "theta_idx", np.asarray(self.weights_da.coords["theta_idx"]))
        if "particle" in self.weights_da.coords:
            df.insert(2, "particle", np.asarray(self.weights_da.coords["particle"]))

        df["weight"] = np.asarray(self.weights_da.values)
        if self.log_Hf_da.size > 0:
            df["log_Hf"] = np.asarray(self.log_Hf_da.values)

        if ignore_nan:
            df = df.dropna()
        return cast(pd.DataFrame, df)

    def weighted_summary(
        self,
        quantiles: tuple[float, ...] = (0.025, 0.5, 0.975),
    ) -> pd.DataFrame:
        """Return weighted means, standard deviations, and quantiles."""
        if self.posterior_da.size == 0 or self.weights_da.size == 0:
            return pd.DataFrame()

        values = np.asarray(self.posterior_da.values)
        weights = np.asarray(self.weights_da.values, dtype=float)
        weights = weights / np.sum(weights)
        variables = cast(list[str], list(self.posterior_da.coords["variable"].values))
        qs = np.asarray(quantiles, dtype=float)

        rows = []
        for i, name in enumerate(variables):
            vals = values[:, i]
            mean = float(np.sum(weights * vals))
            sd = float(np.sqrt(np.sum(weights * (vals - mean) ** 2)))
            qvals = _weighted_quantile(vals, weights, qs)
            row = {"parameter": name, "mean": mean, "sd": sd}
            row.update({f"q{q:g}": float(v) for q, v in zip(qs, qvals)})
            rows.append(row)
        return pd.DataFrame(rows)

    def traces(self) -> pd.DataFrame:
        """Return Stage 1 traces as a tidy DataFrame."""
        if self.traces_da.size == 0:
            return pd.DataFrame()
        df = (
            self.traces_da.to_dataset(dim="variable")
            .to_dataframe()
            .reset_index()
            .assign(method=self.method)
        )
        cols = ["theta_idx", "iteration", "method", "logLik"]
        other_cols = [c for c in df.columns if c not in cols]
        return cast(pd.DataFrame, df[cols + other_cols])

    def CLL(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame()

    def ESS(self, average: bool = False) -> pd.DataFrame:
        return pd.DataFrame({"ess": [self.ess], "n_samples": [self.n_samples]})

    def print_summary(self, n: int = 5):
        print(f"Method: {self.method}")
        print(f"Number of parameter sets: {len(self.theta)}")
        print(f"Number of particles (J): {self.J}")
        print(f"Number of iterations (M): {self.M}")
        print(f"Number of weighted samples: {self.n_samples}")
        print(f"Deconvolution ESS: {self.ess}")
        print(f"Active parameters: {self.active_params}")
        print(f"Execution time: {self.execution_time} seconds")
        summary = self.weighted_summary()
        if not summary.empty:
            print(f"\nWeighted summary (first {n} rows):")
            print(summary.head(n).to_string(index=False))

    @staticmethod
    def merge(*results: "PompBIFResult") -> "PompBIFResult":
        raise NotImplementedError("Merging PompBIFResult is not yet implemented.")


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
        return cast(pd.DataFrame, df)

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

    @staticmethod
    def merge(*results: "PompPMCMCResult") -> "PompPMCMCResult":
        """Concatenate PMCMC chains from multiple results."""
        if len(results) == 0:
            raise ValueError("At least one PompPMCMCResult must be provided.")
        first = results[0]
        for result in results:
            if not isinstance(result, PompPMCMCResult):
                raise TypeError("All results must be PompPMCMCResult.")
            if result.J != first.J:
                raise ValueError("All results must have the same J.")
            if result.Nmcmc != first.Nmcmc:
                raise ValueError("All results must have the same Nmcmc.")
            if list(result.traces_da.coords["variable"].values) != list(
                first.traces_da.coords["variable"].values
            ):
                raise ValueError("All results must have the same variable ordering.")

        merged_da = xr.concat(
            [result.traces_da for result in results], dim="theta_idx"
        ).assign_coords(theta_idx=np.arange(sum(r.n_chains for r in results)))
        theta_objs = [result.theta for result in results if result.theta is not None]
        merged_theta = PompParameters.merge(*theta_objs) if theta_objs else None
        merged_accepts = np.concatenate(
            [np.asarray(result.accepts).ravel() for result in results]
        )
        execution_times = [
            result.execution_time
            for result in results
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
        return cast(pd.DataFrame, df)

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

    @staticmethod
    def merge(*results: "PompABCResult") -> "PompABCResult":
        """Concatenate ABC-MCMC chains from multiple results."""
        if len(results) == 0:
            raise ValueError("At least one PompABCResult must be provided.")
        first = results[0]
        for result in results:
            if not isinstance(result, PompABCResult):
                raise TypeError("All results must be PompABCResult.")
            if result.Nabc != first.Nabc:
                raise ValueError("All results must have the same Nabc.")
            if result.epsilon != first.epsilon:
                raise ValueError("All results must have the same epsilon.")
            if list(result.traces_da.coords["variable"].values) != list(
                first.traces_da.coords["variable"].values
            ):
                raise ValueError("All results must have the same variable ordering.")

        merged_da = xr.concat(
            [result.traces_da for result in results], dim="theta_idx"
        ).assign_coords(theta_idx=np.arange(sum(r.n_chains for r in results)))
        theta_objs = [result.theta for result in results if result.theta is not None]
        merged_theta = PompParameters.merge(*theta_objs) if theta_objs else None
        merged_accepts = np.concatenate(
            [np.asarray(result.accepts).ravel() for result in results]
        )
        execution_times = [
            result.execution_time
            for result in results
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

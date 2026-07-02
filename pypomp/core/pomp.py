"""
This module implements the OOP structure for POMP models.
"""

import importlib
import cloudpickle
from copy import deepcopy
import time
from typing import Callable, Any, cast
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import warnings
from typing import Union, overload, Literal
from .viz import plot_traces_internal, plot_simulations_internal

from pypomp.types import ParamDict
from .metadata import ModelMetadata
from pypomp import functional as F
from .model_struct import _RInit, _RProc, _DMeas, _RMeas
import xarray as xr
from .algorithms.helpers import _calc_ys_covars, run_jax_batch_sharded
from .algorithms.bif import _bif_deconvolution_diag
from .rw_sigma import RWSigma
from .learning_rate import LearningRate
from .par_trans import ParTrans
from .optimizer import Optimizer, Adam
from .results import (
    ResultsHistory,
    PompPFilterResult,
    PompMIFResult,
    PompBIFResult,
    PompTrainResult,
    PompPMCMCResult,
    PompABCResult,
)
from .parameters import PompParameters
from pypomp.maths import logmeanexp
from pypomp import benchmarks
from pypomp.functional.structs import PompStruct
from pypomp.proposals import _expand_proposal


def _flat_dprior(theta_arr: jax.Array) -> jax.Array:
    """Default flat improper log-prior."""
    return jnp.zeros((), dtype=theta_arr.dtype)


class Pomp:
    """
    A class representing a Partially Observed Markov Process (POMP) model.

    This class provides a structured way to define and work with POMP models, which are
    used for modeling time series data where the underlying state process is only
    partially observed. The class encapsulates the model components including the
    initial state distribution, process model, and measurement model.

    In particular, the class provides methods for:

    - Simulation of the model

    - Particle filtering

    - Iterated filtering

    - Model training using a differentiable particle filter


    **⚠️ IMPORTANT: Defining Model Components**

    The `rinit`, `rproc`, `dmeas`, and `rmeas` arguments expect user-defined
    functions. **You MUST read the documentation for each argument to understand the required argument names, type hints, and return types.** The `Pomp` object will fail to initialize if these functions do not strictly
    adhere to the specifications.

    - **State initialization simulator (rinit):** See :ref:`rinit-tutorial`.
    - **State transition simulator (rproc):** See :ref:`rproc-tutorial`.
    - **Measurement density (dmeas):** See :ref:`dmeas-tutorial`.
    - **Measurement simulator (rmeas):** See :ref:`rmeas-tutorial`.

    Parameters
    ----------
    ys : pd.DataFrame
        The measurement data frame. The row index must contain the observation times.
    theta : PompParameters
        Initial parameter(s) for the model. Accepts:
        - An existing :class:`~pypomp.core.parameters.PompParameters` object
        Vectorized methods (like pfilter) will run in parallel over multiple parameter sets stored inside the `PompParameters` object.
    statenames : list[str]
        List of all latent state variable names.
    t0 : float
        The initial time for the model (typically before the first observation).
    rinit : Callable
        Initial state simulator function.
    rproc : Callable
        Process simulator function (defining a single time step).
    dmeas : Callable, optional
        Measurement density function (log-likelihood).
    rmeas : Callable, optional
        Measurement simulator function.
    par_trans : :class:`~pypomp.core.par_trans.ParTrans`, optional
        Parameter transformation object used to move parameters
        between the natural space and the estimation space. Defaults to the identity transformation.
    covars : pd.DataFrame, optional
        Time-varying covariates. The row index must contain the covariate times.
    nstep : int, optional
        The number of integration steps to take between observations.
        Passed automatically to the `RProc` component. Must be None if `dt` is provided.
    dt : float, optional
        Fixed time step size for the process model.
        Passed automatically to the `RProc` component. Must be None if `nstep` is provided.
    accumvars : tuple[str, ...], optional
        Names of accumulator state variables (e.g., incidence tracking). These are reset to 0 at the start of each observation interval.
    validate_logic : bool, optional
        Whether to validate the logic of the model components.
    order : str, optional
        The interpolation order for time-varying covariates ("linear" or "constant").
    """

    ys: pd.DataFrame
    """The measurement data frame with observation times as the index."""

    _theta: PompParameters | None
    """Internal storage for model parameters in canonical order."""

    canonical_param_names: list[str]
    """Ordered list of parameter names used throughout the model."""

    statenames: list[str]
    """Names of all latent state variables in the process model."""

    t0: float
    """Initial time for the model (typically before the first observation)."""

    rinit: _RInit
    """Simulator for the initial state distribution."""

    rproc: _RProc
    """Process model simulator handling state transitions between observation times."""

    dmeas: _DMeas | None
    """Measurement density used to evaluate the likelihood of observations."""

    rmeas: _RMeas | None
    """Measurement simulator used to generate synthetic observations."""

    par_trans: ParTrans
    """Parameter transformation object mapping between natural and estimation spaces."""

    covars: pd.DataFrame | None
    """Time-varying covariates for the model, if applicable."""

    _covars_extended: np.ndarray | None
    """Internal covariate array interpolated/aligned to the integration grid."""

    _nstep_array: np.ndarray
    """Number of integration steps between successive observation times."""

    _dt_array_extended: np.ndarray
    """Time step sizes for each integration step over the full time grid."""

    _max_steps_per_interval: int
    """Maximum number of integration steps between any two observation times."""

    accumvars: list[str] | None
    """Names of accumulator state variables that are reset at each observation time."""

    _accumvars_indices: tuple[int, ...] | None
    """Indices of accumulator state variables within the full state vector."""

    results_history: ResultsHistory
    """A :class:`~pypomp.core.results.ResultsHistory` object storing the history of results from :meth:`pfilter`, :meth:`mif`, and :meth:`train` calls."""

    fresh_key: jax.Array | None
    """Running a method that accepts a JAX PRNG key will store a fresh, unused key here."""

    metadata: ModelMetadata
    """Environment and version metadata initialized when this instance was built."""

    def __init__(
        self,
        ys: pd.DataFrame,
        theta: PompParameters,
        statenames: tuple[str, ...] | list[str],
        t0: float,
        rinit: Callable,
        rproc: Callable,
        dmeas: Callable | None = None,
        rmeas: Callable | None = None,
        par_trans: ParTrans | None = None,
        nstep: int | None = None,
        dt: float | None = None,
        accumvars: tuple[str, ...] | list[str] | None = None,
        covars: pd.DataFrame | None = None,
        validate_logic: bool = True,
        order: str = "linear",
    ):
        if not isinstance(ys, pd.DataFrame):
            raise TypeError("ys must be a pandas DataFrame")
        if covars is not None and not isinstance(covars, pd.DataFrame):
            raise TypeError("covars must be a pandas DataFrame or None")

        if not isinstance(theta, PompParameters):
            raise TypeError("theta must be a PompParameters instance")
        self._theta = theta

        # Extract parameter names from first theta dict
        self.canonical_param_names = self._theta.get_param_names()

        # If statenames not provided, we need to infer them
        if statenames is None:
            raise ValueError(
                "statenames must be provided as a list of state variable names"
            )

        if not isinstance(statenames, list) or not all(
            isinstance(name, str) for name in statenames
        ):
            raise ValueError("statenames must be a tuple or list of strings")

        if accumvars is not None:
            if not all(isinstance(name, str) for name in accumvars):
                raise ValueError("accumvars must be a tuple or list of strings")
            if not all(name in statenames for name in accumvars):
                raise ValueError("all accumvars must be in statenames")
            self._accumvars_indices = tuple(
                tuple(statenames).index(name) for name in accumvars
            )
        else:
            self._accumvars_indices = None

        self.statenames = list(statenames)
        self.accumvars = list(accumvars) if accumvars is not None else None
        self.ys = ys
        self.covars = covars
        self.t0 = float(t0)
        self.results_history = ResultsHistory()
        self.fresh_key = None
        self.metadata = ModelMetadata()

        if covars is not None:
            self.covar_names = list(covars.columns)
        else:
            self.covar_names = []

        self.par_trans = par_trans or ParTrans()
        self.rinit = _RInit(
            struct=rinit,
            statenames=self.statenames,
            param_names=self.canonical_param_names,
            covar_names=self.covar_names,
            par_trans=self.par_trans,
            validate_logic=validate_logic,
        )

        if dmeas is not None:
            self.dmeas = _DMeas(
                struct=dmeas,
                statenames=self.statenames,
                param_names=self.canonical_param_names,
                covar_names=self.covar_names,
                par_trans=self.par_trans,
                y_names=list(self.ys.columns),
                validate_logic=validate_logic,
            )
        else:
            self.dmeas = None

        if rmeas is not None:
            self.rmeas = _RMeas(
                struct=rmeas,
                statenames=self.statenames,
                param_names=self.canonical_param_names,
                covar_names=self.covar_names,
                par_trans=self.par_trans,
                y_names=list(self.ys.columns),
                validate_logic=validate_logic,
            )
        else:
            self.rmeas = None

        if self.dmeas is None and self.rmeas is None:
            raise ValueError("You must supply at least one of dmeas or rmeas")

        (
            self._covars_extended,
            self._dt_array_extended,
            self._nstep_array,
            self._max_steps_per_interval,
        ) = _calc_ys_covars(
            t0=self.t0,
            times=np.array(self.ys.index),
            ctimes=np.array(self.covars.index) if self.covars is not None else None,
            covars=np.array(self.covars) if self.covars is not None else None,
            dt=dt,
            nstep=nstep,
            order=order,
        )

        self.rproc = _RProc(
            struct=rproc,
            statenames=self.statenames,
            param_names=self.canonical_param_names,
            covar_names=self.covar_names,
            par_trans=self.par_trans,
            nstep=nstep,
            dt=dt,
            accumvars=self._accumvars_indices,
            validate_logic=validate_logic,
            nstep_array=self._nstep_array,
            max_steps_bound=self._max_steps_per_interval,
        )

    @property
    def theta(self) -> PompParameters:
        """The parameter object for the model."""
        if self._theta is None:
            raise ValueError("Model parameters have not been set (theta is None).")
        return self._theta

    @theta.setter
    def theta(self, value: PompParameters | None):
        if value is not None and not isinstance(value, PompParameters):
            raise TypeError("theta must be a PompParameters instance")
        self._theta = value

    def _prepare_theta_input(
        self,
        theta: PompParameters | None,
    ) -> PompParameters:
        """
        Prepare the theta input for the method.
        """
        if theta is None:
            return self.theta
        if not isinstance(theta, PompParameters):
            raise TypeError("theta must be a PompParameters object or None")
        if set(theta.get_param_names()) != set(self.canonical_param_names):
            raise ValueError(
                "theta parameter names must match canonical_param_names up to reordering"
            )
        return theta

    def _update_fresh_key(
        self, key: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """
        Updates the fresh_key attribute and returns a new key along with the old key.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple containing the new key and the old key.
                The old key is the key that was used to update the fresh_key attribute.
                The new key is the key that should be used for the next method call.
        """
        old_key = self.fresh_key if key is None else key
        if old_key is None:
            raise ValueError(
                "Both the key argument and the fresh_key attribute are None. At least one key must be given."
            )
        self.fresh_key, new_key = jax.random.split(old_key)
        return new_key, old_key

    def to_struct(self) -> PompStruct:
        """
        Exports the static data and compiled simulator functions into a lightweight
        JAX PyTree (PompStruct) for use with the functional API (pypomp.functional).

        Returns:
            PompStruct: The compiled structural representation of the model.
        """
        return PompStruct(
            ys=jnp.array(self.ys),
            dt_array_extended=jnp.array(self._dt_array_extended),
            nstep_array=jnp.array(self._nstep_array),
            t0=self.t0,
            times=jnp.array(self.ys.index),
            covars_extended=jnp.array(self._covars_extended)
            if self._covars_extended is not None
            else None,
            accumvars=self.rproc.accumvars,
            rinit_pf=self.rinit.struct_pf,
            rproc_pf=self.rproc.struct_pf_interp,
            dmeas_pf=self.dmeas.struct_pf if self.dmeas is not None else None,
            rinit_per=self.rinit.struct_per,
            rproc_per=self.rproc.struct_per_interp,
            dmeas_per=self.dmeas.struct_per if self.dmeas is not None else None,
            rmeas_pf=self.rmeas.struct_pf if self.rmeas is not None else None,
            par_trans=self.par_trans,
            param_names=self.canonical_param_names,
        )

    @staticmethod
    def sample_params(
        param_bounds: dict[str, tuple[float, float]], n: int, key: jax.Array
    ) -> PompParameters:
        """
        Samples multiple sets of parameters from independent uniform distributions.

        This utility method generates random parameter vectors within specified lower and
        upper bounds. It is commonly used to create initial parameter guesses or 'starting
        points' for global optimization.

        Args:
            param_bounds (dict): Dictionary mapping parameter names to (lower, upper) bounds
            n (int): Number of parameter sets to sample
            key (jax.Array): JAX random key for reproducibility

        Returns:
            PompParameters: A PompParameters object containing the sampled parameters
        """
        param_names = list(param_bounds.keys())
        low = jnp.array([param_bounds[p][0] for p in param_names])
        high = jnp.array([param_bounds[p][1] for p in param_names])

        sampled = jax.random.uniform(
            key, shape=(n, len(param_names)), minval=low, maxval=high
        )

        da = xr.DataArray(
            np.expand_dims(np.array(sampled), axis=1),
            dims=["theta_idx", "unit", "parameter"],
            coords={
                "theta_idx": np.arange(n),
                "unit": ["shared"],
                "parameter": param_names,
            },
        )
        return PompParameters(da)

    def print_metadata(self) -> None:
        """
        Displays technical metadata regarding the creation and runtime environment of this `Pomp` instance.

        This includes information such as the timestamp of creation, the versions of key
        dependencies, and other environment-specific details useful
        for reproducibility and debugging.
        """
        self.metadata.print_metadata()

    def pfilter(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0,
        reps: int = 1,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        track_time: bool = True,
    ) -> None:
        """
        Evaluates the likelihood of the model via the particle filter (bootstrap filter).

        The particle filter (also known as Sequential Monte Carlo) estimates the log-likelihood
        of the data given a specific set of parameters by propagating a swarm of particles
        through the latent state space. It can also be used to estimate the latent states
        over time (via filtering or prediction means).

        This implementation leverages JAX to efficiently vectorize the algorithm across
        multiple parameter sets simultaneously. Results are automatically stored in the
        model's history and can be accessed using :meth:`Pomp.results()`.

        Args:
            J (int): The number of particles
            key (jax.Array, optional): The random key. Defaults to self.fresh_key.
            theta (PompParameters, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Providing a :class:`~pypomp.core.parameters.PompParameters` object with multiple parameter sets enables faster, vectorized
                execution across all parameter sets.
            thresh (float, optional): Threshold value to determine whether to
                resample particles. Defaults to 0.
            reps (int, optional): Number of replicates to run. Defaults to 1.
            CLL (bool, optional): Boolean flag controlling whether to compute and store
                the conditional log-likelihoods at each time point.
            ESS (bool, optional): Boolean flag controlling whether to compute and store
                the effective sample size at each time point.
            filter_mean (bool, optional): Boolean flag controlling whether to compute
                and store the filtered mean at each time point.
            prediction_mean (bool, optional): Boolean flag controlling whether to
                compute and store the prediction mean at each time point.
            track_time (bool, optional): Boolean flag controlling whether to track the
                execution time.
        Returns:
            None. Updates :attr:`Pomp.results_history` with a :class:`~pypomp.core.results.PompPFilterResult` containing the log-likelihoods,
            and optionally the conditional log-likelihoods (CLL), effective sample size (ESS),
            filtered means, and prediction means if requested.
        """
        start_time = time.time()

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_for_results = deepcopy(theta_obj_in)
        new_key, old_key = self._update_fresh_key(key)
        n_theta_reps = theta_obj_in.num_replicates()

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        thetas_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        rep_keys = jax.random.split(new_key, n_theta_reps * reps).reshape(
            n_theta_reps, reps, *new_key.shape
        )

        results_jax = run_jax_batch_sharded(
            F.pfilter,
            {1: 0, 4: 0},
            {"logLik": 0, "CLL": 0, "ESS": 0, "filter_mean": 0, "prediction_mean": 0},
            self.to_struct(),
            thetas_array,
            J,
            thresh,
            rep_keys,
            CLL,
            ESS,
            filter_mean,
            prediction_mean,
        )

        results = jax.device_get(results_jax)

        del results_jax

        logLiks = results["logLik"]
        logLik_da = xr.DataArray(logLiks, dims=["theta_idx", "rep"])

        if track_time is True:
            execution_time = time.time() - start_time
        else:
            execution_time = None

        CLL_da = None
        ESS_da = None
        filter_mean_da = None
        prediction_mean_da = None

        if CLL and "CLL" in results:
            CLL_da = xr.DataArray(
                results["CLL"],
                dims=["theta_idx", "rep", "time"],
                coords={"time": self.ys.index},
            )

        if ESS and "ESS" in results:
            ESS_da = xr.DataArray(
                results["ESS"],
                dims=["theta_idx", "rep", "time"],
                coords={"time": self.ys.index},
            )

        if filter_mean and "filter_mean" in results:
            filter_mean_da = xr.DataArray(
                results["filter_mean"],
                dims=["theta_idx", "rep", "time", "state"],
                coords={"time": self.ys.index},
            )

        if prediction_mean and "prediction_mean" in results:
            prediction_mean_da = xr.DataArray(
                results["prediction_mean"],
                dims=["theta_idx", "rep", "time", "state"],
                coords={"time": self.ys.index},
            )

        del results

        logLik_estimates = logmeanexp(logLiks, axis=-1, ignore_nan=False)
        theta_obj_in.logLik = logLik_estimates
        self.theta = theta_obj_in

        result = PompPFilterResult(
            method="pfilter",
            execution_time=execution_time,
            key=old_key,
            theta=theta_for_results,
            logLiks=logLik_da,
            J=J,
            reps=reps,
            thresh=thresh,
            CLL_da=CLL_da,
            ESS_da=ESS_da,
            filter_mean=filter_mean_da,
            prediction_mean=prediction_mean_da,
        )

        self.results_history.add(result)

    def mif(
        self,
        J: int,
        M: int,
        rw_sd: RWSigma,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0,
        n_monitors: int = 0,
        track_time: bool = True,
    ) -> None:
        """
        Estimates model parameters by maximizing the marginal likelihood via the Iterated Filtering (IF2) algorithm.

        The Iterated Filtering algorithm estimates maximum likelihood parameters by
        introducing random perturbations to the parameters and sequentially filtering them
        alongside the state variables. Over successive iterations (cooling cycles), the
        perturbation variance is decayed, allowing the parameters to converge to their MLEs.

        This implementation leverages JAX to efficiently vectorize the algorithm across
        multiple initial parameter sets simultaneously. Results are automatically stored in
        the model's history and can be accessed using :meth:`Pomp.results()`.

        Args:
            J (int): The number of particles.
            M (int): Number of algorithm iterations.
            rw_sd (:class:`~pypomp.core.rw_sigma.RWSigma`): Random walk sigma object.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (PompParameters, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Providing a :class:`~pypomp.core.parameters.PompParameters` object with multiple parameter sets enables faster, vectorized
                execution across all parameter sets.
            thresh (float): Resampling threshold. Defaults to 0.
            n_monitors (int): Number of particle filter runs to average for
                log-likelihood estimation. Defaults to 0 (uses estimate from perturbed
                filter).
            track_time (bool): Boolean flag controlling whether to track the
                execution time.
        Returns:
            None. Updates :attr:`Pomp.results_history` with a :class:`~pypomp.core.results.PompMIFResult` containing the log-likelihoods,
            parameter traces, and diagnostic information from the Iterated Filtering (IF2) run.
        """
        start_time = time.time()

        rw_param_names = list(rw_sd.all_names)
        if set(rw_param_names) != set(self.canonical_param_names):
            raise ValueError(
                "rw_sd.sigmas keys must match canonical_param_names up to reordering. "
                f"Got {sorted(rw_param_names)}, expected {sorted(self.canonical_param_names)}."
            )

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)

        new_key, old_key = self._update_fresh_key(key)
        n_reps = theta_obj_in.num_replicates()
        sigmas_array, sigmas_init_array = rw_sd._return_arrays(
            param_names=self.canonical_param_names
        )
        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0.")

        keys = jax.random.split(new_key, n_reps)

        theta_array_3d = jnp.repeat(theta_array[:, jnp.newaxis, :], J, axis=1)

        nLLs_jax, theta_traces_jax, final_swarm_jax = run_jax_batch_sharded(
            F.mif,
            {1: 0, 8: 0},
            [0, 0, 0],
            self.to_struct(),
            theta_array_3d,
            sigmas_array,
            sigmas_init_array,
            M,
            rw_sd.cooling_fn,
            J,
            thresh,
            keys,
            n_monitors,
        )

        nLLs = jax.device_get(nLLs_jax)
        theta_traces = jax.device_get(theta_traces_jax)

        del nLLs_jax, theta_traces_jax, final_swarm_jax

        param_names = self.canonical_param_names
        trace_vars = ["logLik"] + param_names

        # Prepend nan for the log-likelihood of the initial parameters (at iteration 0)
        nans = np.full((n_reps, 1), np.nan)
        logliks_with_nan = np.concatenate([nans, -nLLs], axis=1)  # shape: (n_reps, M+1)

        trace_data = np.concatenate(
            [logliks_with_nan[:, :, np.newaxis], theta_traces], axis=-1
        )

        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_reps),
                "iteration": np.arange(M + 1),
                "variable": trace_vars,
            },
        )

        final_thetas_mean = theta_traces[:, M, :]  # shape: (n_reps, n_params)

        final_theta_da = xr.DataArray(
            final_thetas_mean,
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_reps),
                "parameter": param_names,
            },
        )
        self.theta = PompParameters(final_theta_da, logLik=-nLLs)

        if track_time is True:
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result = PompMIFResult(
            method="mif",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=traces_da,
            J=J,
            M=M,
            rw_sd=rw_sd,
            thresh=thresh,
            n_monitors=n_monitors,
        )

        self.results_history.add(result)

    def bif(
        self,
        J: int,
        M: int,
        perturb_sd: RWSigma,
        rw_sd: RWSigma | None = None,
        a: float = 0.1,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0,
        n_monitors: int = 0,
        track_time: bool = True,
    ) -> None:
        """
        Bayesian iterated filtering via fixed-kernel cloud deconvolution.

        BIF runs a perturbed filtering recursion in the estimation parameter
        scale, using a fixed initial perturbation kernel at the start of each
        outer iteration. The final parameter cloud is then reweighted by a
        leave-one-out Gaussian deconvolution estimate. Parameters with zero
        ``perturb_sd`` are treated as fixed and are excluded from the
        deconvolution kernel.

        Args:
            J: Number of particles in the parameter cloud.
            M: Number of Stage 1 outer iterations.
            perturb_sd: Fixed initial perturbation standard deviations. These
                define the diagonal deconvolution kernel ``Q`` in estimation
                scale. At least one entry must be positive.
            rw_sd: Optional within-trajectory random-walk standard deviations.
                These are geometrically cooled by ``a`` across Stage 1. If
                ``None``, no within-trajectory random walk is used.
            a: Geometric cooling fraction for the within-trajectory random
                walk. The fixed initial perturbation is not cooled.
            dprior: Pure-JAX log prior on the estimation-scale parameter
                vector. If ``None``, a flat improper prior is used.
            key: JAX PRNG key. Defaults to ``self.fresh_key``.
            theta: Starting parameter values. Multiple parameter sets run as
                independent Stage 1 starts and are pooled for deconvolution.
            thresh: Adaptive resampling threshold.
            n_monitors: Number of monitoring particle filters per iteration.
                Use 0 to store the perturbed-filter objective.
            track_time: Whether to store execution time.
        """
        start_time = time.time()

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J must be >= 1.")
        if M < 1:
            raise ValueError("M must be >= 1.")
        if not isinstance(perturb_sd, RWSigma):
            raise TypeError("perturb_sd must be a RWSigma object.")
        if rw_sd is not None and not isinstance(rw_sd, RWSigma):
            raise TypeError("rw_sd must be a RWSigma object or None.")

        param_names = self.canonical_param_names

        def sigma_array(obj: RWSigma, name: str) -> jax.Array:
            sigma_names = list(obj.all_names)
            if set(sigma_names) != set(param_names):
                raise ValueError(
                    f"{name}.sigmas keys must match canonical_param_names up to reordering. "
                    f"Got {sorted(sigma_names)}, expected {sorted(param_names)}."
                )
            return jnp.asarray([obj.sigmas[p] for p in param_names])

        perturb_sigmas_array = sigma_array(perturb_sd, "perturb_sd")
        if rw_sd is None:
            rw_sd = RWSigma({p: 0.0 for p in param_names})
        rw_sigmas_array = sigma_array(rw_sd, "rw_sd")

        perturb_sigmas_np = np.asarray(perturb_sigmas_array, dtype=float)
        active_idx = np.where(perturb_sigmas_np > 0)[0]
        if len(active_idx) == 0:
            raise ValueError("At least one perturb_sd entry must be positive.")
        active_params = [param_names[i] for i in active_idx]

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)
        n_reps = theta_obj_in.num_replicates()
        if J * n_reps <= 1:
            raise ValueError("BIF deconvolution requires at least two cloud samples.")

        new_key, old_key = self._update_fresh_key(key)
        theta_obj_est = deepcopy(theta_obj_in)
        theta_obj_est.transform(self.par_trans, direction="to_est")
        theta_array = theta_obj_est.to_jax_array(param_names)
        theta_tiled = jnp.tile(theta_array, (J, 1, 1))
        keys = jax.random.split(new_key, n_reps)

        _dprior: Callable = dprior if dprior is not None else _flat_dprior

        nLLs_jax, theta_traces_jax, final_thetas_jax = F.bif(
            self.to_struct(),
            theta_tiled,
            rw_sigmas_array,
            perturb_sigmas_array,
            M,
            a,
            J,
            thresh,
            keys,
            _dprior,
            n_monitors,
        )

        nLLs = jax.device_get(nLLs_jax)
        theta_traces = jax.device_get(theta_traces_jax)
        final_thetas_est = jax.device_get(final_thetas_jax)

        del nLLs_jax, theta_traces_jax, final_thetas_jax

        n_params = len(param_names)
        trace_vars = ["logLik"] + param_names
        trace_data = np.zeros((n_reps, M + 1, len(trace_vars)), dtype=float)
        for i in range(n_reps):
            logliks_with_nan = np.concatenate([np.array([np.nan]), -nLLs[i]])
            param_traces = self.par_trans._transform_array(
                theta_traces[i],
                param_names,
                direction="from_est",
            )
            trace_data[i, :, 0] = logliks_with_nan
            trace_data[i, :, 1:] = param_traces

        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_reps),
                "iteration": np.arange(M + 1),
                "variable": trace_vars,
            },
        )

        flat_cloud_est = final_thetas_est.reshape(n_reps * J, n_params)
        flat_cloud_nat = self.par_trans._transform_array(
            flat_cloud_est,
            param_names,
            direction="from_est",
        )
        final_thetas_nat = flat_cloud_nat.reshape(n_reps, J, n_params)

        cloud_active = flat_cloud_est[:, active_idx]
        sd_active = perturb_sigmas_np[active_idx]
        log_Hf_jax, weights_jax, ess_jax = _bif_deconvolution_diag(
            jnp.asarray(cloud_active),
            jnp.asarray(sd_active),
        )
        log_Hf = np.asarray(jax.device_get(log_Hf_jax), dtype=float)
        weights = np.asarray(jax.device_get(weights_jax), dtype=float)
        ess = float(jax.device_get(ess_jax))

        sample_idx = np.arange(n_reps * J)
        sample_theta_idx = np.repeat(np.arange(n_reps), J)
        sample_particle = np.tile(np.arange(J), n_reps)
        sample_coords = {
            "sample": sample_idx,
            "theta_idx": ("sample", sample_theta_idx),
            "particle": ("sample", sample_particle),
        }

        cloud_da = xr.DataArray(
            final_thetas_nat,
            dims=["theta_idx", "particle", "variable"],
            coords={
                "theta_idx": np.arange(n_reps),
                "particle": np.arange(J),
                "variable": param_names,
            },
        )
        cloud_est_da = xr.DataArray(
            final_thetas_est,
            dims=["theta_idx", "particle", "variable"],
            coords={
                "theta_idx": np.arange(n_reps),
                "particle": np.arange(J),
                "variable": param_names,
            },
        )
        posterior_da = xr.DataArray(
            flat_cloud_nat,
            dims=["sample", "variable"],
            coords={"sample": sample_idx, "variable": param_names},
        )
        weights_da = xr.DataArray(
            weights,
            dims=["sample"],
            coords=sample_coords,
        )
        log_Hf_da = xr.DataArray(
            log_Hf,
            dims=["sample"],
            coords=sample_coords,
        )

        posterior_mean = {
            p: float(np.sum(weights * flat_cloud_nat[:, i]))
            for i, p in enumerate(param_names)
        }
        self.theta = PompParameters(posterior_mean)

        if track_time is True:
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result = PompBIFResult(
            method="bif",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=traces_da,
            cloud_da=cloud_da,
            cloud_est_da=cloud_est_da,
            posterior_da=posterior_da,
            weights_da=weights_da,
            log_Hf_da=log_Hf_da,
            J=J,
            M=M,
            perturb_sd=perturb_sd,
            rw_sd=rw_sd,
            a=a,
            thresh=thresh,
            n_monitors=n_monitors,
            active_params=active_params,
            ess=ess,
        )

        self.results_history.add(result)

    def train(
        self,
        J: int,
        M: int,
        eta: LearningRate,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.97,
        thresh: int = 0,
        alpha_cooling: float = 1.0,
        n_monitors: int = 1,
        track_time: bool = True,
    ) -> None:
        """
        Optimizes parameters for a continuous-state model using a differentiable particle filter and gradient-based methods.

        This method performs Maximum Likelihood Estimation (MLE) using MOP, a differentiable particle filter for continuous-state POMPs. It computes gradients of the log-likelihood with respect to the parameters via reverse-mode automatic differentiation (using JAX), and updates the parameters using optimizers (e.g., Adam, SGD).

        It bears repeating that this optimizer is only valid for continuous-state POMPs! For discrete-state models, use :meth:`Pomp.mif()` or :meth:`Pomp.dpop_train()`.

        This implementation leverages JAX to efficiently vectorize the algorithm across
        multiple initial parameter sets simultaneously.
        Results are automatically stored in the model's history and can be accessed using :meth:`Pomp.results()`.

        Args:
            J (int): The number of particles in the MOP objective for obtaining the gradient and/or Hessian.
            M (int): Maximum iteration for the gradient descent optimization.
            eta (:class:`~pypomp.core.learning_rate.LearningRate`): Learning rates per parameter as a :class:`~pypomp.core.learning_rate.LearningRate` object.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (PompParameters, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Providing a :class:`~pypomp.core.parameters.PompParameters` object with multiple parameter sets enables faster, vectorized
                execution across all parameter sets.
            optimizer (:class:`~pypomp.core.optimizer.Optimizer`, optional): The optimizer configuration object to use
                (e.g., `pypomp.Adam()`, `pypomp.SGD()`, `pypomp.Newton()`, `pypomp.FullMatrixAdam()`, etc.).
                Defaults to `pypomp.Adam()`. Hyperparameters like learning rate scaling, line search
                (`scale`, `ls`, `c`, `max_ls_itn`), gradient clipping (`clip_norm`), or Adam beta values
                are configured directly inside the optimizer instance.
            alpha (float, optional): Discount factor for MOP.
            thresh (int, optional): Threshold value to determine whether to resample
                particles.
            alpha_cooling (float, optional): Cooling factor for the MOP discount factor (alpha) using cosine decay. This factor represents the multiplier for the distance of alpha from 1.0 by the end of training (i.e., alpha approaches 1.0). Defaults to 1.0 (no cooling).
            n_monitors (int, optional): Number of particle filter runs to average for
                log-likelihood estimation.
            track_time (bool, optional): Boolean flag controlling whether to track the
                execution time.

        Returns:
            None. Updates :attr:`Pomp.results_history` with a :class:`~pypomp.core.results.PompTrainResult` containing the log-likelihoods,
            parameter traces, and optimizer details from the training run.
        """
        start_time = time.time()

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)

        n_reps = theta_obj_in.num_replicates()

        theta_obj_in.transform(self.par_trans, direction="to_est")
        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0")

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

        # Convert eta to JAX array in canonical order
        eta_array = eta.to_array(self.canonical_param_names, M)

        new_key, old_key = self._update_fresh_key(key)
        keys = jnp.array(jax.random.split(new_key, n_reps))

        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        nLLs, theta_ests = run_jax_batch_sharded(
            F.train,
            {1: 0, 8: 0},
            [0, 0],
            self.to_struct(),
            theta_array,
            J,
            optimizer,
            M,
            eta_array,
            thresh,
            alpha,
            keys,
            alpha_cooling,
            n_monitors,
        )

        theta_ests_natural = self.par_trans._transform_array(
            np.asarray(theta_ests),
            self.canonical_param_names,
            direction="from_est",
        )

        joined_array = xr.DataArray(
            np.concatenate(
                [
                    -nLLs[..., np.newaxis],  # shape: (theta_idx, iteration, 1)
                    theta_ests_natural,  # shape: (theta_idx, iteration, n_theta)
                ],
                axis=-1,
            ),
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": range(0, n_reps),
                "iteration": range(0, M + 1),
                "variable": ["logLik"] + self.canonical_param_names,
            },
        )

        final_theta_da = xr.DataArray(
            theta_ests_natural[:, -1, :],
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_reps),
                "parameter": self.canonical_param_names,
            },
        )
        self.theta = PompParameters(final_theta_da, logLik=np.asarray(-nLLs))

        if track_time is True:
            nLLs.block_until_ready()
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result = PompTrainResult(
            method="train",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=joined_array,
            optimizer=optimizer,
            J=J,
            M=M,
            eta=eta,
            alpha=alpha,
            thresh=thresh,
            alpha_cooling=alpha_cooling,
        )

        self.results_history.add(result)

    def dpop_train(
        self,
        J: int,
        M: int,
        eta: LearningRate,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.8,
        alpha_cooling: float = 1.0,
        decay: float = 0.0,
        process_weight_state: str | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Optimizes model parameters using the DPOP differentiable particle filter and gradient-based methods.

        .. warning::
            This method is experimental. Its API and behavior are subject to change in future releases.

        This method trains the model parameters to maximize the DPOP objective function using
        first-order optimizers like Adam or SGD, with optional learning rate decay. Gradients
        are computed efficiently via JAX reverse-mode automatic differentiation.

        Parameters
        ----------
        J : int
            Number of particles.
        M : int
            Number of gradient steps.
        eta : LearningRate
            Per-parameter learning rates as a LearningRate object. A full
            per-iteration schedule is applied (row m used at iteration m), so
            ``LearningRate(rates).cosine_decay(0.05, M)`` works as expected.
        optimizer : Optimizer, default Adam()
            Optimizer configuration object, e.g. ``Adam()`` or ``SGD()``. Adam
            hyperparameters (``beta1``, ``beta2``, ``epsilon``) are read from the
            object; pass ``Adam(beta1=0.0)`` to disable momentum (e.g. for the
            high-variance alpha=0 arm, matching the dmop/IFAD convention).
        alpha : float, default 0.8
            DPOP discount / cooling factor.
        alpha_cooling : float, default 1.0
            Cosine cooling factor for alpha. This factor represents the
            multiplier for the distance of alpha from 1.0 by the end of
            training. The default keeps alpha fixed.
        decay : float, default 0.0
            Learning-rate decay coefficient. At iteration m, the effective
            learning rate is ``eta / (1 + decay * m)``.
        process_weight_state : str or None, default None
            Name of the state component that stores the accumulated
            process log-weight (e.g. ``"logw"``).
        key : jax.Array or None, default None
            Random key. If None, uses ``self.fresh_key``.
        theta : PompParameters, default None
            Optional initial parameter(s). Defaults to self.theta.

        Returns
        -------
        nll_history : jax.Array, shape (M+1,)
            Mean DPOP negative log-likelihood per observation at each step.
        theta_history : jax.Array, shape (M+1, p)
            Parameter vector (estimation space) at each step.
        """
        warnings.warn(
            "dpop_train is experimental and its API and behavior are subject to change.",
            category=FutureWarning,
            stacklevel=2,
        )

        from .algorithms.train_dpop import dpop_train as _dpop_train

        new_key, _ = self._update_fresh_key(key)
        theta_obj = self._prepare_theta_input(theta)
        theta_nat = theta_obj.params()[0]
        param_names = self.canonical_param_names
        theta_est_dict = self.par_trans.to_est(cast(ParamDict, theta_nat))
        theta_init = jnp.array([theta_est_dict[name] for name in param_names])

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

        # Full (M, p) per-iteration LR schedule (e.g. from
        # LearningRate(...).cosine_decay(...)); the kernel indexes row m.
        eta_array = eta.to_array(param_names, M)

        ys_array = jnp.array(self.ys.values)
        dt_array_extended = self._dt_array_extended
        nstep_array = self._nstep_array
        t0 = self.t0
        times_array = jnp.array(self.ys.index.values)

        rinitializer = self.rinit.struct_pf
        rprocess_interp = self.rproc.struct_pf_interp

        if self.dmeas is None:
            raise ValueError("dpop_train requires self.dmeas to be not None.")
        dmeasure = self.dmeas.struct_pf

        accumvars = self.rproc.accumvars
        covars_extended = self._covars_extended

        if process_weight_state is None:
            raise ValueError(
                "dpop_train requires a process-weight state. "
                "Please provide `process_weight_state` as the name of the "
                "state variable that accumulates the transition log-weight "
                "(e.g. 'logw')."
            )

        try:
            process_weight_index = int(self.statenames.index(process_weight_state))
        except ValueError as e:
            raise ValueError(
                f"State '{process_weight_state}' not found in statenames "
                f"{self.statenames}"
            ) from e

        ntimes = len(self.ys)
        opt_name = optimizer.__class__.__name__
        beta1 = getattr(optimizer, "beta1", 0.9)
        beta2 = getattr(optimizer, "beta2", 0.999)
        epsilon = getattr(optimizer, "epsilon", 1e-8)
        theta_hist, nll_hist = _dpop_train(
            theta_init=theta_init,
            ys=ys_array,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            t0=t0,
            times=times_array,
            J=J,
            rinitializer=rinitializer,
            rprocess_interp=rprocess_interp,
            dmeasure=dmeasure,
            accumvars=accumvars,
            covars_extended=covars_extended,
            alpha=alpha,
            process_weight_index=process_weight_index,
            ntimes=ntimes,
            key=new_key,
            M=M,
            eta=eta_array,
            optimizer=opt_name,
            decay=decay,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            alpha_cooling=alpha_cooling,
        )

        return nll_hist, theta_hist

    def pmcmc(
        self,
        J: int,
        Nmcmc: int,
        proposal,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        thresh: float = 0.0,
        track_time: bool = True,
    ) -> None:
        """
        Particle Markov chain Monte Carlo (PMMH) for Bayesian parameter inference.

        Runs one independent PMCMC chain for each parameter replicate in ``theta``.
        Each chain uses a bootstrap particle filter likelihood estimate inside a
        Metropolis-Hastings update. Results are stored in
        :attr:`Pomp.results_history`.

        Args:
            J: Number of particles per particle-filter likelihood evaluation.
            Nmcmc: Number of MCMC iterations per chain.
            proposal: Proposal object from :mod:`pypomp.proposals`.
            dprior: Pure-JAX log-prior function with signature
                ``dprior(theta_arr) -> scalar``. If ``None``, a flat improper
                prior is used.
            key: JAX PRNG key. Defaults to :attr:`fresh_key`.
            theta: Starting parameter values. Defaults to :attr:`theta`.
            thresh: Adaptive resampling threshold passed to the particle filter.
            track_time: Whether to record execution time.

        Returns:
            None. Updates :attr:`Pomp.results_history` with a
            :class:`~pypomp.core.results.PompPMCMCResult`.
        """
        start_time = time.time()

        if self.dmeas is None:
            raise ValueError("pmcmc requires self.dmeas to be not None.")
        if J < 1:
            raise ValueError("J must be >= 1.")
        if Nmcmc < 1:
            raise ValueError("Nmcmc must be >= 1.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)
        n_chains = theta_obj_in.num_replicates()
        if n_chains < 1:
            raise ValueError("pmcmc requires at least one starting parameter set.")

        new_key, old_key = self._update_fresh_key(key)
        canonical_names = self.canonical_param_names
        theta_array = theta_obj_in.to_jax_array(canonical_names)
        proposal = _expand_proposal(proposal, canonical_names)
        log_prior = dprior if dprior is not None else _flat_dprior
        keys = jax.random.split(new_key, n_chains)

        ll_jax, lp_jax, theta_jax, accepts_jax = F.pmcmc(
            self.to_struct(),
            theta_array,
            proposal,
            log_prior,
            Nmcmc,
            J,
            thresh,
            keys,
        )

        ll_traces, lp_traces, theta_traces, accepts = jax.device_get(
            (ll_jax, lp_jax, theta_jax, accepts_jax)
        )

        trace_vars = ["logLik", "log_prior"] + list(canonical_names)
        trace_data = np.concatenate(
            [ll_traces[..., np.newaxis], lp_traces[..., np.newaxis], theta_traces],
            axis=-1,
        )
        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_chains),
                "iteration": np.arange(Nmcmc + 1),
                "variable": trace_vars,
            },
        )

        final_theta_da = xr.DataArray(
            theta_traces[:, -1, :],
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_chains),
                "parameter": canonical_names,
            },
        )
        self.theta = PompParameters(final_theta_da, logLik=ll_traces[:, -1])

        execution_time = time.time() - start_time if track_time else None
        result = PompPMCMCResult(
            method="pmcmc",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=traces_da,
            Nmcmc=Nmcmc,
            J=J,
            accepts=np.asarray(accepts, dtype=np.int32),
        )
        self.results_history.add(result)

    def abc(
        self,
        Nabc: int,
        probes: dict[str, Callable],
        scale: dict[str, float],
        epsilon: float,
        proposal,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        track_time: bool = True,
    ) -> None:
        """
        Approximate Bayesian Computation with a Metropolis-Hastings outer loop.

        The probe functions must be pure JAX callables accepting a simulated
        observation array with shape ``(n_obs, ydim)`` and returning a scalar.
        One independent ABC-MCMC chain is run for each parameter replicate in
        ``theta``. Results are stored in :attr:`Pomp.results_history`.

        Args:
            Nabc: Number of ABC-MCMC iterations per chain.
            probes: Mapping from probe name to pure-JAX summary statistic.
            scale: Positive scaling factor for each probe.
            epsilon: ABC distance threshold.
            proposal: Proposal object from :mod:`pypomp.proposals`.
            dprior: Pure-JAX log-prior function. If ``None``, a flat improper
                prior is used.
            key: JAX PRNG key. Defaults to :attr:`fresh_key`.
            theta: Starting parameter values. Defaults to :attr:`theta`.
            track_time: Whether to record execution time.

        Returns:
            None. Updates :attr:`Pomp.results_history` with a
            :class:`~pypomp.core.results.PompABCResult`.
        """
        start_time = time.time()

        if self.rmeas is None:
            raise ValueError("abc requires self.rmeas to be not None.")
        if Nabc < 1:
            raise ValueError("Nabc must be >= 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not probes:
            raise ValueError("probes must be a non-empty dict.")
        if set(scale.keys()) != set(probes.keys()):
            raise ValueError("scale keys must match probes keys.")
        for name, value in scale.items():
            if value <= 0:
                raise ValueError(f"scale['{name}'] must be positive.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_obj_for_result = deepcopy(theta_obj_in)
        n_chains = theta_obj_in.num_replicates()
        if n_chains < 1:
            raise ValueError("abc requires at least one starting parameter set.")

        new_key, old_key = self._update_fresh_key(key)
        canonical_names = self.canonical_param_names
        theta_array = theta_obj_in.to_jax_array(canonical_names)
        proposal = _expand_proposal(proposal, canonical_names)
        log_prior = dprior if dprior is not None else _flat_dprior

        probe_names = sorted(probes.keys())
        scale_arr = jnp.asarray([float(scale[name]) for name in probe_names])

        def probe_fn(y_arr: jax.Array) -> jax.Array:
            return jnp.stack(
                [jnp.asarray(probes[name](y_arr)).reshape(()) for name in probe_names]
            )

        obs_probes = probe_fn(jnp.asarray(self.ys.values))
        keys = jax.random.split(new_key, n_chains)
        ydim = int(self.ys.shape[1])

        dist_jax, lp_jax, theta_jax, accepts_jax = F.abc(
            self.to_struct(),
            theta_array,
            proposal,
            log_prior,
            probe_fn,
            obs_probes,
            scale_arr,
            float(epsilon),
            ydim,
            Nabc,
            keys,
        )

        dist_traces, lp_traces, theta_traces, accepts = jax.device_get(
            (dist_jax, lp_jax, theta_jax, accepts_jax)
        )

        trace_vars = ["distance", "log_prior"] + list(canonical_names)
        trace_data = np.concatenate(
            [
                dist_traces[..., np.newaxis],
                lp_traces[..., np.newaxis],
                theta_traces,
            ],
            axis=-1,
        )
        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_chains),
                "iteration": np.arange(Nabc + 1),
                "variable": trace_vars,
            },
        )

        final_theta_da = xr.DataArray(
            theta_traces[:, -1, :],
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(n_chains),
                "parameter": canonical_names,
            },
        )
        self.theta = PompParameters(final_theta_da)

        execution_time = time.time() - start_time if track_time else None
        result = PompABCResult(
            method="abc",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_for_result,
            traces_da=traces_da,
            Nabc=Nabc,
            epsilon=float(epsilon),
            accepts=np.asarray(accepts, dtype=np.int32),
        )
        self.results_history.add(result)

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        *,
        as_pomp: Literal[True],
    ) -> "Pomp": ...

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: bool = False,
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], "Pomp"]:
        """
        Simulates the latent state and measurement processes of the POMP model.

        This method propagates the system's latent state through time according to the
        process model (`rproc`) and generates corresponding simulated observations from
        the measurement model (`rmeas`).

        This implementation leverages JAX to efficiently vectorize the simulations across
        multiple parameter sets and simulation replicates simultaneously.

        Args:
            key (jax.Array, optional): The random key for random number generation.
                Defaults to self.fresh_key.
            theta (PompParameters, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Providing a :class:`~pypomp.core.parameters.PompParameters` object with multiple parameter sets enables faster, vectorized
                execution across all parameter sets.
            times (jax.Array, optional): Times at which to generate observations.
                Defaults to self.ys.index.
            nsim (int): The number of simulations to perform. Defaults to 1.
            as_pomp (bool): If True, returns a new Pomp object containing the simulated
                observations for the first parameter replicate and simulation, instead of DataFrames.

        Returns:
            If as_pomp is False:
                tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the simulated unobserved state values and the simulated observed values.
                The columns are as follows:
                - theta_idx: The index of the parameter set.
                - sim: The index of the simulation.
                - time: The time points at which the observations were made.
                - Remaining columns contain the features of the state and observation processes.
            If as_pomp is True:
                Pomp: A deep copy of the original model, where the `ys` attribute contains one dataset of simulated observations.
        """
        if as_pomp:
            if nsim > 1:
                warnings.warn(
                    "as_pomp is True, but nsim > 1. Only 1 simulation will be performed as_pomp overrides nsim.",
                    UserWarning,
                )
            nsim = 1

        theta_obj_in = self._prepare_theta_input(theta)

        if self.rmeas is None:
            raise ValueError(
                "self.rmeas cannot be None. Did you forget to supply it to the object or method?"
            )

        thetas_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, thetas_array.shape[0])
        times_array = jnp.array(self.ys.index) if times is None else times
        X_sims, Y_sims = F.simulate(
            self.to_struct(),
            thetas_array,
            nsim,
            keys,
            times=times_array,
        )

        def _to_long(
            arr: Union[jax.Array, np.ndarray],
            times_vec: Union[jax.Array, np.ndarray, pd.Index],
            prefix: str,
        ) -> pd.DataFrame:
            vals = np.asarray(arr)  # (n_theta, n_sim, n_time, n_feat)
            n_theta_l, n_sim_l, n_time_l, n_feat_l = vals.shape
            flat = vals.reshape(n_theta_l * n_sim_l * n_time_l, n_feat_l)
            theta_idx_l = np.repeat(np.arange(n_theta_l), n_sim_l * n_time_l)
            sim_idx_l = np.tile(np.repeat(np.arange(n_sim_l), n_time_l), n_theta_l)
            time_vals_l = np.tile(
                np.asarray(times_vec).reshape(1, -1), (n_theta_l * n_sim_l, 1)
            ).reshape(-1)
            cols = pd.Index([f"{prefix}_{i}" for i in range(n_feat_l)])
            df = pd.DataFrame(flat, columns=cols)
            df.insert(0, "time", time_vals_l)
            df.insert(0, "sim", sim_idx_l)
            df.insert(0, "theta_idx", theta_idx_l)
            return df

        times0 = np.concatenate([np.array([self.t0]), np.array(times_array)])
        X_sims_long = _to_long(X_sims, times0, "state")
        Y_sims_long = _to_long(Y_sims, np.array(times_array), "obs")

        if as_pomp:
            simulated_ys_long = Y_sims_long[
                (Y_sims_long["theta_idx"] == 0) & (Y_sims_long["sim"] == 0)
            ].copy()
            simulated_ys = pd.DataFrame(
                simulated_ys_long.drop(columns=["theta_idx", "sim", "time"])
            )
            simulated_ys.index = pd.Index(simulated_ys_long["time"])
            simulated_ys.columns = self.ys.columns

            pomp_copy = deepcopy(self)
            pomp_copy.ys = simulated_ys
            pomp_copy.theta = theta_obj_in.subset([0])
            return pomp_copy

        return X_sims_long, Y_sims_long

    def probe(
        self,
        probes: dict[str, Callable[[pd.DataFrame], float]],
        nsim: int = 100,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
    ) -> pd.DataFrame:
        """
        Evaluates model diagnostics by comparing 'probes' (summary statistics) of real data against simulated data.

        This method is useful for assessing model goodness-of-fit by checking if specific
        features of the observed data (e.g., mean, autocorrelation, peak height) are
        well-captured by simulations generated from the model's parameters. It calculates
        the specified probe statistics for the original dataset and for multiple simulation
        replicates, providing a basis for visual or formal comparison.

        Args:
            probes (dict[str, Callable[[pd.DataFrame], float]]): A dictionary of probe functions.
                Each function should receive a DataFrame of observations (with time as the index,
                or a single dataframe component) and return a numeric scalar.
                Example: `{"mean": lambda df: df["obs"].mean()}`
            nsim (int, optional): Number of simulations to run per parameter set. Defaults to 100.
            key (jax.Array, optional): JAX random key for the simulations.
            theta (PompParameters, optional): Parameters to simulate from.


        Returns:
            pd.DataFrame: A long-format DataFrame with columns:
                `probe`, `value`, `is_real_data`, `theta_idx`, `sim`
        """
        sim_result = self.simulate(nsim=nsim, key=key, theta=theta, as_pomp=False)
        assert isinstance(sim_result, tuple)
        _, y_sims = sim_result

        results = []

        for name, func in probes.items():
            results.append(
                {
                    "probe": name,
                    "value": float(func(self.ys)),
                    "is_real_data": True,
                    "theta_idx": pd.NA,
                    "sim": pd.NA,
                }
            )

        def apply_probes(group):
            theta_idx, sim_id = group.name
            df = pd.DataFrame(group.drop(columns=["time"]))
            df.index = pd.Index(group["time"])
            df.columns = self.ys.columns
            for name, func in probes.items():
                results.append(
                    {
                        "probe": name,
                        "value": float(func(df)),
                        "is_real_data": False,
                        "theta_idx": theta_idx,
                        "sim": sim_id,
                    }
                )

        y_sims.groupby(["theta_idx", "sim"]).apply(apply_probes, include_groups=False)  # type: ignore[call-overload]

        return pd.DataFrame(results)

    def traces(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the full trace of log-likelihoods and parameters from the entire result history.
        Columns are

            - theta_idx: The index of the parameter set (for all methods)
            - iteration: The global iteration number for that parameter set (increments over all mif/train calls for that set; for pfilter, the last iteration for that set)
            - method: 'pfilter', 'mif', or 'train'
            - logLik: The log-likelihood estimate (averaged over reps for pfilter)
            - <param>: One column for each parameter
        """
        return self.results_history.traces()

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the results of the method run at the given index in the model's history.

        This method provides a convenient way to access the outcome of previous runs
        (e.g., `pfilter`, `mif`, or `train`). It returns a tidy DataFrame containing
        the final log-likelihoods and parameter values for all replicates associated
        with that specific run.

        Args:
            index (int): The index of the result to return. Defaults to -1 (the last result).
            ignore_nan (bool): If True, ignore NaNs when computing the log-likelihood.

        Returns:
            pd.DataFrame: A DataFrame with the results of the method run at the given index.
        """
        return self.results_history.results(index=index, ignore_nan=ignore_nan)

    def CLL(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """
        Returns a tidy DataFrame with the conditional log-likelihoods of the method run at the given index.

        Args:
            index (int, optional): The index of the result to retrieve. Defaults to -1.
            average (bool, optional): Boolean flag controlling whether to average
                the conditional log-likelihoods over replicates using logmeanexp.
                Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame with the conditional log-likelihoods.
        """
        return self.results_history.CLL(index=index, average=average)

    def ESS(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """
        Returns a tidy DataFrame with the effective sample size of the method run at the given index.

        Args:
            index (int, optional): The index of the result to retrieve. Defaults to -1.
            average (bool, optional): Boolean flag controlling whether to average
                the effective sample size over replicates using arithmetic mean.
                Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame with the effective sample size.
        """
        return self.results_history.ESS(index=index, average=average)

    def time(self):
        """
        Return a DataFrame summarizing the execution times of methods run.

        Returns:
            pd.DataFrame: A DataFrame where each row contains:
                - 'method': The name of the method run.
                - 'time': The execution time in seconds.
        """
        return self.results_history.time()

    def prune(self, n: int = 1, refill: bool = True):
        """
        Filters the current set of parameter replicates to keep only the top `n` performers based on their most recent log-likelihood estimates.

        This method is commonly used after an estimation run (like `pfilter` or `mif`) to
        discard poorly performing parameter sets and focus subsequent computational effort
        on the most promising candidates. If `refill` is enabled, the kept parameters are
        duplicated to maintain the original number of replicates.

        Args:
            n (int): Number of top thetas to keep.
            refill (bool): If True, repeat the top n thetas to match the previous number of theta sets.
        """
        self.theta.prune(n=n, refill=refill)

    def plot_traces(self, show: bool = True) -> Any:
        """
        Plot the parameter and log-likelihood traces from the entire result history.
        Each facet shows a parameter or logLik. The x-axis is iteration, y-axis is value.
        Lines connect mif/train points for the same replication; pfilter points are dots. Color by replication.

        Args:
            show (bool): Whether to display the plot. Defaults to True.
        """
        traces = self.traces()
        fig = plot_traces_internal(traces, title="Pomp Traces")

        if fig is not None and show:
            fig.show()
        return fig

    def plot_simulations(
        self,
        key: jax.Array,
        nsim: int = 20,
        mode: str = "lines",
        theta: PompParameters | None = None,
        show: bool = True,
    ) -> Any:
        """
        Generates an interactive plot comparing simulated trajectories from the model against the actual observed data.

        This visualization helps assess whether the model (with its current parameters)
        produces behavior that is qualitatively similar to the observed system. It can
        display individual simulated paths ('lines') or confidence intervals ('quantiles')
        to represent the distribution of possible outcomes.

        Args:
            key (jax.Array): JAX random key for simulation.
            nsim (int): Number of simulations to perform. Defaults to 20.
            mode (str): Plotting mode, either "lines" (individual sims) or "quantiles" (shaded region).
                Defaults to "lines".
            theta (PompParameters, optional): Parameters to use for simulation. Defaults to the first replicate in self.theta.
            show (bool): Whether to display the plot. Defaults to True.
        """
        if theta is None:
            theta = (
                self.theta.subset([0])
                if self.theta and self.theta.num_replicates() > 1
                else self.theta
            )
        elif not isinstance(theta, PompParameters):
            raise TypeError("theta must be a PompParameters instance")

        _, sims = self.simulate(nsim=nsim, theta=theta, key=key)
        fig = plot_simulations_internal(sims, self.ys, mode=mode)

        if fig is not None and show:
            fig.show()
        return fig

    def print_summary(self, n: int = 5):
        """
        Prints a high-level summary of the POMP model instance and its estimation history.

        The summary includes:
        - Basic model statistics such as the number of observations, time steps, and parameters.
        - The current number of parameter replicates stored in the object.
        - A summary of the results history, listing the execution of estimation methods (e.g., pfilter, mif, train) and their corresponding performance metrics.
        """
        print("Basics:")
        print("-------")
        print(f"Number of observations: {len(self.ys)}")
        print(f"Number of time steps: {len(self._dt_array_extended)}")
        print(f"Number of parameters: {self.theta.num_params()}")
        print(f"Number of parameter sets: {self.theta.num_replicates()}")
        print()
        self.results_history.print_summary(n=n)

    def __eq__(self, other):
        """
        Check structural equality with another Pomp object.

        Two Pomp instances are considered equal if they:
        - Are of the same type
        - Have identical canonical parameter names
        - Have equal parameter sets (self.theta)
        - Have identical data (ys) and covariates (covars)
        - Have the same state names and initial time t0
        - Have equivalent model components (rinit, rproc, dmeas, rmeas)
        - Have equal fresh_key values (or both None)
        """
        if not isinstance(other, type(self)):
            return False

        # Canonical parameter names
        if self.canonical_param_names != other.canonical_param_names:
            return False

        # Parameter sets
        if (self._theta is None) != (other._theta is None):
            return False
        if self._theta is not None and other._theta is not None:
            if self._theta != other._theta:
                return False

        # Data and covariates
        if not self.ys.equals(other.ys):
            return False
        if (self.covars is None) != (other.covars is None):
            return False
        if self.covars is not None and other.covars is not None:
            if not self.covars.equals(other.covars):
                return False
        # Handle _covars_extended (can be None or JAX array)
        if (self._covars_extended is None) != (other._covars_extended is None):
            return False
        if self._covars_extended is not None and other._covars_extended is not None:
            if not jax.numpy.array_equal(self._covars_extended, other._covars_extended):
                return False
        # Compare JAX arrays using array_equal
        if not jax.numpy.array_equal(self._nstep_array, other._nstep_array):
            return False
        if not jax.numpy.array_equal(self._dt_array_extended, other._dt_array_extended):
            return False
        if self._max_steps_per_interval != other._max_steps_per_interval:
            return False

        # State names and initial time
        if self.statenames != other.statenames:
            return False
        if float(self.t0) != float(other.t0):
            return False

        # Model components: rely on their own __eq__ implementations
        if self.rinit != other.rinit:
            return False
        if self.rproc != other.rproc:
            return False
        if (self.dmeas is None) != (other.dmeas is None):
            return False
        if self.dmeas is not None and self.dmeas != other.dmeas:
            return False
        if (self.rmeas is None) != (other.rmeas is None):
            return False
        if self.rmeas is not None and self.rmeas != other.rmeas:
            return False

        if self.results_history != other.results_history:
            return False

        if self.par_trans != other.par_trans:
            return False

        # fresh_key: both None or numerically equal
        if (self.fresh_key is None) != (other.fresh_key is None):
            return False
        if self.fresh_key is not None and other.fresh_key is not None:
            if not jax.numpy.array_equal(
                jax.random.key_data(self.fresh_key),
                jax.random.key_data(other.fresh_key),
            ):
                return False

        return True

    @staticmethod
    def merge(*pomp_objs: "Pomp") -> "Pomp":
        """
        Merges multiple `Pomp` objects into a single instance by combining their parameter replicates and results histories.

        All provided `Pomp` objects must share the same structural components (e.g., state
        names, parameter names, and model logic). The resulting object will contain the
        union of all parameter sets and their corresponding estimation results, which is
        particularly useful for consolidating parallelized simulation or estimation runs.
        """
        if len(pomp_objs) == 0:
            raise ValueError("At least one Pomp object must be provided.")
        first = pomp_objs[0]

        for obj in pomp_objs:
            if not isinstance(obj, type(first)):
                raise TypeError("All merged objects must be of type Pomp.")
            if obj.canonical_param_names != first.canonical_param_names:
                raise ValueError(
                    "All Pomp objects must have the same canonical_param_names."
                )
            if obj.statenames != first.statenames:
                raise ValueError("All Pomp objects must have the same statenames.")
            if not obj.ys.equals(first.ys):
                raise ValueError("All Pomp objects must have the same ys data.")
            if obj.t0 != first.t0:
                raise ValueError("All Pomp objects must have the same t0.")
            if obj.rinit != first.rinit or obj.rproc != first.rproc:
                raise ValueError("All Pomp objects must have the same rinit and rproc.")
            if (obj.dmeas is None) != (first.dmeas is None):
                raise ValueError(
                    "All Pomp objects must have the same dmeas (both None or both not None)."
                )
            if obj.dmeas is not None and obj.dmeas != first.dmeas:
                raise ValueError("All Pomp objects must have the same dmeas.")
            if (obj.rmeas is None) != (first.rmeas is None):
                raise ValueError(
                    "All Pomp objects must have the same rmeas (both None or both not None)."
                )
            if obj.rmeas is not None and obj.rmeas != first.rmeas:
                raise ValueError("All Pomp objects must have the same rmeas.")
            if obj.par_trans != first.par_trans:
                raise ValueError("All Pomp objects must have the same par_trans.")

        thetas = []
        for obj in pomp_objs:
            if obj._theta is None:
                raise ValueError("Cannot merge Pomp objects with no parameters.")
            thetas.append(obj._theta)

        merged_theta = PompParameters.merge(*thetas)
        merged_history = ResultsHistory.merge(
            *[obj.results_history for obj in pomp_objs]
        )

        merged_pomp = deepcopy(first)
        merged_pomp._theta = merged_theta
        merged_pomp.results_history = merged_history
        merged_pomp.fresh_key = first.fresh_key

        return merged_pomp

    def __getstate__(self):
        """
        Custom pickling method to handle wrapped function objects.  This is
        necessary because the JAX-wrapped functions are not picklable.
        """
        state = self.__dict__.copy()

        # Use cloudpickle to store model functions by-value. This ensures that
        # the unpickling environment does not require the original source modules.
        if hasattr(self.rinit, "struct"):
            original_func = self.rinit.original_func
            state["_rinit_func_bytes"] = cloudpickle.dumps(original_func)

        if hasattr(self.rproc, "struct"):
            original_func = self.rproc.original_func
            state["_rproc_func_bytes"] = cloudpickle.dumps(original_func)
            state["_rproc_dt"] = getattr(self.rproc, "dt", None)
            state["_rproc_nstep"] = getattr(self.rproc, "nstep", None)
            state["_rproc_accumvars"] = getattr(self.rproc, "accumvars", None)

        if self.dmeas is not None and hasattr(self.dmeas, "struct"):
            original_func = self.dmeas.original_func
            state["_dmeas_func_bytes"] = cloudpickle.dumps(original_func)

        if self.rmeas is not None and hasattr(self.rmeas, "struct"):
            original_func = self.rmeas.original_func
            state["_rmeas_func_bytes"] = cloudpickle.dumps(original_func)

        # Store JAX key as raw bits (key is not picklable directly)
        if self.fresh_key is not None:
            state["_fresh_key_data"] = jax.random.key_data(self.fresh_key)

        # Remove the wrapped objects and key from state
        state.pop("rinit", None)
        state.pop("rproc", None)
        state.pop("dmeas", None)
        state.pop("rmeas", None)
        state.pop("fresh_key", None)

        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to reconstruct wrapped function objects. This is
        necessary because the JAX-wrapped functions are not picklable.
        """
        # Restore basic attributes
        self.__dict__.update(state)

        # Reconstruct JAX key from raw bits
        if "_fresh_key_data" in state:
            try:
                self.fresh_key = cast(
                    jax.Array, jax.random.wrap_key_data(state["_fresh_key_data"])
                )
            except Exception as e:
                warnings.warn(f"Failed to reconstruct JAX fresh_key: {e}", UserWarning)
                self.fresh_key = None
        elif "fresh_key" not in self.__dict__:
            self.fresh_key = None

        def _load_func(prefix: str) -> Any:
            func_bytes_key = f"_{prefix}_func_bytes"
            func_name_key = f"_{prefix}_func_name"
            module_key = f"_{prefix}_module"

            try:
                # Modern approach (by-value): Uses cloudpickle bytes to remove
                # environment dependencies.
                if func_bytes_key in state:
                    return cloudpickle.loads(state[func_bytes_key])

                # Legacy approach (by-reference): Provided for backward compatibility
                # with objects pickled in older versions of pypomp.
                elif func_name_key in state:
                    module = importlib.import_module(state[module_key])
                    return getattr(module, state[func_name_key])
            except Exception as e:
                warnings.warn(
                    f"Failed to reconstruct {prefix} function: {e}. "
                    f"The model may be unusable for simulations or estimation.",
                    UserWarning,
                )
            return None

        # Reconstruct rinit
        obj_rinit = _load_func("rinit")
        if obj_rinit is not None:
            if isinstance(obj_rinit, _RInit):
                self.rinit = obj_rinit
            else:
                self.rinit = _RInit(
                    struct=obj_rinit,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                )

        # Reconstruct rproc
        obj_rproc = _load_func("rproc")
        if obj_rproc is not None:
            if isinstance(obj_rproc, _RProc):
                self.rproc = obj_rproc
            else:
                kwargs = {}
                if state.get("_rproc_dt") is not None:
                    kwargs["dt"] = state["_rproc_dt"]
                if (
                    state.get("_rproc_nstep") is not None
                    and state.get("_rproc_dt") is None
                ):
                    kwargs["nstep"] = state["_rproc_nstep"]
                if state.get("_rproc_accumvars") is not None:
                    kwargs["accumvars"] = state["_rproc_accumvars"]
                self.rproc = _RProc(
                    struct=obj_rproc,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    **kwargs,
                )
                if state.get("_rproc_nstep") is not None:
                    if state.get("_rproc_dt") is not None:
                        self.rproc.nstep = state["_rproc_nstep"]

        # Reconstruct dmeas
        obj_dmeas = _load_func("dmeas")
        if obj_dmeas is not None:
            if isinstance(obj_dmeas, _DMeas):
                self.dmeas = obj_dmeas
            else:
                self.dmeas = _DMeas(
                    struct=obj_dmeas,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    y_names=list(self.ys.columns) if hasattr(self, "ys") else None,
                )

        # Reconstruct rmeas
        obj_rmeas = _load_func("rmeas")
        if obj_rmeas is not None:
            if isinstance(obj_rmeas, _RMeas):
                self.rmeas = obj_rmeas
            else:
                self.rmeas = _RMeas(
                    struct=obj_rmeas,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    y_names=list(self.ys.columns) if hasattr(self, "ys") else None,
                )

        # Set defaults if reconstruction failed or was missing
        if not hasattr(self, "rinit"):
            self.rinit = None  # type: ignore
        if not hasattr(self, "rproc"):
            self.rproc = None  # type: ignore
        if not hasattr(self, "rmeas"):
            self.rmeas = None
        if not hasattr(self, "dmeas"):
            self.dmeas = None

        # Clean up temporary state variables
        for key in [
            "_rinit_func_bytes",
            "_rinit_func_name",
            "_rinit_module",
            "_rproc_func_bytes",
            "_rproc_func_name",
            "_rproc_dt",
            "_rproc_nstep",
            "_rproc_accumvars",
            "_rproc_module",
            "_dmeas_func_bytes",
            "_dmeas_func_name",
            "_dmeas_module",
            "_rmeas_func_bytes",
            "_rmeas_func_name",
            "_rmeas_module",
            "_fresh_key_data",
        ]:
            if key in self.__dict__:
                del self.__dict__[key]

    def arma(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        log_ys: bool = False,
        suppress_warnings: bool = True,
    ) -> float:
        """
        Fits an independent ARIMA model to the observation data and returns the estimated
        log-likelihood.

        This is a wrapper around `pypomp.benchmarks.arma`.

        Args:
            order (tuple, optional): The (p, d, q) order for the ARIMA model. Defaults to (1, 0, 1).
            log_ys (bool, optional): If True, fits the model to log(y+1). Defaults to False.
            suppress_warnings (bool, optional): If True, suppresses individual warnings from statsmodels
                and issues a summary warning instead. Defaults to True.

        Returns:
            float: The sum of the log-likelihoods.
        """
        return benchmarks.arma(
            self.ys, order=order, log_ys=log_ys, suppress_warnings=suppress_warnings
        )

    def negbin(
        self, autoregressive: bool = False, suppress_warnings: bool = True
    ) -> float:
        """
        Fits a Negative Binomial model to the observation data and returns
        the log-likelihood.

        This is a wrapper around `pypomp.benchmarks.negbin`.

        Args:
            autoregressive (bool, optional): If True, fits an AR(1) model.
                Defaults to False (iid).
            suppress_warnings (bool, optional): If True, suppresses individual warnings from statsmodels/optimization
                and issues a summary warning instead. Defaults to True.

        Returns:
            float: The sum of the log-likelihoods.
        """
        return benchmarks.negbin(
            self.ys,
            autoregressive=autoregressive,
            suppress_warnings=suppress_warnings,
        )

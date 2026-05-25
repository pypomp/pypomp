"""
This module implements the OOP structure for POMP models.
"""

import importlib
import cloudpickle
from copy import deepcopy
import time
from typing import Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import warnings
from typing import Union, overload, Literal
from .viz import plot_traces_internal, plot_simulations_internal

from pypomp.types import ThetaInput
from .metadata import ModelMetadata
from pypomp import functional as F
from .model_struct import _RInit, _RProc, _DMeas, _RMeas
import xarray as xr
from .algorithms.helpers import _calc_ys_covars
from .algorithms.pfilter import _pfilter_internal, _vmapped_pfilter_internal
from .rw_sigma import RWSigma
from .learning_rate import LearningRate
from .par_trans import ParTrans
from .optimizer import Optimizer, Adam
from .results import (
    ResultsHistory,
    PompPFilterResult,
    PompMIFResult,
    PompTrainResult,
    PompPMCMCResult,
    PompABCResult,
)
from .parameters import PompParameters
from pypomp.maths import logmeanexp
from pypomp import benchmarks
from pypomp.functional.structs import PompStruct


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
    theta : ThetaInput
        Initial parameter(s) for the model. Accepts:
        - A single dictionary: dict[str, Numeric]
        - A list of dictionaries: list[dict[str, Numeric]]
        - An existing PompParameters object
        Numeric values (e.g. jax.Array, int) are automatically coerced to
        standard Python floats for internal storage. Vectorized methods
        (like pfilter) will run in parallel over list/PompParameters inputs.
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
    par_trans : ParTrans, optional
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
    """

    ys: pd.DataFrame
    """The measurement data frame with observation times as the index."""

    _theta: PompParameters
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
    """History of results from `pfilter`, `mif`, and `train` calls."""

    fresh_key: jax.Array | None
    """Running a method that takes a key will store a fresh, unused key here."""

    metadata: ModelMetadata
    """Environment and version metadata initialized when this instance was built."""

    def __init__(
        self,
        ys: pd.DataFrame,
        theta: ThetaInput,
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
    ):
        if not isinstance(ys, pd.DataFrame):
            raise TypeError("ys must be a pandas DataFrame")
        if covars is not None and not isinstance(covars, pd.DataFrame):
            raise TypeError("covars must be a pandas DataFrame or None")

        if isinstance(theta, PompParameters):
            self._theta = theta
        else:
            self._theta = PompParameters(theta)

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
            order="linear",
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
        return self._theta

    @theta.setter
    def theta(self, value: ThetaInput):
        if isinstance(value, PompParameters):
            self._theta = value
        else:
            self._theta = PompParameters(value)

    def _prepare_theta_input(
        self,
        theta: ThetaInput,
    ) -> PompParameters:
        """
        Prepare the theta input for the method.
        """
        if theta is None:
            return self.theta
        elif isinstance(theta, dict) or isinstance(theta, list):
            theta = PompParameters(theta)
        elif isinstance(theta, PompParameters):
            pass
        else:
            raise TypeError(
                "theta must be a dictionary, a list of dictionaries, or a PompParameters object"
            )
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
        )

    @staticmethod
    def sample_params(
        param_bounds: dict[str, tuple[float, float]], n: int, key: jax.Array
    ) -> list[dict[str, float]]:
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
            list[dict]: List of n dictionaries containing sampled parameters
        """
        keys = jax.random.split(key, len(param_bounds))
        param_sets = []

        for i in range(n):
            params = {}
            for j, (param_name, (lower, upper)) in enumerate(param_bounds.items()):
                subkey = jax.random.split(keys[j], n)[i]
                params[param_name] = float(
                    jax.random.uniform(subkey, shape=(), minval=lower, maxval=upper)
                )
            param_sets.append(params)

        return param_sets

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
        theta: ThetaInput = None,
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
        model's history and can be accessed using `self.results()`.

        Args:
            J (int): The number of particles
            key (jax.Array, optional): The random key. Defaults to self.fresh_key.
            theta (ThetaInput, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Accepts:
                - A single dictionary: dict[str, Numeric]
                - A list of dictionaries: list[dict[str, Numeric]]
                - An existing PompParameters object
                Providing a list or PompParameters object enables faster, vectorized
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
            None. Updates `self.results_history` with a `PompPFilterResult` containing the log-likelihoods,
            and optionally the conditional log-likelihoods (CLL), effective sample size (ESS),
            filtered means, and prediction means if requested.
        """
        start_time = time.time()

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        n_theta_reps = theta_obj_in.num_replicates()

        new_key, old_key = self._update_fresh_key(key)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        thetas_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        rep_keys = jax.random.split(new_key, n_theta_reps * reps).reshape(
            n_theta_reps, reps, *new_key.shape
        )

        if len(jax.devices()) > 1:
            mesh = jax.sharding.Mesh(jax.devices(), axis_names=("theta_reps",))
            sharding_spec = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("theta_reps", None)
            )
            rep_keys_sharding_spec = jax.sharding.NamedSharding(
                mesh,
                jax.sharding.PartitionSpec(
                    "theta_reps", *([None] * (rep_keys.ndim - 1))
                ),
            )
            thetas_array = jax.device_put(thetas_array, sharding_spec)
            rep_keys = jax.device_put(rep_keys, rep_keys_sharding_spec)

        results_jax = F.pfilter(
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
            theta=theta_obj_in.to_list(),
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
        a: float,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
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
        the model's history and can be accessed using `self.results()`.

        Args:
            J (int): The number of particles.
            M (int): Number of algorithm iterations.
            rw_sd (RWSigma): Random walk sigma object.
            a (float): Decay factor for RWSigma over 50 iterations.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (ThetaInput, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Accepts:
                - A single dictionary: dict[str, Numeric]
                - A list of dictionaries: list[dict[str, Numeric]]
                - An existing PompParameters object
                Providing a list or PompParameters object enables faster, vectorized
                execution across all parameter sets.
            thresh (float): Resampling threshold. Defaults to 0.
            n_monitors (int): Number of particle filter runs to average for
                log-likelihood estimation. Defaults to 0 (uses estimate from perturbed
                filter).
            track_time (bool): Boolean flag controlling whether to track the
                execution time.
        Returns:
            None. Updates `self.results_history` with a `PompMIFResult` containing the log-likelihoods,
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
        theta_list_in = theta_obj_in.to_list()
        n_reps = theta_obj_in.num_replicates()

        new_key, old_key = self._update_fresh_key(key)
        theta_obj_in.transform(self.par_trans, direction="to_est")
        sigmas_array, sigmas_init_array = rw_sd._return_arrays(
            param_names=self.canonical_param_names
        )
        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0.")

        keys = jax.random.split(new_key, n_reps)

        theta_tiled = jnp.tile(theta_array, (J, 1, 1))

        if len(jax.devices()) > 1:
            mesh = jax.sharding.Mesh(jax.devices(), axis_names=("reps",))
            sharding_spec = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, "reps", None)
            )
            theta_tiled = jax.device_put(theta_tiled, sharding_spec)

        nLLs_jax, theta_traces_jax, final_thetas_jax = F.mif(
            self.to_struct(),
            theta_tiled,
            sigmas_array,
            sigmas_init_array,
            M,
            a,
            J,
            thresh,
            keys,
            n_monitors,
        )

        nLLs = jax.device_get(nLLs_jax)
        theta_traces = jax.device_get(theta_traces_jax)
        final_thetas = jax.device_get(final_thetas_jax)

        del nLLs_jax, theta_traces_jax, final_thetas_jax

        final_theta_ests = []
        param_names = self.canonical_param_names
        trace_vars = ["logLik"] + param_names
        trace_data = np.zeros((n_reps, M + 1, len(trace_vars)), dtype=float)

        for i in range(n_reps):
            # Prepend nan for the log-likelihood of the initial parameters
            logliks_with_nan = np.concatenate([np.array([np.nan]), -nLLs[i]])

            param_traces = theta_traces[i]  # shape: (M+1, n_params)

            # Transform traces from estimation space to natural space
            param_traces = self.par_trans.transform_array(
                param_traces, param_names, direction="from_est"
            )
            trace_data[i, :, 0] = logliks_with_nan
            trace_data[i, :, 1:] = param_traces
            final_theta_ests.append(final_thetas[i])

        traces_da = xr.DataArray(
            trace_data,
            dims=["theta_idx", "iteration", "variable"],
            coords={
                "theta_idx": np.arange(n_reps),
                "iteration": np.arange(M + 1),
                "variable": trace_vars,
            },
        )

        theta = [
            self.par_trans.to_floats(
                theta=dict(
                    zip(
                        self.canonical_param_names,
                        np.mean(theta_est, axis=0).tolist(),
                    )
                ),
                direction="from_est",
            )
            for theta_est in final_theta_ests
        ]
        logLik_estimates = -nLLs
        self.theta = PompParameters(theta, logLik=logLik_estimates)

        del final_theta_ests

        if track_time is True:
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result = PompMIFResult(
            method="mif",
            execution_time=execution_time,
            key=old_key,
            theta=theta_list_in,
            traces_da=traces_da,
            J=J,
            M=M,
            rw_sd=rw_sd,
            a=a,
            thresh=thresh,
            n_monitors=n_monitors,
        )

        self.results_history.add(result)

    def train(
        self,
        J: int,
        M: int,
        eta: LearningRate,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
        optimizer: Optimizer = Adam(),
        alpha: float = 0.97,
        thresh: int = 0,
        alpha_cooling: float = 1.0,
        n_monitors: int = 1,
        track_time: bool = True,
    ) -> None:
        """
        Optimizes model parameters using a differentiable particle filter and gradient-based methods.

        This method performs Maximum Likelihood Estimation (MLE) by treating the particle filter
        as a differentiable computational graph. It computes gradients of the log-likelihood
        with respect to the parameters via reverse-mode automatic differentiation (using JAX),
        and updates the parameters using optimizers (e.g., Adam, SGD).

        This implementation leverages JAX to efficiently vectorize the algorithm across
        multiple initial parameter sets simultaneously.
        Results are automatically stored in the model's history and can be accessed using
        `self.results()`.

        Args:
            J (int): The number of particles in the MOP objective for obtaining the gradient and/or Hessian.
            M (int): Maximum iteration for the gradient descent optimization.
            eta (LearningRate): Learning rates per parameter as a LearningRate object.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (ThetaInput, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Accepts:
                - A single dictionary: dict[str, Numeric]
                - A list of dictionaries: list[dict[str, Numeric]]
                - An existing PompParameters object
                Providing a list or PompParameters object enables faster, vectorized
                execution across all parameter sets.
            optimizer (Optimizer, optional): The optimizer configuration object to use
                (e.g., `pp.Adam()`, `pp.SGD()`, `pp.Newton()`, `pp.FullMatrixAdam()`, etc.).
                Defaults to `pp.Adam()`. Hyperparameters like learning rate scaling, line search
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
            None. Updates `self.results_history` with a `PompTrainResult` containing the log-likelihoods,
            parameter traces, and optimizer details from the training run.
        """
        start_time = time.time()

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        theta_list_in = theta_obj_in.to_list()

        theta_obj_in.transform(self.par_trans, direction="to_est")
        n_reps = theta_obj_in.num_replicates()
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

        opt_name = optimizer.__class__.__name__
        beta1 = getattr(optimizer, "beta1", 0.9)
        beta2 = getattr(optimizer, "beta2", 0.999)
        epsilon = getattr(optimizer, "epsilon", 1e-8 if opt_name == "Adam" else 1e-4)
        c = optimizer.c
        max_ls_itn = optimizer.max_ls_itn
        clip_norm = optimizer.clip_norm
        scale = optimizer.scale
        ls = optimizer.ls

        nLLs, theta_ests = F.train(
            self.to_struct(),
            theta_array,
            J,
            opt_name,
            M,
            eta_array,
            c,
            max_ls_itn,
            thresh,
            scale,
            ls,
            alpha,
            keys,
            alpha_cooling,
            n_monitors,
            clip_norm,
            beta1,
            beta2,
            epsilon,
        )

        theta_ests_natural = np.stack(
            [
                self.par_trans.transform_array(
                    np.asarray(theta_ests[i]),
                    self.canonical_param_names,
                    direction="from_est",
                )
                for i in range(n_reps)
            ],
            axis=0,
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

        theta = [
            self.par_trans.to_floats(
                theta=dict(
                    zip(self.canonical_param_names, theta_ests[i, -1, :].tolist())
                ),
                direction="from_est",
            )
            for i in range(n_reps)
        ]
        logLik_estimates = np.asarray(-nLLs)
        self.theta = PompParameters(theta, logLik=logLik_estimates)

        if track_time is True:
            nLLs.block_until_ready()
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result = PompTrainResult(
            method="train",
            execution_time=execution_time,
            key=old_key,
            theta=theta_list_in,
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
        optimizer: str = "Adam",
        alpha: float = 0.8,
        decay: float = 0.0,
        process_weight_state: str | None = None,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Optimizes model parameters using the DPOP differentiable particle filter and gradient-based methods.

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
            Learning rates per parameter as a LearningRate object.
        optimizer : str, default "Adam"
            Optimizer to use: "Adam" or "SGD".
        alpha : float, default 0.8
            DPOP discount / cooling factor.
        decay : float, default 0.0
            Learning-rate decay coefficient. At iteration m, the effective
            learning rate is ``eta / (1 + decay * m)``.
        process_weight_state : str or None, default None
            Name of the state component that stores the accumulated
            process log-weight (e.g. ``"logw"``).
        key : jax.Array or None, default None
            Random key. If None, uses ``self.fresh_key``.
        theta : ThetaInput, default None
            Optional initial parameter(s). Accepts dict[str, Numeric],
            list[dict[str, Numeric]], or PompParameters.
            Numeric values are coerced to floats. Defaults to self.theta.

        Returns
        -------
        nll_history : jax.Array, shape (M+1,)
            Mean DPOP negative log-likelihood per observation at each step.
        theta_history : jax.Array, shape (M+1, p)
            Parameter vector (estimation space) at each step.
        """
        from .algorithms.train_dpop import dpop_train as _dpop_train

        new_key, _ = self._update_fresh_key(key)
        theta_obj = self._prepare_theta_input(theta)
        theta_nat = theta_obj.to_list()[0]
        param_names = self.canonical_param_names
        theta_est_dict = self.par_trans.to_est(theta_nat)
        theta_init = jnp.array([theta_est_dict[name] for name in param_names])

        if not isinstance(eta, LearningRate):
            raise TypeError("eta must be a LearningRate object")

        # For now, dpop_train only uses a constant learning rate across iterations
        # Extract the first row of the schedule
        eta_array = eta.to_array(param_names, M)[0]

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
            optimizer=optimizer,
            decay=decay,
        )

        return nll_hist, theta_hist

    def pmcmc(
        self,
        J: int,
        Nmcmc: int,
        proposal: Callable,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
        thresh: float = 0.0,
        reps: int = 1,
        verbose: bool = False,
    ) -> None:
        """
        Particle Markov chain Monte Carlo (PMMH) for Bayesian parameter
        estimation.

        Runs a particle random-walk Metropolis-Hastings chain for *Nmcmc*
        iterations, using the particle filter log-likelihood as a noisy but
        unbiased estimate of the marginal likelihood
        (Andrieu, Doucet & Holenstein 2010).

        Args:
            J: Number of particles per particle-filter evaluation.
            Nmcmc: Number of MCMC iterations to perform.
            proposal: A symmetric proposal function with signature
                ``proposal(theta, key, n=0, accepts=0) -> theta_proposed``
                where *theta* is a ``dict[str, float]``.
                See :mod:`pypomp.proposals` for constructors.
            dprior: Log-prior density function with signature
                ``dprior(theta_dict) -> float`` returning the log-prior
                density.  If ``None``, a flat improper prior is used
                (always returns 0).
            key: JAX PRNG key.  If ``None``, uses ``self.fresh_key``.
            theta: Starting parameter values (single dict or
                PompParameters with one replicate).  If ``None``, uses
                ``self.theta``.
            thresh: Threshold for adaptive resampling in the particle
                filter.  Defaults to 0.0 (resample at every time step).
            reps: Number of independent particle-filter runs per
                likelihood evaluation.  The log-likelihoods are combined
                via ``logmeanexp`` to reduce Monte Carlo variance.
            verbose: If ``True``, print progress after each iteration.
        """
        start_time = time.time()

        # --- Validate inputs ---
        if self.dmeas is None:
            raise ValueError("dmeas is required for pmcmc.")
        if J < 1:
            raise ValueError("J must be >= 1.")
        if Nmcmc < 1:
            raise ValueError("Nmcmc must be >= 1.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        if theta_obj_in.num_replicates() != 1:
            raise ValueError(
                "pmcmc requires exactly one parameter replicate as starting "
                "point.  Got "
                f"{theta_obj_in.num_replicates()}."
            )

        new_key, old_key = self._update_fresh_key(key)

        # Default flat prior
        _dprior: Callable = dprior if dprior is not None else lambda theta_dict: 0.0

        # --- Extract model internals (constant across iterations) ---
        param_names = self.canonical_param_names
        dt_arr = np.asarray(self._dt_array_extended)
        nstep_arr = np.asarray(self._nstep_array)
        t0 = self.t0
        times = np.asarray(self.ys.index)
        ys = np.asarray(self.ys)
        covars_ext = (
            np.asarray(self._covars_extended)
            if self._covars_extended is not None
            else None
        )
        rinit_fn = self.rinit.struct_pf
        rproc_fn = self.rproc.struct_pf_interp
        dmeas_fn = self.dmeas.struct_pf
        accumvars = self.rproc.accumvars

        # --- Helper: run pfilter and return loglik ---
        def _run_pfilter(theta_array: jax.Array, pf_key: jax.Array) -> float:
            """Run particle filter(s) and return scalar loglik estimate."""
            if reps == 1:
                result = _pfilter_internal(
                    theta_array, dt_arr, nstep_arr, t0, times, ys,
                    J, rinit_fn, rproc_fn, dmeas_fn, accumvars,
                    covars_ext, thresh, pf_key,
                )
                return -float(result["neg_loglik"])
            else:
                pf_keys = jax.random.split(pf_key, reps)
                results = _vmapped_pfilter_internal(
                    theta_array, dt_arr, nstep_arr, t0, times, ys,
                    J, rinit_fn, rproc_fn, dmeas_fn, accumvars,
                    covars_ext, thresh, pf_keys,
                    False, False, False, False, False,  # CLL, ESS, filter_mean, prediction_mean, should_trans
                )
                neg_logliks = results["neg_loglik"]
                return float(logmeanexp(-neg_logliks))

        # --- Initialise chain ---
        theta_dict = theta_obj_in.to_list()[0]
        theta_array = jnp.array([theta_dict[p] for p in param_names])

        n_params = len(param_names)
        trace_names = ["loglik", "log_prior"] + list(param_names)
        trace_arr = np.full((Nmcmc + 1, 2 + n_params), np.nan)

        # Evaluate prior and likelihood at starting point
        log_prior = float(_dprior(theta_dict))
        if not np.isfinite(log_prior):
            raise ValueError("Non-finite log prior at starting parameters.")

        new_key, pf_key = jax.random.split(new_key)
        loglik = _run_pfilter(theta_array, pf_key)
        if not np.isfinite(loglik):
            raise ValueError("Non-finite log likelihood at starting parameters.")

        trace_arr[0, :2] = [loglik, log_prior]
        trace_arr[0, 2:] = [theta_dict[p] for p in param_names]

        accepts = 0

        # --- Main MCMC loop ---
        for n in range(1, Nmcmc + 1):
            new_key, prop_key, pf_key = jax.random.split(new_key, 3)

            # Propose
            theta_prop = proposal(theta_dict, prop_key, n=n, accepts=accepts)

            # Prior
            log_prior_prop = float(_dprior(theta_prop))

            if np.isfinite(log_prior_prop):
                # Particle-filter likelihood
                theta_prop_array = jnp.array(
                    [theta_prop[p] for p in param_names]
                )
                loglik_prop = _run_pfilter(theta_prop_array, pf_key)

                # Metropolis-Hastings acceptance (symmetric proposal)
                log_alpha = loglik_prop + log_prior_prop - loglik - log_prior

                new_key, accept_key = jax.random.split(new_key)
                u = float(jax.random.uniform(accept_key))

                if np.isfinite(log_alpha) and np.log(u) < log_alpha:
                    theta_dict = theta_prop
                    theta_array = theta_prop_array
                    loglik = loglik_prop
                    log_prior = log_prior_prop
                    accepts += 1

            # Store trace
            trace_arr[n, :2] = [loglik, log_prior]
            trace_arr[n, 2:] = [theta_dict[p] for p in param_names]

            if verbose:
                print(
                    f"PMCMC iteration {n} of {Nmcmc} | "
                    f"acceptance rate: {accepts / n:.3f} | "
                    f"loglik: {loglik:.2f}"
                )

        # --- Package results ---
        execution_time = time.time() - start_time

        # Update self.theta with the final accepted parameters
        self.theta = PompParameters(theta_dict)

        result = PompPMCMCResult(
            method="pmcmc",
            execution_time=execution_time,
            key=old_key,
            theta=[theta_dict],
            traces_arr=trace_arr,
            trace_names=trace_names,
            Nmcmc=Nmcmc,
            J=J,
            reps=reps,
            accepts=accepts,
        )

        self.results_history.add(result)

    def abc(
        self,
        Nabc: int,
        probes: dict[str, Callable],
        scale: dict[str, float],
        epsilon: float,
        proposal: Callable,
        dprior: Callable | None = None,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
        verbose: bool = False,
    ) -> None:
        """
        Approximate Bayesian Computation (ABC-MCMC) for likelihood-free
        parameter estimation.

        Implements the likelihood-free MCMC sampler of Marin et al. (2012,
        Algorithm 3) with a symmetric proposal distribution, matching
        R pomp's ``abc()``.

        Args:
            Nabc: Number of ABC-MCMC iterations to perform.
            probes: Dictionary mapping probe names to callable summary
                statistics.  Each callable has signature
                ``probe_fn(ys: pd.DataFrame) -> float`` where *ys* has the
                same format as ``self.ys`` (time index, observation columns).
            scale: Dictionary mapping probe names to positive floats used
                to normalize probe distances.  Must have the same keys as
                *probes*.
            epsilon: Tolerance threshold (positive float).  A proposal is
                accepted when the scaled distance is less than
                ``epsilon ** 2``.
            proposal: A symmetric proposal function with signature
                ``proposal(theta, key, n=0, accepts=0) -> theta_proposed``
                where *theta* is a ``dict[str, float]``.
                See :mod:`pypomp.proposals` for constructors.
            dprior: Log-prior density function with signature
                ``dprior(theta_dict) -> float``.  If ``None``, a flat
                improper prior is used (always returns 0).
            key: JAX PRNG key.  If ``None``, uses ``self.fresh_key``.
            theta: Starting parameter values (single dict or
                PompParameters with one replicate).  If ``None``, uses
                ``self.theta``.
            verbose: If ``True``, print progress after each iteration.
        """
        start_time = time.time()

        # --- Validate inputs ---
        if self.rmeas is None:
            raise ValueError("rmeas is required for abc.")
        if Nabc < 1:
            raise ValueError("Nabc must be >= 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not probes:
            raise ValueError("probes must be a non-empty dict.")
        if set(scale.keys()) != set(probes.keys()):
            raise ValueError("scale keys must match probes keys.")
        for k, v in scale.items():
            if v <= 0:
                raise ValueError(f"scale['{k}'] must be positive.")

        theta_obj_in = deepcopy(self._prepare_theta_input(theta))
        if theta_obj_in.num_replicates() != 1:
            raise ValueError(
                "abc requires exactly one parameter replicate as starting "
                "point.  Got "
                f"{theta_obj_in.num_replicates()}."
            )

        new_key, old_key = self._update_fresh_key(key)

        # Default flat prior
        _dprior: Callable = dprior if dprior is not None else lambda theta_dict: 0.0

        # --- Compute observed probes ---
        probe_names = sorted(probes.keys())
        obs_probes = np.array([probes[k](self.ys) for k in probe_names])
        scale_arr = np.array([scale[k] for k in probe_names])
        obs_col_names = list(self.ys.columns)

        # --- Helper: simulate and compute distance ---
        def _simulate_distance(theta_dict: dict, sim_key: jax.Array) -> float:
            """Simulate one dataset and return scaled Euclidean distance."""
            theta_param = PompParameters(theta_dict)
            _, Y_sims = self.simulate(nsim=1, key=sim_key, theta=theta_param)
            # Reconstruct DataFrame matching self.ys format
            sim_row = Y_sims[(Y_sims["theta_idx"] == 0) & (Y_sims["sim"] == 0)]
            sim_ys = sim_row.set_index("time")
            obs_col_map = {
                f"obs_{i}": col for i, col in enumerate(obs_col_names)
            }
            sim_ys = sim_ys.rename(columns=obs_col_map)[obs_col_names]
            # Compute probes and distance
            sim_probes = np.array([probes[k](sim_ys) for k in probe_names])
            return float(np.sum(((obs_probes - sim_probes) / scale_arr) ** 2))

        # --- Initialise chain ---
        param_names = self.canonical_param_names
        n_params = len(param_names)
        trace_names = ["distance", "log_prior"] + list(param_names)
        trace_arr = np.full((Nabc + 1, 2 + n_params), np.nan)

        theta_dict = theta_obj_in.to_list()[0]

        # Evaluate prior at starting point
        log_prior = float(_dprior(theta_dict))
        if not np.isfinite(log_prior):
            raise ValueError("Non-finite log prior at starting parameters.")

        # Distance at starting point
        new_key, sim_key = jax.random.split(new_key)
        distance = _simulate_distance(theta_dict, sim_key)

        trace_arr[0, :2] = [distance, log_prior]
        trace_arr[0, 2:] = [theta_dict[p] for p in param_names]

        accepts = 0

        # --- Main ABC-MCMC loop ---
        for n in range(1, Nabc + 1):
            new_key, prop_key, sim_key, accept_key = jax.random.split(new_key, 4)

            # Propose
            theta_prop = proposal(theta_dict, prop_key, n=n, accepts=accepts)

            # Prior ratio check (before costly simulation)
            log_prior_prop = float(_dprior(theta_prop))

            if np.isfinite(log_prior_prop):
                # MH prior ratio (symmetric proposal => just prior ratio)
                log_alpha_prior = log_prior_prop - log_prior
                u = float(jax.random.uniform(accept_key))

                if np.log(u) < log_alpha_prior:
                    # Simulate and compute distance
                    distance_prop = _simulate_distance(theta_prop, sim_key)

                    if distance_prop < epsilon ** 2:
                        theta_dict = theta_prop
                        distance = distance_prop
                        log_prior = log_prior_prop
                        accepts += 1

            # Store trace
            trace_arr[n, :2] = [distance, log_prior]
            trace_arr[n, 2:] = [theta_dict[p] for p in param_names]

            if verbose:
                print(
                    f"ABC iteration {n} of {Nabc} | "
                    f"acceptance rate: {accepts / n:.3f} | "
                    f"distance: {distance:.4f}"
                )

        # --- Package results ---
        execution_time = time.time() - start_time

        self.theta = PompParameters(theta_dict)

        result = PompABCResult(
            method="abc",
            execution_time=execution_time,
            key=old_key,
            theta=[theta_dict],
            traces_arr=trace_arr,
            trace_names=trace_names,
            Nabc=Nabc,
            epsilon=epsilon,
            accepts=accepts,
        )

        self.results_history.add(result)

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        *,
        as_pomp: Literal[True],
    ) -> "Pomp": ...

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: ThetaInput = None,
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
            theta (ThetaInput, optional): Parameters involved in the POMP model.
                Defaults to self.theta. Accepts:
                - A single dictionary: dict[str, Numeric]
                - A list of dictionaries: list[dict[str, Numeric]]
                - An existing PompParameters object
                Providing a list or PompParameters object enables faster, vectorized
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
        theta: ThetaInput = None,
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
            theta (ThetaInput, optional): Parameters to simulate from.


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
        theta: ThetaInput = None,
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
            theta (ThetaInput, optional): Parameters to use for simulation. Defaults to the first replicate in self.theta.
            show (bool): Whether to display the plot. Defaults to True.
        """
        if theta is None:
            theta = (
                self.theta.subset([0])
                if self.theta and self.theta.num_replicates() > 1
                else self.theta
            )

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
        if self.theta != other.theta:
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

        merged_theta = PompParameters.merge(*[obj._theta for obj in pomp_objs])
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
                self.fresh_key = jax.random.wrap_key_data(state["_fresh_key_data"])
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

"""
This module implements the OOP structure for POMP models.
"""

import importlib
from copy import deepcopy
import time
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .mop import _mop_internal
from .mif import _jv_mif_internal
from .train import _vmapped_train_internal
from pypomp.model_struct import RInit, RProc, DMeas, RMeas
import xarray as xr
from .simulate import _jv_simulate_internal
from .pfilter import _vmapped_pfilter_internal2
from .internal_functions import _calc_ys_covars
from .RWSigma_class import RWSigma
from .ParTrans_class import ParTrans
from .results import (
    ResultsHistory,
    PompPFilterResult,
    PompMIFResult,
    PompTrainResult,
)
from .parameters import PompParameters
from .util import logmeanexp


class Pomp:
    """
    A class representing a Partially Observed Markov Process (POMP) model.

    This class provides a structured way to define and work with POMP models, which are
    used for modeling time series data where the underlying state process is only
    partially observed. The class encapsulates the model components including the
    initial state distribution, process model, and measurement model.

    The class provides methods for:

    - Simulation of the model

    - Particle filtering

    - Maximum likelihood estimation

    - Iterated filtering

    - Model training using a differentiable particle filter

    Attributes:
        ys (pd.DataFrame): The measurement data frame with observation times as index
        theta (dict): Model parameters, where each value is a float
        rinit (RInit): Simulator for the initial state distribution
        rproc (RProc): Simulator for the process model
        dmeas (DMeas | None): Density evaluation for the measurement model
        rmeas (RMeas | None): Measurement simulator
        par_trans (ParTrans | None): Parameter transformation object
        covars (pd.DataFrame | None): Covariates for the model if applicable
        results_history (ResultsHistory | None): History of the results for the pfilter, mif, and train
            methods run on the object. This includes the algorithmic parameters used.
        fresh_key (jax.Array | None): Running a method that takes a key argument will
            store a fresh, unused key in this attribute. Subsequent calls to a method
            that requires a key will use this key unless a new key is provided as an
            argument.
    """

    ys: pd.DataFrame
    _theta: PompParameters
    canonical_param_names: list[str]
    statenames: list[str]
    t0: float
    rinit: RInit
    rproc: RProc
    dmeas: DMeas | None
    rmeas: RMeas | None
    par_trans: ParTrans
    covars: pd.DataFrame | None
    _covars_extended: np.ndarray | None
    _nstep_array: np.ndarray
    _dt_array_extended: np.ndarray
    _max_steps_per_interval: int
    ydim: int | None
    accumvars: list[str] | None
    _accumvars_indices: tuple[int, ...] | None
    results_history: ResultsHistory
    fresh_key: jax.Array | None

    def __init__(
        self,
        ys: pd.DataFrame,
        theta: dict | list[dict] | PompParameters,
        statenames: tuple[str, ...] | list[str],
        t0: float,
        rinit: Callable,
        rproc: Callable,
        dmeas: Callable | None = None,
        rmeas: Callable | None = None,
        par_trans: ParTrans | None = None,
        nstep: int | None = None,
        dt: float | None = None,
        ydim: int | None = None,
        accumvars: tuple[str, ...] | list[str] | None = None,
        covars: pd.DataFrame | None = None,
    ):
        """
        Initializes the necessary components for a specific POMP model.

        Args:
            ys (pd.DataFrame): The measurement data frame. The row index must contain the observation times.
            theta (dict or list[dict]): Parameters involved in the POMP model. Each
                value should be a float. Can be a single dict or a list of dicts.
            statenames (list[str]): List of state variable names.
            t0 (float): The initial time for the model.
            rinit (Callable): Initial state simulator function.
            rproc (Callable): Process simulator function.
            dmeas (Callable, optional): Measurement density function.
            rmeas (Callable, optional): Measurement simulator function.
            par_trans (ParTrans, optional): Parameter transformation object.
                If provided, the parameters will be transformed to and from the estimation parameter space. Defaults to the identity transformation.
            covars (pd.DataFrame, optional): Covariates or None if not applicable.
                The row index must contain the covariate times.
            nstep (int, optional): The number of steps to take for the fixedstep method.
                Must be None if dt is provided.
            dt (float, optional): The time step to use for the time_helper method.
                Must be None if nstep is provided.
            ydim (int, optional): The dimension of the measurement vector. Only
                required if rmeas is provided.
            accumvars (tuple[int, ...], optional): The indices of accumulator state
                variables. These are reset to 0 at the beginning of each observation interval.
        """
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

        if covars is not None:
            self.covar_names = list(covars.columns)
        else:
            self.covar_names = []

        self.par_trans = par_trans or ParTrans()
        self.rinit = RInit(
            struct=rinit,
            statenames=self.statenames,
            param_names=self.canonical_param_names,
            covar_names=self.covar_names,
            par_trans=self.par_trans,
        )

        self.rproc = RProc(
            struct=rproc,
            statenames=self.statenames,
            param_names=self.canonical_param_names,
            covar_names=self.covar_names,
            par_trans=self.par_trans,
            nstep=nstep,
            dt=dt,
            accumvars=self._accumvars_indices,
        )

        if dmeas is not None:
            self.dmeas = DMeas(
                struct=dmeas,
                statenames=self.statenames,
                param_names=self.canonical_param_names,
                covar_names=self.covar_names,
                par_trans=self.par_trans,
                y_names=list(self.ys.columns),
            )
        else:
            self.dmeas = None

        if rmeas is not None:
            if ydim is None:
                raise ValueError("rmeas function must have ydim attribute")
            self.rmeas = RMeas(
                struct=rmeas,
                ydim=ydim,
                statenames=self.statenames,
                param_names=self.canonical_param_names,
                covar_names=self.covar_names,
                par_trans=self.par_trans,
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
            ys=np.array(self.ys),
            ctimes=np.array(self.covars.index) if self.covars is not None else None,
            covars=np.array(self.covars) if self.covars is not None else None,
            dt=self.rproc.dt,
            nstep=self.rproc.nstep,
            order="linear",
        )
        self.rproc.rebuild_interp(self._nstep_array, self._max_steps_per_interval)

    @property
    def theta(self) -> PompParameters:
        return self._theta

    @theta.setter
    def theta(self, value: dict | list[dict] | PompParameters):
        if isinstance(value, PompParameters):
            self._theta = value
        else:
            self._theta = PompParameters(value)

    def _prepare_theta_input(
        self, theta: dict | list[dict] | PompParameters | None
    ) -> PompParameters:
        """
        Prepare the theta input for the method.
        """
        if theta is None:
            return self.theta
        elif isinstance(theta, dict) or isinstance(theta, list):
            return PompParameters(theta)
        elif isinstance(theta, PompParameters):
            return theta
        else:
            raise TypeError(
                "theta must be a dictionary, a list of dictionaries, or a PompParameters object"
            )

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

    @staticmethod
    def sample_params(param_bounds: dict, n: int, key: jax.Array) -> list[dict]:
        """
        Sample n sets of parameters from uniform distributions.

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

    def mop(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
        alpha: float = 0.97,
    ) -> list[jax.Array]:
        """
        Runs the Measurement Off-Parameter (MOP) differentiable particle filter.

        Args:
            J (int): The number of particles.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict or list[dict], optional): Parameters involved in the POMP model.
                Defaults to self.theta. Can be a single dict or a list of dicts.
            alpha (float, optional): Cooling factor for the random perturbations.
                Defaults to 0.97.

        Returns:
            list[jax.Array]: The estimated log-likelihood(s) of the observed data given the model parameters. Always a list, even if only one theta is provided.
        """
        theta_obj = self._prepare_theta_input(theta)

        theta_list = theta_obj.to_list()
        theta_list_trans = [self.par_trans.to_est(theta_i) for theta_i in theta_list]

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0")

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, len(theta_list))
        results = []
        for theta_i, k in zip(theta_list_trans, keys):
            results.append(
                -_mop_internal(
                    theta=jnp.array(
                        [theta_i[name] for name in self.canonical_param_names]
                    ),
                    ys=jnp.array(self.ys),
                    dt_array_extended=jnp.array(self._dt_array_extended),
                    nstep_array=jnp.array(self._nstep_array),
                    t0=self.t0,
                    times=jnp.array(self.ys.index),
                    J=J,
                    rinitializer=self.rinit.struct_pf,
                    rprocess_interp=self.rproc.struct_pf_interp,
                    dmeasure=self.dmeas.struct_pf,
                    covars_extended=jnp.array(self._covars_extended)
                    if self._covars_extended is not None
                    else None,
                    accumvars=self.rproc.accumvars,
                    alpha=alpha,
                    key=k,
                )
            )
        return results

    def pfilter(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: dict | list[dict] | PompParameters | None = None,
        thresh: float = 0,
        reps: int = 1,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        track_time: bool = True,
    ) -> None:
        """
        Instance method for the particle filtering algorithm.

        Args:
            J (int): The number of particles
            key (jax.Array, optional): The random key. Defaults to self.fresh_key.
            theta (dict or list[dict], optional): Parameters involved in the POMP model.
                Each value must be a float. Replaced with Pomp.theta if None. Can be a
                single dict or a list of dicts.
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
           None. Updates self.results with a dictionary containing the log-likelihoods,
           algorithmic parameters used. The conditional log-likelihoods (CLL),
           effective sample size (ESS), filtered mean, and prediction mean at each time
           point are also included if their respective boolean flags are set to True.
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
        thetas_repl = jnp.vstack(
            [jnp.tile(thetas_array[i], (reps, 1)) for i in range(n_theta_reps)]
        )

        rep_keys = jax.random.split(new_key, thetas_repl.shape[0])

        results = _vmapped_pfilter_internal2(
            thetas_repl,
            jnp.array(self._dt_array_extended),
            jnp.array(self._nstep_array),
            self.t0,
            jnp.array(self.ys.index),
            jnp.array(self.ys),
            J,
            self.rinit.struct_pf,
            self.rproc.struct_pf_interp,
            self.dmeas.struct_pf,
            self.rproc.accumvars,
            jnp.array(self._covars_extended)
            if self._covars_extended is not None
            else None,
            thresh,
            rep_keys,
            CLL,
            ESS,
            filter_mean,
            prediction_mean,
        )

        # any_diagnostics = CLL or ESS or filter_mean or prediction_mean
        neg_logliks = results["neg_loglik"]

        logLik_da = xr.DataArray(
            (-neg_logliks).reshape(n_theta_reps, reps), dims=["theta", "replicate"]
        )

        if track_time is True:
            neg_logliks.block_until_ready()
            execution_time = time.time() - start_time
        else:
            execution_time = None

        # obtain diagnostics using names
        CLL_da = None
        ESS_da = None
        filter_mean_da = None
        prediction_mean_da = None

        if CLL and "CLL" in results:
            CLL_arr = results["CLL"]
            CLL_da = xr.DataArray(
                CLL_arr.reshape(n_theta_reps, reps, -1),
                dims=["theta", "replicate", "time"],
            )

        if ESS and "ESS" in results:
            ESS_arr = results["ESS"]
            ESS_da = xr.DataArray(
                ESS_arr.reshape(n_theta_reps, reps, -1),
                dims=["theta", "replicate", "time"],
            )

        if filter_mean and "filter_mean" in results:
            filter_mean_arr = results["filter_mean"]
            filter_mean_da = xr.DataArray(
                filter_mean_arr.reshape(n_theta_reps, reps, *filter_mean_arr.shape[1:]),
                dims=["theta", "replicate", "time", "state"],
            )

        if prediction_mean and "prediction_mean" in results:
            prediction_mean_arr = results["prediction_mean"]
            prediction_mean_da = xr.DataArray(
                prediction_mean_arr.reshape(
                    n_theta_reps, reps, *prediction_mean_arr.shape[1:]
                ),
                dims=["theta", "replicate", "time", "state"],
            )

        logLik_estimates = np.apply_along_axis(
            logmeanexp, -1, (-neg_logliks).reshape(n_theta_reps, reps), ignore_nan=False
        )
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
            CLL=CLL_da,
            ESS=ESS_da,
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
        theta: dict | list[dict] | PompParameters | None = None,
        thresh: float = 0,
        track_time: bool = True,
    ) -> None:
        """
        Instance method for conducting the iterated filtering (IF2) algorithm,
        which uses the initialized instance parameters and calls the 'mif'
        function.

        Args:
            J (int): The number of particles.
            M (int): Number of algorithm iterations.
            rw_sd (RWSigma): Random walk sigma object.
            a (float): Decay factor for RWSigma over 50 iterations.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, list[dict], optional): Initial parameters for the POMP model.
                Defaults to self.theta.
            thresh (float, optional): Resampling threshold. Defaults to 0.
            track_time (bool, optional): Boolean flag controlling whether to track the
                execution time.
        Returns:
            None. Updates self.results with traces (pandas DataFrames) containing log-likelihoods and parameter estimates averaged over particles, and theta.
        """
        start_time = time.time()

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

        nLLs, theta_ests = _jv_mif_internal(
            theta_tiled,
            jnp.array(self._dt_array_extended),
            jnp.array(self._nstep_array),
            self.t0,
            jnp.array(self.ys.index),
            jnp.array(self.ys),
            self.rinit.struct_per,
            self.rproc.struct_per_interp,
            self.dmeas.struct_per,
            sigmas_array,
            sigmas_init_array,
            self.rproc.accumvars,
            jnp.array(self._covars_extended)
            if self._covars_extended is not None
            else None,
            M,
            a,
            J,
            thresh,
            keys,
        )

        final_theta_ests = []
        param_names = self.canonical_param_names
        trace_vars = ["logLik"] + param_names
        trace_data = np.zeros((n_reps, M + 1, len(trace_vars)), dtype=float)

        for i in range(n_reps):
            # Prepend nan for the log-likelihood of the initial parameters
            logliks_with_nan = np.concatenate([np.array([np.nan]), -nLLs[i]])
            # Average parameter estimates over particles for each iteration
            param_traces = np.stack(
                [
                    np.mean(theta_ests[i, :, :, j], axis=1)
                    for j in range(len(param_names))
                ],
                axis=1,
            )  # shape: (M+1, n_params)
            # Transform traces from estimation space to natural space
            param_traces = self.par_trans.transform_array(
                param_traces, param_names, direction="from_est"
            )
            trace_data[i, :, 0] = logliks_with_nan
            trace_data[i, :, 1:] = param_traces
            final_theta_ests.append(theta_ests[i])

        traces_da = xr.DataArray(
            trace_data,
            dims=["replicate", "iteration", "variable"],
            coords={
                "replicate": np.arange(n_reps),
                "iteration": np.arange(M + 1),
                "variable": trace_vars,
            },
        )

        theta = [
            self.par_trans.to_floats(
                theta=dict(
                    zip(
                        self.canonical_param_names,
                        np.mean(theta_ests[-1], axis=0).tolist(),
                    )
                ),
                direction="from_est",
            )
            for theta_ests in final_theta_ests
        ]
        logLik_estimates = -nLLs
        self.theta = PompParameters(theta, logLik=logLik_estimates)

        if track_time is True:
            nLLs.block_until_ready()
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
        )

        self.results_history.add(result)

    def train(
        self,
        J: int,
        M: int,
        eta: dict[str, float],
        key: jax.Array | None = None,
        theta: dict | list[dict] | PompParameters | None = None,
        optimizer: str = "SGD",
        alpha: float = 0.97,
        thresh: int = 0,
        scale: bool = False,
        ls: bool = False,
        c: float = 0.1,
        max_ls_itn: int = 10,
        n_monitors: int = 0,
        track_time: bool = True,
    ) -> None:
        """
        Instance method for conducting the MOP gradient-based iterative optimization method.

        Args:
            J (int): The number of particles in the MOP objective for obtaining the gradient and/or Hessian.
            M (int): Maximum iteration for the gradient descent optimization.
            eta (dict[str, float]): Learning rates per parameter as a dictionary.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            optimizer (str, optional): The gradient-based iterative optimization method
                to use. Options include "SGD", "Newton", "WeightedNewton", and "BFGS".
                Note: options other than "SGD" might be quite slow. The SGD option itself can take ~3x longer per iteration than mif does.
            alpha (float, optional): Discount factor for MOP.
            thresh (int, optional): Threshold value to determine whether to resample
                particles.
            scale (bool, optional): Boolean flag controlling whether to normalize the
                search direction.
            ls (bool, optional): Boolean flag controlling whether to use the line
                search algorithm. Note: the line search algorithm can be quite slow.
            Line Search Parameters (only used when ls=True):

                c (float, optional): The Armijo condition constant for line search which controls how much the negative log-likelihood needs to decrease before the line search algorithm continues.

                max_ls_itn (int, optional): Maximum number of iterations for the line search algorithm.

            n_monitors (int, optional): Number of particle filter runs to average for
                log-likelihood estimation.
            track_time (bool, optional): Boolean flag controlling whether to track the
                execution time.

        Returns:
            None. Updates self.results with lists for logLik, thetas_out, and theta.
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

        # Validate eta dictionary keys match canonical parameter names
        if set(eta.keys()) != set(self.canonical_param_names):
            raise ValueError(
                f"eta keys {set(eta.keys())} must match parameter names {set(self.canonical_param_names)}"
            )

        # Convert eta dict to JAX array in canonical order
        eta_array = jnp.array([eta[param] for param in self.canonical_param_names])

        new_key, old_key = self._update_fresh_key(key)
        keys = jnp.array(jax.random.split(new_key, n_reps))

        theta_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        n_obs = len(self.ys)

        nLLs, theta_ests = _vmapped_train_internal(
            theta_array,
            jnp.array(self.ys),
            jnp.array(self._dt_array_extended),
            jnp.array(self._nstep_array),
            self.t0,
            jnp.array(self.ys.index),
            self.rinit.struct_pf,
            self.rproc.struct_pf_interp,
            self.dmeas.struct_pf,
            self.rproc.accumvars,
            jnp.array(self._covars_extended)
            if self._covars_extended is not None
            else None,
            J,
            optimizer,
            M,
            eta_array,
            c,
            max_ls_itn,
            thresh,
            scale,
            ls,
            alpha,
            keys,
            n_monitors,
            n_obs,
        )

        theta_ests_natural = np.stack(
            [
                self.par_trans.transform_array(
                    theta_ests[i], self.canonical_param_names, direction="from_est"
                )
                for i in range(n_reps)
            ],
            axis=0,
        )

        joined_array = xr.DataArray(
            np.concatenate(
                [
                    -nLLs[..., np.newaxis],  # shape: (replicate, iteration, 1)
                    theta_ests_natural,  # shape: (replicate, iteration, n_theta)
                ],
                axis=-1,
            ),
            dims=["replicate", "iteration", "variable"],
            coords={
                "replicate": range(0, n_reps),
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
        logLik_estimates = -nLLs
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
            ls=ls,
            c=c,
            max_ls_itn=max_ls_itn,
        )

        self.results_history.add(result)

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: dict | list[dict] | PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulates the evolution of a system over time using a Partially Observed
        Markov Process (POMP) model.

        Args:
            key (jax.Array, optional): The random key for random number generation.
                Defaults to self.fresh_key.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            times (jax.Array, optional): Times at which to generate observations.
                Defaults to self.ys.index.
            nsim (int): The number of simulations to perform. Defaults to 1.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the simulated unobserved state values and the simulated observed values in dataframes.
            The columns are as follows:
            - replicate: The index of the parameter set.
            - sim: The index of the simulation.
            - time: The time points at which the observations were made.
            - Remaining columns contain the features of the state and observation processes.
        """
        theta_obj_in = self._prepare_theta_input(theta)

        if self.rmeas is None:
            raise ValueError(
                "self.rmeas cannot be None. Did you forget to supply it to the object or method?"
            )

        thetas_array = theta_obj_in.to_jax_array(self.canonical_param_names)

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, thetas_array.shape[0])
        times_array = jnp.array(self.ys.index) if times is None else times
        X_sims, Y_sims = _jv_simulate_internal(
            self.rinit.struct_pf,
            self.rproc.struct_pf_interp,
            self.rmeas.struct_pf,
            thetas_array,
            self.t0,
            times_array,
            jnp.array(self._dt_array_extended),
            jnp.array(self._nstep_array),
            self.rmeas.ydim,
            jnp.array(self._covars_extended)
            if self._covars_extended is not None
            else None,
            self.rproc.accumvars,
            nsim,
            keys,
        )

        def _to_long(
            arr: np.ndarray, times_vec: np.ndarray, prefix: str
        ) -> pd.DataFrame:
            vals = np.asarray(arr)  # (n_theta, n_time, n_feat, n_sim)
            n_theta_l, n_time_l, n_feat_l, n_sim_l = vals.shape
            flat = np.transpose(vals, (0, 3, 1, 2)).reshape(
                n_theta_l * n_sim_l * n_time_l, n_feat_l
            )
            theta_idx_l = np.repeat(np.arange(n_theta_l), n_sim_l * n_time_l)
            sim_idx_l = np.tile(np.repeat(np.arange(n_sim_l), n_time_l), n_theta_l)
            time_vals_l = np.tile(
                np.asarray(times_vec).reshape(1, -1), (n_theta_l * n_sim_l, 1)
            ).reshape(-1)
            cols = pd.Index([f"{prefix}_{i}" for i in range(n_feat_l)])
            df = pd.DataFrame(flat, columns=cols)
            df.insert(0, "time", time_vals_l)
            df.insert(0, "sim", sim_idx_l)
            df.insert(0, "replicate", theta_idx_l)
            return df

        times0 = np.concatenate([np.array([self.t0]), np.array(times_array)])
        X_sims_long = _to_long(X_sims, times0, "state")
        Y_sims_long = _to_long(Y_sims, np.array(times_array), "obs")

        return X_sims_long, Y_sims_long

    def traces(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the full trace of log-likelihoods and parameters from the entire result history.
        Columns:
            - replicate: The index of the parameter set (for all methods)
            - iteration: The global iteration number for that parameter set (increments over all mif/train calls for that set; for pfilter, the last iteration for that set)
            - method: 'pfilter', 'mif', or 'train'
            - logLik: The log-likelihood estimate (averaged over reps for pfilter)
            - <param>: One column for each parameter
        """
        return self.results_history.traces()

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the results of the method run at the given index.

        Args:
            index (int): The index of the result to return. Defaults to -1 (the last result).
            ignore_nan (bool): If True, ignore NaNs when computing the log-likelihood.

        Returns:
            pd.DataFrame: A DataFrame with the results of the method run at the given index.
        """
        return self.results_history.results(index=index, ignore_nan=ignore_nan)

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
        Replace self.theta with a list of the top n thetas based on the most recent available log-likelihood estimates.
        Optionally, refill the list to the previous length by repeating the top n thetas.

        Args:
            n (int): Number of top thetas to keep.
            refill (bool): If True, repeat the top n thetas to match the previous number of theta sets.
        """
        self.theta.prune(n=n, refill=refill)

    def plot_traces(self, show: bool = True) -> sns.FacetGrid | None:
        """
        Plot the parameter and log-likelihood traces from the entire result history.
        Each facet shows a parameter or logLik. The x-axis is iteration, y-axis is value.
        Lines connect mif/train points for the same replication; pfilter points are dots. Color by replication.

        Args:
            show (bool): Whether to display the plot. Defaults to True.
        """

        traces = self.traces()
        if traces.empty:
            print("No trace data to plot.")
            return
        # Melt the DataFrame to long format for FacetGrid
        value_vars = [
            col
            for col in traces.columns
            if col not in ["replicate", "iteration", "method"]
        ]
        df_long = traces.melt(
            id_vars=["replicate", "iteration", "method"],
            value_vars=value_vars,
            var_name="variable",
            value_name="value",
        )
        # Set up FacetGrid
        g = sns.FacetGrid(
            df_long,
            col="variable",
            sharex=True,
            sharey=False,
            hue="replicate",
            col_wrap=3,
            height=3.5,
            aspect=1.2,
            palette="tab10",
        )

        # Plot lines for mif/train, dots for pfilter
        def facet_plot(data, color, **kwargs):
            # Lines for mif/train
            for rep, group in data.groupby("replicate"):
                for method in ["mif", "train"]:
                    sub = group[group["method"] == method]
                    if len(sub) > 1:
                        plt.plot(
                            sub["iteration"], sub["value"], "-", color=color, alpha=0.8
                        )
                    elif len(sub) == 1:
                        plt.scatter(
                            sub["iteration"],
                            sub["value"],
                            color=color,
                            marker="o",
                            alpha=0.8,
                        )
                # Dots for pfilter
                sub = group[group["method"] == "pfilter"]
                if not sub.empty:
                    plt.scatter(
                        sub["iteration"],
                        sub["value"],
                        color=color,
                        marker="o",
                        edgecolor="k",
                        zorder=3,
                    )

        g.map_dataframe(facet_plot)
        g.add_legend(title="Replicate")
        g.set_axis_labels("Iteration", "Value")
        g.set_titles(col_template="{col_name}")
        plt.tight_layout()
        if show:
            plt.show()
        return g

    def print_summary(self):
        """
        Print a summary of the Pomp object.
        """
        print("Basics:")
        print("-------")
        print(f"Number of observations: {len(self.ys)}")
        print(f"Number of time steps: {len(self._dt_array_extended)}")
        print(f"Number of parameters: {self.theta.num_params()}")
        print()
        self.results_history.print_summary()

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
        if self.covars is not None:
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
        """Merge replications from multiple Pomp objects into a single object."""
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

        # Store function names and parameters instead of the wrapped objects
        if hasattr(self.rinit, "struct"):
            original_func = self.rinit.original_func
            state["_rinit_func_name"] = original_func.__name__
            state["_rinit_module"] = original_func.__module__

        if hasattr(self.rproc, "struct"):
            original_func = self.rproc.original_func
            state["_rproc_func_name"] = original_func.__name__
            state["_rproc_dt"] = getattr(self.rproc, "dt", None)
            state["_rproc_nstep"] = getattr(self.rproc, "nstep", None)
            state["_rproc_accumvars"] = getattr(self.rproc, "accumvars", None)
            state["_rproc_module"] = original_func.__module__

        if self.dmeas is not None and hasattr(self.dmeas, "struct"):
            original_func = self.dmeas.original_func
            state["_dmeas_func_name"] = original_func.__name__
            state["_dmeas_module"] = original_func.__module__

        if self.rmeas is not None and hasattr(self.rmeas, "struct"):
            original_func = self.rmeas.original_func
            state["_rmeas_func_name"] = original_func.__name__
            state["_rmeas_ydim"] = self.rmeas.ydim
            state["_rmeas_module"] = original_func.__module__

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
            self.fresh_key = jax.random.wrap_key_data(state["_fresh_key_data"])
        elif "fresh_key" not in self.__dict__:
            self.fresh_key = None

        # Reconstruct rinit
        if "_rinit_func_name" in state:
            module = importlib.import_module(state["_rinit_module"])
            obj = getattr(module, state["_rinit_func_name"])
            if isinstance(obj, RInit):
                self.rinit = obj
            else:
                self.rinit = RInit(
                    struct=obj,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                )

        # Reconstruct rproc
        if "_rproc_func_name" in state:
            module = importlib.import_module(state["_rproc_module"])
            obj = getattr(module, state["_rproc_func_name"])
            if isinstance(obj, RProc):
                self.rproc = obj
            else:
                kwargs = {}
                if state["_rproc_dt"] is not None:
                    kwargs["dt"] = state["_rproc_dt"]
                # If nstep is provided, but dt is not, use nstep. This prevents an error
                # being thrown by RProc when both are not None.
                if state["_rproc_nstep"] is not None and state["_rproc_dt"] is None:
                    kwargs["nstep"] = state["_rproc_nstep"]
                if state["_rproc_accumvars"] is not None:
                    kwargs["accumvars"] = state["_rproc_accumvars"]
                self.rproc = RProc(
                    struct=obj,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    **kwargs,
                )
                # Restore nstep if it was set (even if dt was originally provided)
                # This handles the case where rebuild_interp set nstep after initial construction
                if "_rproc_nstep" in state and state["_rproc_nstep"] is not None:
                    # If dt is None, nstep was already set via kwargs
                    # If dt is not None but nstep was stored, it means rebuild_interp set it
                    # In that case, restore it directly (bypassing RProc validation)
                    if state["_rproc_dt"] is not None:
                        self.rproc.nstep = state["_rproc_nstep"]

        # Reconstruct dmeas
        if "_dmeas_func_name" in state:
            module = importlib.import_module(state["_dmeas_module"])
            obj = getattr(module, state["_dmeas_func_name"])
            if isinstance(obj, DMeas):
                self.dmeas = obj
            else:
                self.dmeas = DMeas(
                    struct=obj,
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                    y_names=list(self.ys.columns) if hasattr(self, "ys") else None,
                )

        # Reconstruct rmeas
        if "_rmeas_func_name" in state:
            module = importlib.import_module(state["_rmeas_module"])
            obj = getattr(module, state["_rmeas_func_name"])
            if isinstance(obj, RMeas):
                self.rmeas = obj
            else:
                self.rmeas = RMeas(
                    struct=obj,
                    ydim=state["_rmeas_ydim"],
                    statenames=self.statenames,
                    param_names=self.canonical_param_names,
                    covar_names=self.covar_names,
                    par_trans=self.par_trans,
                )

        # Set rmeas or dmeas to None if not set
        if not hasattr(self, "rmeas"):
            self.rmeas = None
        if not hasattr(self, "dmeas"):
            self.dmeas = None

        # Clean up temporary state variables
        for key in [
            "_rinit_func_name",
            "_rinit_module",
            "_rproc_func_name",
            "_rproc_dt",
            "_rproc_nstep",
            "_rproc_accumvars",
            "_rproc_module",
            "_dmeas_func_name",
            "_dmeas_module",
            "_rmeas_func_name",
            "_rmeas_ydim",
            "_rmeas_module",
            "_fresh_key_data",
        ]:
            if key in self.__dict__:
                del self.__dict__[key]

"""
This module implements the OOP structure for POMP models.
"""

import importlib
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
from .util import logmeanexp, logmeanexp_se
from .rw_sd_class import RWSigma
from .parameter_trans import parameter_trans, materialize_partrans


def _theta_dict_to_array(theta_dict: dict, param_names: list[str]) -> jax.Array:
    """
    Convert a theta dictionary to a JAX array using canonical parameter ordering.

    Args:
        theta_dict: Dictionary mapping parameter names to values
        param_names: List of parameter names in canonical order

    Returns:
        JAX array with parameter values in canonical order
    """
    # Reorder dict to match canonical order
    ordered_values = [theta_dict[name] for name in param_names]
    return jnp.array(ordered_values)


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
        covars (pd.DataFrame | None): Covariates for the model if applicable
        results_history (list | None): History of the results for the pfilter, mif, and train
            methods run on the object. This includes the algorithmic parameters used.
        fresh_key (jax.Array | None): Running a method that takes a key argument will
            store a fresh, unused key in this attribute. Subsequent calls to a method
            that requires a key will use this key unless a new key is provided as an
            argument.
    """

    @staticmethod
    def _validate_theta(theta):
        """
        Validates that theta is a dict or a list of dicts, and all values are floats.
        Raises TypeError if invalid.
        """
        if isinstance(theta, dict):
            if not all(isinstance(val, float) for val in theta.values()):
                raise TypeError("Each value of theta must be a float")
        elif isinstance(theta, list):
            if not all(isinstance(t, dict) for t in theta):
                raise TypeError("Each element of the theta list must be a dictionary")
            for t in theta:
                if not all(isinstance(val, float) for val in t.values()):
                    raise TypeError(
                        "Each value in the theta dictionaries must be a float"
                    )
        else:
            raise TypeError("theta must be a dictionary or a list of dictionaries")

    def __init__(
        self,
        ys: pd.DataFrame,
        theta: dict | list[dict],
        statenames: list[str],
        t0: float,
        rinit: Callable,
        rproc: Callable,
        dmeas: Callable | None = None,
        rmeas: Callable | None = None,
        nstep: int | None = None,
        dt: float | None = None,
        ydim: int | None = None,
        accumvars: tuple[int, ...] | None = None,
        covars: pd.DataFrame | None = None,
        partrans=None,
        paramnames=None, 
    ):
        """
        Initializes the necessary components for a specific POMP model.

        Args:
            ys (pd.DataFrame): The measurement data frame. The row index must contain the
                observation times.
            theta (dict or list[dict]): Parameters involved in the POMP model. Each
                value should be a float. Can be a single dict or a list of dicts.
            rinit (Callable): Initial state simulator function.
            rproc (Callable): Process simulator function.
            dmeas (Callable, optional): Measurement density function.
            rmeas (Callable, optional): Measurement simulator function.
            covars (pd.DataFrame, optional): Covariates or None if not applicable.
                The row index must contain the covariate times.
            statenames (list[str], optional): List of state variable names. If None,
                will be inferred from the rinit function.
        """
        if not isinstance(ys, pd.DataFrame):
            raise TypeError("ys must be a pandas DataFrame")
        if covars is not None and not isinstance(covars, pd.DataFrame):
            raise TypeError("covars must be a pandas DataFrame or None")

        self._validate_theta(theta)

        if isinstance(theta, dict):
            self.theta = [theta]
        elif isinstance(theta, list):
            self.theta = theta
        else:
            raise TypeError("theta must be a dictionary or a list of dictionaries")

        # Extract parameter names from first theta dict
        self.param_names = list(self.theta[0].keys())

        # Validate that all theta dicts have the same keys
        for theta_dict in self.theta:
            if set(theta_dict.keys()) != set(self.param_names):
                raise ValueError("All theta dictionaries must have the same keys")

        # If statenames not provided, we need to infer them
        if statenames is None:
            raise ValueError(
                "statenames must be provided as a list of state variable names"
            )

        if not isinstance(statenames, list) or not all(
            isinstance(name, str) for name in statenames
        ):
            raise ValueError("statenames must be a list of strings")

        self.statenames = statenames
        self.ys = ys
        self.covars = covars
        self.t0 = float(t0)
        self.results_history = []
        self.fresh_key = None

        if covars is not None:
            self.covar_names = list(covars.columns)
        else:
            self.covar_names = []

        self.rinit = RInit(rinit, statenames, self.param_names, self.covar_names)
        self.rproc = RProc(
            rproc, statenames, self.param_names, self.covar_names, nstep, dt, accumvars
        )

        if dmeas is not None:
            self.dmeas = DMeas(dmeas, statenames, self.param_names, self.covar_names)
        else:
            self.dmeas = None

        if rmeas is not None:
            if ydim is None:
                raise ValueError("rmeas function must have ydim attribute")
            self.rmeas = RMeas(
                rmeas, ydim, statenames, self.param_names, self.covar_names
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
        names = paramnames or list(self.theta[0].keys())
        partrans_spec = partrans or parameter_trans()
        self.partrans = materialize_partrans(partrans_spec, names)

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
            list[jax.Array]: The estimated log-likelihood(s) of the observed data given the model
                parameters. Always a list, even if only one theta is provided.
        """
        theta = theta or self.theta
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0")

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, len(theta_list))
        results = []
        for theta_i, k in zip(theta_list, keys):
            results.append(
                -_mop_internal(
                    theta=_theta_dict_to_array(theta_i, self.param_names),
                    ys=jnp.array(self.ys),
                    dt_array_extended=self._dt_array_extended,
                    nstep_array=self._nstep_array,
                    t0=self.t0,
                    times=jnp.array(self.ys.index),
                    J=J,
                    rinitializer=self.rinit.struct_pf,
                    rprocess_interp=self.rproc.struct_pf_interp,
                    dmeasure=self.dmeas.struct_pf,
                    covars_extended=self._covars_extended,
                    accumvars=self.rproc.accumvars,
                    alpha=alpha,
                    partrans=self.partrans,
                    key=k,
                )
            )
        return results

    def pfilter(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
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

        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        thetas_repl = jnp.vstack(
            [
                jnp.tile(_theta_dict_to_array(theta_i, self.param_names), (reps, 1))
                for theta_i in theta_list
            ]
        )

        rep_keys = jax.random.split(new_key, thetas_repl.shape[0])

        results = _vmapped_pfilter_internal2(
            thetas_repl,
            self._dt_array_extended,
            self._nstep_array,
            self.t0,
            jnp.array(self.ys.index),
            jnp.array(self.ys),
            J,
            self.rinit.struct_pf,
            self.rproc.struct_pf_interp,
            self.dmeas.struct_pf,
            self.rproc.accumvars,
            self._covars_extended,
            thresh,
            rep_keys,
            CLL,
            ESS,
            filter_mean,
            prediction_mean,
        )

        # index = 0
        n_theta = len(theta_list)
        # any_diagnostics = CLL or ESS or filter_mean or prediction_mean
        neg_logliks = results["neg_loglik"]

        logLik_da = xr.DataArray(
            (-neg_logliks).reshape(n_theta, reps), dims=["theta", "replicate"]
        )

        if track_time is True:
            neg_logliks.block_until_ready()
            execution_time = time.time() - start_time
        else:
            execution_time = None

        result_dict = {
            "method": "pfilter",
            "logLiks": logLik_da,
            "theta": theta_list,
            "J": J,
            "reps": reps,
            "thresh": thresh,
            "key": old_key,
            "execution_time": execution_time,
        }

        # obtain diagnostics using names
        if CLL and "CLL" in results:
            CLL_arr = results["CLL"]
            result_dict["CLL"] = xr.DataArray(
                CLL_arr.reshape(n_theta, reps, -1), dims=["theta", "replicate", "time"]
            )

        if ESS and "ESS" in results:
            ESS_arr = results["ESS"]
            result_dict["ESS"] = xr.DataArray(
                ESS_arr.reshape(n_theta, reps, -1), dims=["theta", "replicate", "time"]
            )

        if filter_mean and "filter_mean" in results:
            filter_mean_arr = results["filter_mean"]
            result_dict["filter_mean"] = xr.DataArray(
                filter_mean_arr.reshape(n_theta, reps, *filter_mean_arr.shape[1:]),
                dims=["theta", "replicate", "time", "state"],
            )

        if prediction_mean and "prediction_mean" in results:
            prediction_mean_arr = results["prediction_mean"]
            result_dict["prediction_mean"] = xr.DataArray(
                prediction_mean_arr.reshape(
                    n_theta, reps, *prediction_mean_arr.shape[1:]
                ),
                dims=["theta", "replicate", "time", "state"],
            )

        self.results_history.append(result_dict)

    def mif(
        self,
        J: int,
        M: int,
        rw_sd: RWSigma,
        a: float,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
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

        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]
        sigmas_array, sigmas_init_array = rw_sd._return_arrays(
            param_names=self.param_names
        )
        theta_array = jnp.array(
            [_theta_dict_to_array(theta_i, self.param_names) for theta_i in theta_list]
        )

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0.")

        keys = jax.random.split(new_key, len(theta_list))

        nLLs, theta_ests = _jv_mif_internal(
            jnp.tile(theta_array, (J, 1, 1)),
            self._dt_array_extended,
            self._nstep_array,
            self.t0,
            jnp.array(self.ys.index),
            jnp.array(self.ys),
            self.rinit.struct_per,
            self.rproc.struct_per_interp,
            self.dmeas.struct_per,
            sigmas_array,
            sigmas_init_array,
            self.rproc.accumvars,
            self._covars_extended,
            self.partrans,
            M,
            a,
            J,
            thresh,
            keys,
        )

        final_theta_ests = []
        n_paramsets = len(theta_list)
        param_names = self.param_names
        trace_vars = ["logLik"] + param_names
        trace_data = np.zeros((n_paramsets, M + 1, len(trace_vars)), dtype=float)

        for i, theta_i in enumerate(theta_list):
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
            trace_data[i, :, 0] = logliks_with_nan
            trace_data[i, :, 1:] = param_traces
            final_theta_ests.append(theta_ests[i])

        traces_da = xr.DataArray(
            trace_data,
            dims=["replicate", "iteration", "variable"],
            coords={
                "replicate": np.arange(n_paramsets),
                "iteration": np.arange(M + 1),
                "variable": trace_vars,
            },
        )

        self.theta = [
            dict(zip(theta_list[0].keys(), np.mean(theta_ests[-1], axis=0).tolist()))
            for theta_ests in final_theta_ests
        ]

        if track_time is True:
            nLLs.block_until_ready()
            execution_time = time.time() - start_time
        else:
            execution_time = None

        self.results_history.append(
            {
                "method": "mif",
                "traces": traces_da,
                "theta": theta_list,
                "J": J,
                "M": M,
                "rw_sd": rw_sd,
                "a": a,
                "thresh": thresh,
                "key": old_key,
                "execution_time": execution_time,
            }
        )

    def train(
        self,
        J: int,
        M: int,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
        optimizer: str = "Newton",
        eta: float = 0.0025,
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
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            optimizer (str, optional): The gradient-based iterative optimization method
                to use. Options include "Newton", "WeightedNewton", and "BFGS".
                Defaults to "Newton".
            eta (float, optional): Learning rate.
            alpha (float, optional): Discount factor for MOP.
            thresh (int, optional): Threshold value to determine whether to resample
                particles.
            scale (bool, optional): Boolean flag controlling whether to normalize the
                search direction.
            ls (bool, optional): Boolean flag controlling whether to use the line
                search algorithm.
            Line Search Parameters (only used when ls=True):
                c (float, optional): The Armijo condition constant for line search,
                    which controls how much the negative log-likelihood needs to
                    decrease before the line search algorithm continues.
                max_ls_itn (int, optional): Maximum number of iterations for the line
                    search algorithm.
            n_monitors (int, optional): Number of particle filter runs to average for
                log-likelihood estimation.
            track_time (bool, optional): Boolean flag controlling whether to track the
                execution time.

        Returns:
            None. Updates self.results with lists for logLik, thetas_out, and theta.
        """
        start_time = time.time()

        theta = theta or self.theta
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0")

        new_key, old_key = self._update_fresh_key(key)
        rep = len(theta_list)
        keys = jax.random.split(new_key, rep)
        if getattr(keys, "ndim", 0) == 0:
            keys = keys[None]

        # Convert theta_list to array format for vmapping
        theta_array = jnp.array(
            [_theta_dict_to_array(theta_i, self.param_names) for theta_i in theta_list]
        )

        # Convert theta_list to array format for vmapping
        theta_array = jnp.stack([jnp.array(list(theta_i.values()), dtype=jnp.float32) for theta_i in theta_list],axis=0)
        n_obs = len(self.ys)

        # Use vmapped version instead of for loop
        nLLs, theta_ests = _vmapped_train_internal(
            theta_array,
            jnp.array(self.ys),
            self._dt_array_extended,
            self._nstep_array,
            self.t0,
            jnp.array(self.ys.index),
            self.rinit.struct_pf,
            self.rproc.struct_pf_interp,
            self.dmeas.struct_pf,
            self.rproc.accumvars,
            self._covars_extended,
            J,
            optimizer,
            M,
            eta,
            c,
            max_ls_itn,
            thresh,
            scale,
            ls,
            alpha,
            keys,
            n_monitors,
            n_obs,
            partrans=self.partrans,
        )

        joined_array = xr.DataArray(
            np.concatenate(
                [
                    -nLLs[..., np.newaxis],  # shape: (replicate, iteration, 1)
                    theta_ests,  # shape: (replicate, iteration, n_theta)
                ],
                axis=-1,
            ),
            dims=["replicate", "iteration", "variable"],
            coords={
                "replicate": range(0, len(theta_list)),
                "iteration": range(0, M + 1),
                "variable": ["logLik"] + self.param_names,
            },
        )

        self.theta = [
            dict(zip(self.param_names, theta_ests[i, -1, :].tolist()))
            for i in range(len(theta_list))
        ]

        if track_time is True:
            nLLs.block_until_ready()
            execution_time = time.time() - start_time
        else:
            execution_time = None

        self.results_history.append(
            {
                "method": "train",
                "traces": joined_array,
                "theta": theta_list,
                "optimizer": optimizer,
                "J": J,
                "M": M,
                "eta": eta,
                "alpha": alpha,
                "thresh": thresh,
                "ls": ls,
                "c": c,
                "max_ls_itn": max_ls_itn,
                "key": old_key,
                "execution_time": execution_time,
            }
        )

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
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
        theta = theta or self.theta
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.rmeas is None:
            raise ValueError(
                "self.rmeas cannot be None. Did you forget to supply it to the object or method?"
            )

        thetas_array = jnp.vstack(
            [_theta_dict_to_array(theta_i, self.param_names) for theta_i in theta_list]
        )

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
            self._dt_array_extended,
            self._nstep_array,
            self.rmeas.ydim,
            self._covars_extended,
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
        if not self.results_history:
            return pd.DataFrame()

        replicate_list: list[int] = []
        iteration_list: list[int] = []
        method_list: list[str] = []
        loglik_list: list[float] = []
        param_columns: dict[str, list[float]] = {}

        param_names = list(self.theta[0].keys())
        for p in param_names:
            param_columns[p] = []

        # tracks the current iteration for each replicate across rounds
        global_iters: dict[
            int, int
        ] = {}  # key: replicate index, value: current iteration (start at 1)

        for res in self.results_history:
            method = res.get("method")
            if method in ("mif", "train"):
                traces = res[
                    "traces"
                ]  # xarray.DataArray (replicate, iteration, variable)
                n_rep = traces.sizes["replicate"]
                n_iter = traces.sizes["iteration"]
                variable_names = list(traces.coords["variable"].values)

                traces_array = traces.values  # shape: (n_rep, n_iter, n_vars)
                loglik_idx = variable_names.index("logLik")
                param_indices = [variable_names.index(p) for p in param_names]

                for rep_idx in range(n_rep):
                    if rep_idx not in global_iters:
                        global_iters[rep_idx] = 0
                    for iter_idx in range(n_iter):
                        if (
                            global_iters[rep_idx] == 0 or iter_idx > 0
                        ):  # skip starting parameters beyond the the first round
                            replicate_list.append(rep_idx)
                            iteration_list.append(global_iters[rep_idx])
                            method_list.append("mif" if method == "mif" else "train")
                            loglik_list.append(
                                float(traces_array[rep_idx, iter_idx, loglik_idx])
                            )
                            for i, p in enumerate(param_names):
                                param_columns[p].append(
                                    float(
                                        traces_array[
                                            rep_idx, iter_idx, param_indices[i]
                                        ]
                                    )
                                )
                            global_iters[rep_idx] += 1
            elif method == "pfilter":
                logLiks = res["logLiks"]
                thetas = res["theta"]
                for rep_idx, (logLik_arr, theta_dict) in enumerate(
                    zip(logLiks, thetas)
                ):
                    last_iter = global_iters.get(rep_idx, 1) - 1
                    avg_loglik = float(logmeanexp(logLik_arr))

                    replicate_list.append(rep_idx)
                    iteration_list.append(last_iter if last_iter > 0 else 1)
                    method_list.append("pfilter")
                    loglik_list.append(avg_loglik)
                    for p in param_names:
                        param_columns[p].append(float(theta_dict[p]))
            # else: ignore unknown methods

        if not replicate_list:
            return pd.DataFrame()

        data = {
            "replicate": replicate_list,
            "iteration": iteration_list,
            "method": method_list,
            "loglik": loglik_list,
        }
        data.update(param_columns)

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(["iteration", "replicate"]).reset_index(drop=True)
        return df

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """
        Returns a DataFrame with the results of the method run at the given index.

        Args:
            index (int): The index of the result to return. Defaults to -1 (the last
                result).
            ignore_nan (bool): If True, ignore NaNs when computing the log-likelihood.

        Returns:
            pd.DataFrame: A DataFrame with the results of the method run at the given
                index.
        """
        res = self.results_history[index]
        method = res.get("method")
        rows = []
        param_names = list(self.theta[0].keys())
        if method == "pfilter":
            logLiks = res["logLiks"]
            thetas = res["theta"]
            for param_idx, (logLik_arr, theta_dict) in enumerate(zip(logLiks, thetas)):
                # Use underlying NumPy array if available to avoid copies
                arr = getattr(logLik_arr, "values", logLik_arr)
                logLik_arr_np = np.asarray(arr)
                logLik = float(logmeanexp(logLik_arr_np, ignore_nan=ignore_nan))
                se = (
                    float(logmeanexp_se(logLik_arr_np, ignore_nan=ignore_nan))
                    if len(logLik_arr_np) > 1
                    else np.nan
                )
                row = {"logLik": logLik, "se": se}
                row.update({param: float(theta_dict[param]) for param in param_names})
                rows.append(row)
        elif method in ("mif", "train"):
            traces = res["traces"]
            # traces is an xarray.DataArray with dims: (replicate, iteration, variable)
            n_reps = traces.sizes["replicate"]
            last_idx = traces.sizes["iteration"] - 1
            for rep in range(n_reps):
                last_row = traces.sel(replicate=rep, iteration=last_idx)
                logLik_val = float(last_row.sel(variable="logLik").values)
                row = {"logLik": logLik_val, "se": np.nan}
                for param in param_names:
                    row[param] = float(last_row.sel(variable=param).values)
                rows.append(row)
        else:
            raise ValueError(f"Unknown method in results_history: {method}")
        return pd.DataFrame(rows)

    def time(self):
        """
        Return a DataFrame summarizing the execution times of methods run.

        Returns:
            pd.DataFrame: A DataFrame where each row contains:
                - 'method': The name of the method run.
                - 'time': The execution time in seconds.
        """
        rows = []
        for idx, res in enumerate(self.results_history):
            method = res.get("method", None)
            exec_time = res.get("execution_time", None)
            rows.append({"method": method, "time": exec_time})
        df = pd.DataFrame(rows)
        df.index.name = "history_index"
        return df

    def prune(self, n: int = 1, index: int = -1, refill: bool = True):
        """
        Replace self.theta with a list of the top n thetas based on the most recent available log-likelihood estimates.
        Optionally, refill the list to the previous length by repeating the top n thetas.

        Args:
            n (int): Number of top thetas to keep.
            index (int): The index of the result to use for pruning. Defaults to -1 (the last result).
            refill (bool): If True, repeat the top n thetas to match the previous number of theta sets.
        """
        df = self.results(index)
        if df.empty or "logLik" not in df.columns:
            raise ValueError("No log-likelihoods found in results(index).")

        top_indices = df["logLik"].to_numpy().argsort()[-n:][::-1]
        # Extract the corresponding thetas as dicts
        param_names = [col for col in df.columns if col not in ("logLik", "se")]
        top_thetas = [
            {param: df.iloc[i][param] for param in param_names} for i in top_indices
        ]

        if refill:
            prev_len = len(self.theta) if isinstance(self.theta, list) else 1
            repeats = (prev_len + n - 1) // n  # Ceiling division
            new_theta = (top_thetas * repeats)[:prev_len]
        else:
            new_theta = top_thetas

        self.theta = new_theta

    def plot_traces(self, show: bool = True):
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
        allow_list = [
            "J",
            "reps",
            "thresh",
            "execution_time",
            "M",
            "a",
            "optimizer",
            "eta",
            "c",
            "max_ls_itn",
            "ls",
            "alpha",
        ]
        print("Basics:")
        print("-------")
        print(f"Number of observations: {len(self.ys)}")
        print(f"Number of time steps: {len(self._dt_array_extended)}")
        print(f"Number of parameters: {len(self.theta[0])}")
        print()
        if len(self.results_history) > 0:
            print("Results history:")
            print("----------------")
            for idx, entry in enumerate(self.results_history, 1):
                print(f"Results entry {idx}:")
                method = entry.get("method", None)
                if method is not None:
                    print(f"- method: {method}")
                for k, v in entry.items():
                    if k in allow_list:
                        print(f"- {k}: {v}")
                print()

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

        # Remove the wrapped objects from state
        state.pop("rinit", None)
        state.pop("rproc", None)
        state.pop("dmeas", None)
        state.pop("rmeas", None)

        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to reconstruct wrapped function objects. This is
        necessary because the JAX-wrapped functions are not picklable.
        """
        # Restore basic attributes
        self.__dict__.update(state)

        # Reconstruct rinit
        if "_rinit_func_name" in state:
            module = importlib.import_module(state["_rinit_module"])
            obj = getattr(module, state["_rinit_func_name"])
            if isinstance(obj, RInit):
                self.rinit = obj
            else:
                self.rinit = RInit(
                    obj, self.statenames, self.param_names, self.covar_names
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
                    obj, self.statenames, self.param_names, self.covar_names, **kwargs
                )

        # Reconstruct dmeas
        if "_dmeas_func_name" in state:
            module = importlib.import_module(state["_dmeas_module"])
            obj = getattr(module, state["_dmeas_func_name"])
            if isinstance(obj, DMeas):
                self.dmeas = obj
            else:
                self.dmeas = DMeas(
                    obj, self.statenames, self.param_names, self.covar_names
                )

        # Reconstruct rmeas
        if "_rmeas_func_name" in state:
            module = importlib.import_module(state["_rmeas_module"])
            obj = getattr(module, state["_rmeas_func_name"])
            if isinstance(obj, RMeas):
                self.rmeas = obj
            else:
                self.rmeas = RMeas(
                    obj,
                    state["_rmeas_ydim"],
                    self.statenames,
                    self.param_names,
                    self.covar_names,
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
        ]:
            if key in self.__dict__:
                del self.__dict__[key]

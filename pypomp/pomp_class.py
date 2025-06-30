"""
This module implements the OOP structure for POMP models.
"""

import importlib
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .mop import _mop_internal
from .mif import _mif_internal
from .train import _train_internal
from pypomp.model_struct import RInit, RProc, DMeas, RMeas
import xarray as xr
from .simulate import _simulate_internal
from .pfilter import _vmapped_pfilter_internal


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
        results (list | None): History of the results for the pfilter, mif, and train
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
        rinit: RInit,
        rproc: RProc,
        dmeas: DMeas | None = None,
        rmeas: RMeas | None = None,
        covars: pd.DataFrame | None = None,
    ):
        """
        Initializes the necessary components for a specific POMP model.

        Args:
            ys (pd.DataFrame): The measurement data frame. The row index must contain the
                observation times.
            rinit (RInit): Simulator for the process model.
            rproc (RProc): Basic component of the simulator for the process
                model.
            dmeas (DMeas): Basic component of the density evaluation for the
                measurement model.
            rmeas (RMeas): Measurement simulator.
            theta (dict or list[dict]): Parameters involved in the POMP model. Each
                value should be a float. Can be a single dict or a list of dicts.
            covars (pd.DataFrame, optional): Covariates or None if not applicable.
                The row index must contain the covariate times.
        """
        if not isinstance(rinit, RInit):
            raise TypeError("rinit must be an instance of the class RInit")
        if not isinstance(rproc, RProc):
            raise TypeError("rproc must be an instance of the class RProc")
        if dmeas is None and rmeas is None:
            raise ValueError("You must supply at least one of dmeas or rmeas")
        else:
            if dmeas is not None and not isinstance(dmeas, DMeas):
                raise TypeError("dmeas must be an instance of the class DMeas")
            if rmeas is not None and not isinstance(rmeas, RMeas):
                raise TypeError("rmeas must be an instance of the class RMeas")

        self._validate_theta(theta)

        if not isinstance(ys, pd.DataFrame):
            raise TypeError("ys must be a pandas DataFrame")
        if covars is not None and not isinstance(covars, pd.DataFrame):
            raise TypeError("covars must be a pandas DataFrame or None")

        self.ys = ys
        if isinstance(theta, dict):
            self.theta = [theta]
        elif isinstance(theta, list):
            self.theta = theta
        else:
            raise TypeError("theta must be a dictionary or a list of dictionaries")
        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rmeas = rmeas
        self.covars = covars
        self.results = []
        self.fresh_key = None

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
        Instance method for MOP algorithm.

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
                    theta=jnp.array(list(theta_i.values())),
                    t0=self.rinit.t0,
                    times=jnp.array(self.ys.index),
                    ys=jnp.array(self.ys),
                    J=J,
                    rinitializer=self.rinit.struct_pf,
                    rprocess=self.rproc.struct_pf,
                    dmeasure=self.dmeas.struct_pf,
                    ctimes=jnp.array(self.covars.index)
                    if self.covars is not None
                    else None,
                    covars=jnp.array(self.covars) if self.covars is not None else None,
                    alpha=alpha,
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

        Returns:
            None. Updates self.results with lists for logLik and theta.
        """
        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        keys = jax.random.split(new_key, len(theta_list))
        logLik_list = []
        for theta_i, k in zip(theta_list, keys):
            rep_keys = jax.random.split(k, reps)
            results = -_vmapped_pfilter_internal(
                jnp.array(list(theta_i.values())),
                self.rinit.t0,
                jnp.array(self.ys.index),
                jnp.array(self.ys),
                J,
                self.rinit.struct_pf,
                self.rproc.struct_pf,
                self.dmeas.struct_pf,
                jnp.array(self.covars.index) if self.covars is not None else None,
                jnp.array(self.covars) if self.covars is not None else None,
                thresh,
                rep_keys,
            )
            logLik_list.append(xr.DataArray(results, dims=["replicate"]))
        self.results.append(
            {
                "method": "pfilter",
                "logLiks": logLik_list,
                "theta": theta_list,
                "J": J,
                "thresh": thresh,
                "key": old_key,
            }
        )

    def mif(
        self,
        sigmas: float | jax.Array,
        sigmas_init: float | jax.Array,
        M: int,
        a: float,
        J: int,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
        thresh: float = 0,
    ) -> None:
        """
        Instance method for conducting the iterated filtering (IF2) algorithm,
        which uses the initialized instance parameters and calls the 'mif'
        function.

        Args:
            sigmas (float | jax.Array): Perturbation factor for parameters.
            sigmas_init (float | jax.Array): Initial perturbation factor for parameters.
            M (int): Number of algorithm iterations.
            a (float): A fraction specifying the amount to cool sigmas and sigmas_init
                over 50 iterations.
            J (int): The number of particles.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, list[dict], optional): Initial parameters for the POMP model.
                Defaults to self.theta.
            thresh (float, optional): Resampling threshold. Defaults to 0.

        Returns:
            None. Updates self.results with traces (pandas DataFrames) containing log-likelihoods
            and parameter estimates averaged over particles, and theta.
        """
        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0.")

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, len(theta_list))
        traces_list = []
        final_theta_ests = []
        for theta_i, k in zip(theta_list, keys):
            nLLs, theta_ests = _mif_internal(
                theta=jnp.array(list(theta_i.values())),
                t0=self.rinit.t0,
                times=jnp.array(self.ys.index),
                ys=jnp.array(self.ys),
                rinitializers=self.rinit.struct_per,
                rprocesses=self.rproc.struct_per,
                dmeasures=self.dmeas.struct_per,
                sigmas=sigmas,
                sigmas_init=sigmas_init,
                ctimes=jnp.array(self.covars.index)
                if self.covars is not None
                else None,
                covars=jnp.array(self.covars) if self.covars is not None else None,
                M=M,
                a=a,
                J=J,
                thresh=thresh,
                key=k,
            )
            # Prepend nan for the log-likelihood of the initial parameters
            logliks_with_nan = np.concatenate([np.array([np.nan]), -nLLs])

            # Create trace DataFrame
            param_names = list(theta_i.keys())
            trace_data = {}
            trace_data["logLik"] = logliks_with_nan

            # Average parameter estimates over particles for each iteration
            for i, param_name in enumerate(param_names):
                trace_data[param_name] = np.mean(theta_ests[:, :, i], axis=1)

            # Create DataFrame with iteration as index
            trace_df = pd.DataFrame(trace_data, index=range(0, M + 1))
            traces_list.append(trace_df)
            final_theta_ests.append(theta_ests)

        self.theta = [
            dict(zip(theta_list[0].keys(), np.mean(theta_ests[-1], axis=0).tolist()))
            for theta_ests in final_theta_ests
        ]
        self.results.append(
            {
                "method": "mif",
                "traces": traces_list,
                "theta": theta_list,
                "M": M,
                "J": J,
                "sigmas": sigmas,
                "sigmas_init": sigmas_init,
                "a": a,
                "thresh": thresh,
                "key": old_key,
            }
        )

    def train(
        self,
        J: int,
        Jh: int,
        itns: int,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
        optimizer: str = "Newton",
        beta: float = 0.9,
        eta: float = 0.0025,
        c: float = 0.1,
        max_ls_itn: int = 10,
        thresh: int = 0,
        verbose: bool = False,
        scale: bool = False,
        ls: bool = False,
        alpha: float = 0.97,
        n_monitors: int = 1,
    ) -> None:
        """
        Instance method for conducting the MOP gradient-based iterative optimization method.

        Args:
            J (int): The number of particles in the MOP objective for obtaining the gradient.
            Jh (int): The number of particles in the MOP objective for obtaining the Hessian matrix.
            itns (int): Maximum iteration for the gradient descent optimization.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            optimizer (str, optional): The gradient-based iterative optimization method to use.
                Options include "Newton", "weighted Newton", "BFGS", "gradient descent".
                Defaults to "Newton".
            beta (float, optional): Initial step size for the line search algorithm.
                Defaults to 0.9.
            eta (float, optional): Initial step size. Defaults to 0.0025.
            c (float, optional): The user-defined Armijo condition constant.
                Defaults to 0.1.
            max_ls_itn (int, optional): The maximum number of iterations for the line search algorithm.
                Defaults to 10.
            thresh (int, optional): Threshold value to determine whether to resample particles.
                Defaults to 0.
            verbose (bool, optional): Boolean flag controlling whether to print out the
                log-likelihood and parameter information. Defaults to False.
            scale (bool, optional): Boolean flag controlling whether to normalize the
                search direction. Defaults to False.
            ls (bool, optional): Boolean flag controlling whether to use the line search algorithm.
                Defaults to False.
            alpha (float, optional): Discount factor. Defaults to 0.97.
            n_monitors (int, optional): Number of particle filter runs to average for log-likelihood estimation.
                Defaults to 1.

        Returns:
            None. Updates self.results with lists for logLik, thetas_out, and theta.
        """
        theta = theta or self.theta
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")
        if J < 1:
            raise ValueError("J should be greater than 0")
        if Jh < 1:
            raise ValueError("Jh should be greater than 0")

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, len(theta_list))
        logLik_list = []
        thetas_out_list = []
        for theta_i, k in zip(theta_list, keys):
            nLLs, theta_ests = _train_internal(
                theta_ests=jnp.array(list(theta_i.values())),
                t0=self.rinit.t0,
                times=jnp.array(self.ys.index),
                ys=jnp.array(self.ys),
                rinitializer=self.rinit.struct_pf,
                rprocess=self.rproc.struct_pf,
                dmeasure=self.dmeas.struct_pf,
                J=J,
                Jh=Jh,
                ctimes=jnp.array(self.covars.index)
                if self.covars is not None
                else None,
                covars=jnp.array(self.covars) if self.covars is not None else None,
                optimizer=optimizer,
                itns=itns,
                beta=beta,
                eta=eta,
                c=c,
                max_ls_itn=max_ls_itn,
                thresh=thresh,
                verbose=verbose,
                scale=scale,
                ls=ls,
                alpha=alpha,
                key=k,
                n_monitors=n_monitors,
            )
            logLik_list.append(xr.DataArray(-nLLs, dims=["iteration"]))
            thetas_out_list.append(
                xr.DataArray(
                    theta_ests,
                    dims=["iteration", "theta"],
                    coords={
                        "iteration": range(0, itns + 1),
                        "theta": list(theta_i.keys()),
                    },
                )
            )
        self.results.append(
            {
                "method": "train",
                "logLiks": logLik_list,
                "thetas_out": thetas_out_list,
                "theta": theta_list,
                "J": J,
                "Jh": Jh,
                "optimizer": optimizer,
                "itns": itns,
                "beta": beta,
                "eta": eta,
                "c": c,
                "max_ls_itn": max_ls_itn,
                "thresh": thresh,
                "ls": ls,
                "alpha": alpha,
                "key": old_key,
            }
        )

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: dict | list[dict] | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
    ) -> list[dict]:
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
            list[dict]: A list of dictionaries each containing:
                - 'X_sims' (jax.Array): Unobserved state values with shape (n_times, n_states, nsim)
                - 'Y_sims' (jax.Array): Observed values with shape (n_times, n_obs, nsim)
        """
        theta = theta or self.theta
        self._validate_theta(theta)
        theta_list = theta if isinstance(theta, list) else [theta]

        if self.rmeas is None:
            raise ValueError(
                "self.rmeas cannot be None. Did you forget to supply it to the object or method?"
            )

        new_key, old_key = self._update_fresh_key(key)
        keys = jax.random.split(new_key, len(theta_list))
        results = []
        for theta_i, k in zip(theta_list, keys):
            times_arr = jnp.array(self.ys.index) if times is None else times
            X_sims, Y_sims = _simulate_internal(
                rinitializer=self.rinit.struct_pf,
                rprocess=self.rproc.struct_pf,
                rmeasure=self.rmeas.struct_pf,
                theta=jnp.array(list(theta_i.values())),
                t0=self.rinit.t0,
                times=jnp.array(times_arr),
                ydim=self.rmeas.ydim,
                covars=jnp.array(self.covars) if self.covars is not None else None,
                ctimes=jnp.array(self.covars.index)
                if self.covars is not None
                else None,
                nsim=nsim,
                key=k,
            )
            X_sims = xr.DataArray(
                X_sims,
                dims=["time", "element", "sim"],
                coords={
                    "time": jnp.concatenate(
                        [jnp.array([self.rinit.t0]), jnp.array(times_arr)]
                    )
                },
            )
            Y_sims = xr.DataArray(
                Y_sims, dims=["time", "element", "sim"], coords={"time": times_arr}
            )
            results.append({"X_sims": X_sims, "Y_sims": Y_sims})
        return results

    def traces(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the full trace of log-likelihoods and parameters from the entire result history.
        Columns:
            - replication: The index of the parameter set (for all methods)
            - iteration: The global iteration number for that parameter set (increments over all mif/train calls for that set; for pfilter, the last iteration for that set)
            - method: 'pfilter', 'mif', or 'train'
            - loglik: The log-likelihood estimate (averaged over reps for pfilter)
            - <param>: One column for each parameter
        """
        trace_rows = []
        param_names = None
        global_iters = {}  # key: param_idx, value: current iteration (start at 1)

        for res in self.results:
            method = res.get("method")
            if method == "mif":
                traces = res["traces"]
                for param_idx, trace_df in enumerate(traces):
                    if param_names is None:
                        param_names = [
                            col for col in trace_df.columns if col != "logLik"
                        ]
                    if param_idx not in global_iters:
                        global_iters[param_idx] = 1
                    for iter_idx, row_data in trace_df.iterrows():
                        row = {
                            "replication": param_idx,
                            "iteration": global_iters[param_idx],
                            "method": method,
                            "loglik": float(row_data["logLik"]),
                        }
                        row.update(
                            {param: float(row_data[param]) for param in param_names}
                        )
                        trace_rows.append(row)
                        global_iters[param_idx] += 1
            elif method == "train":
                logLiks = res["logLiks"]
                thetas_out = res["thetas_out"]
                for param_idx, (logLik_arr, theta_arr) in enumerate(
                    zip(logLiks, thetas_out)
                ):
                    if param_names is None:
                        if "theta" in theta_arr.coords:
                            param_names = list(theta_arr.coords["theta"].values)
                        else:
                            param_names = [
                                f"param_{j}" for j in range(theta_arr.shape[-1])
                            ]
                    if param_idx not in global_iters:
                        global_iters[param_idx] = 1
                    n_iter = logLik_arr.shape[0]
                    for iter_idx in range(n_iter):
                        theta_vals = theta_arr.isel(iteration=iter_idx).values
                        row = {
                            "replication": param_idx,
                            "iteration": global_iters[param_idx],
                            "method": method,
                            "loglik": float(logLik_arr[iter_idx].values),
                        }
                        row.update(
                            {
                                param: float(val)
                                for param, val in zip(param_names, theta_vals)
                            }
                        )
                        trace_rows.append(row)
                        global_iters[param_idx] += 1
            elif method == "pfilter":
                logLiks = res["logLiks"]
                thetas = res["theta"]
                for param_idx, (logLik_arr, theta_dict) in enumerate(
                    zip(logLiks, thetas)
                ):
                    if param_names is None:
                        param_names = list(theta_dict.keys())
                    # Use the last iteration for this param_idx
                    last_iter = global_iters.get(param_idx, 1) - 1
                    avg_loglik = float(logLik_arr.mean().values)
                    row = {
                        "replication": param_idx,
                        "iteration": last_iter if last_iter > 0 else 1,
                        "method": method,
                        "loglik": avg_loglik,
                    }
                    row.update(
                        {param: float(theta_dict[param]) for param in param_names}
                    )
                    trace_rows.append(row)
            # else: ignore other methods for now
        df = pd.DataFrame(trace_rows)
        if not df.empty:
            df = df.sort_values(["iteration", "replication"]).reset_index(drop=True)
        return df

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
            if col not in ["replication", "iteration", "method"]
        ]
        df_long = traces.melt(
            id_vars=["replication", "iteration", "method"],
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
            hue="replication",
            col_wrap=3,
            height=3.5,
            aspect=1.2,
            palette="tab10",
        )

        # Plot lines for mif/train, dots for pfilter
        def facet_plot(data, color, **kwargs):
            # Lines for mif/train
            for rep, group in data.groupby("replication"):
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
        g.add_legend(title="Replication")
        g.set_axis_labels("Iteration", "Value")
        g.set_titles(col_template="{col_name}")
        plt.tight_layout()
        if show:
            plt.show()
        return g

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
            state["_rinit_t0"] = self.rinit.t0
            state["_rinit_module"] = original_func.__module__

        if hasattr(self.rproc, "struct"):
            original_func = self.rproc.original_func
            state["_rproc_func_name"] = original_func.__name__
            state["_rproc_step_type"] = getattr(self.rproc, "step_type", "onestep")
            state["_rproc_dt"] = getattr(self.rproc, "dt", None)
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
                self.rinit = partial(RInit, t0=state["_rinit_t0"])(obj)

        # Reconstruct rproc
        if "_rproc_func_name" in state:
            module = importlib.import_module(state["_rproc_module"])
            obj = getattr(module, state["_rproc_func_name"])
            if isinstance(obj, RProc):
                self.rproc = obj
            else:
                kwargs = {"step_type": state["_rproc_step_type"]}
                if state["_rproc_dt"] is not None:
                    kwargs["dt"] = state["_rproc_dt"]
                if state["_rproc_accumvars"] is not None:
                    kwargs["accumvars"] = state["_rproc_accumvars"]
                self.rproc = partial(RProc, **kwargs)(obj)

        # Reconstruct dmeas
        if "_dmeas_func_name" in state:
            module = importlib.import_module(state["_dmeas_module"])
            obj = getattr(module, state["_dmeas_func_name"])
            if isinstance(obj, DMeas):
                self.dmeas = obj
            else:
                self.dmeas = DMeas(obj)

        # Reconstruct rmeas
        if "_rmeas_func_name" in state:
            module = importlib.import_module(state["_rmeas_module"])
            obj = getattr(module, state["_rmeas_func_name"])
            if isinstance(obj, RMeas):
                self.rmeas = obj
            else:
                self.rmeas = partial(RMeas, ydim=state["_rmeas_ydim"])(obj)

        # Clean up temporary state variables
        for key in [
            "_rinit_func_name",
            "_rinit_t0",
            "_rinit_module",
            "_rproc_func_name",
            "_rproc_step_type",
            "_rproc_dt",
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

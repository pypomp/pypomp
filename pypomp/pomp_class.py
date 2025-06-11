"""
This module implements the OOP structure for POMP models.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from .mop import _mop_internal
from .mif import _mif_internal
from .train import _train_internal
from .model_struct import RInit
from .model_struct import RProc
from .model_struct import DMeas
from .model_struct import RMeas
import xarray as xr
from .simulate import _simulate_internal
from .pfilter import _pfilter_internal


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

    def __init__(
        self,
        ys: pd.DataFrame,
        theta: dict,
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
            theta (dict): Parameters involved in the POMP model. Each value should be a
                float.
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

        if not isinstance(theta, dict):
            raise TypeError("theta must be a dictionary")
        if not all(isinstance(val, float) for val in theta.values()):
            raise TypeError("Each value of theta must be a float")

        if not isinstance(ys, pd.DataFrame):
            raise TypeError("ys must be a pandas DataFrame")
        if covars is not None and not isinstance(covars, pd.DataFrame):
            raise TypeError("covars must be a pandas DataFrame or None")

        self.ys = ys
        self.theta = theta
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
                "Both the key argument and the fresh_key attribute are None."
            )
        self.fresh_key, new_key = jax.random.split(old_key)
        return new_key, old_key

    def mop(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: dict | None = None,
        alpha: float = 0.97,
    ) -> jax.Array:
        """
        Instance method for MOP algorithm.

        Args:
            J (int): The number of particles.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            alpha (float, optional): Cooling factor for the random perturbations.
                Defaults to 0.97.

        Returns:
            jax.Array: The estimated log-likelihood of the observed data given the model
                parameters.
        """
        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0")

        if not isinstance(theta, dict):
            raise TypeError("theta must be a dictionary")
        if not all(isinstance(val, float) for val in theta.values()):
            raise TypeError("Each value of theta must be a float")

        return -_mop_internal(
            theta=jnp.array(list(theta.values())),
            t0=self.rinit.t0,
            times=jnp.array(self.ys.index),
            ys=jnp.array(self.ys),
            J=J,
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            ctimes=jnp.array(self.covars.index) if self.covars is not None else None,
            covars=jnp.array(self.covars) if self.covars is not None else None,
            alpha=alpha,
            key=new_key,
        )

    def pfilter(
        self,
        J: int,
        key: jax.Array | None = None,
        theta: dict | None = None,
        thresh: float = 0,
        reps: int = 1,
    ) -> None:
        """
        Instance method for the particle filtering algorithm.

        Args:
            J (int): The number of particles
            key (jax.Array, optional): The random key. Defaults to self.fresh_key.
            theta (dict, optional): Parameters involved in the POMP model.
                Each value must be a float. Replaced with Pomp.theta if None.
            thresh (float, optional): Threshold value to determine whether to
                resample particles. Defaults to 0.
            reps (int, optional): Number of replicates to run. Defaults to 1.

        Returns:
            None
        """
        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        if not isinstance(theta, dict):
            raise TypeError("theta must be a dictionary")
        if not all(isinstance(val, float) for val in theta.values()):
            raise TypeError("Each value of theta must be a float")

        # Generate keys for each replicate
        keys = jax.random.split(new_key, reps)
        # Run multiple replicates using a simple for loop
        results = []
        for k in keys:
            results.append(
                _pfilter_internal(
                    theta=jnp.array(list(theta.values())),
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
                    thresh=thresh,
                    key=k,
                )
            )
        results = -jnp.array(results)
        self.results.append(
            {
                "logLik": xr.DataArray(results, dims=["replicate"]),
                "theta": theta,
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
        theta: dict | None = None,
        thresh: float = 0,
        verbose: bool = False,
        n_monitors: int = 1,
    ) -> None:
        """
        Instance method for conducting the iterated filtering (IF2) algorithm,
        which uses the initialized instance parameters and calls the 'mif'
        function.

        Args:
            sigmas (float | jax.Array): Perturbation factor for parameters.
            sigmas_init (float | jax.Array): Initial perturbation factor for parameters.
            M (int): Number of algorithm iterations.
            a (float): Decay factor for sigmas.
            J (int): The number of particles.
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, optional): Initial parameters for the POMP model.
                Defaults to self.theta.
            thresh (float, optional): Resampling threshold. Defaults to 0.
            verbose (bool, optional): Flag to print log-likelihood and parameter information. Defaults to False.
            n_monitors (int, optional): Number of particle filter runs to average for log-likelihood estimation.
                Defaults to 1.

        Returns:
            None
        """
        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        if not isinstance(theta, dict):
            raise TypeError("theta must be a dictionary")
        if not all(isinstance(val, float) for val in theta.values()):
            raise TypeError("Each value of theta must be a float")

        nLLs, theta_ests = _mif_internal(
            theta=jnp.array(list(theta.values())),
            t0=self.rinit.t0,
            times=jnp.array(self.ys.index),
            ys=jnp.array(self.ys),
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            rinitializers=self.rinit.struct_per,
            rprocesses=self.rproc.struct_per,
            dmeasures=self.dmeas.struct_per,
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            ctimes=jnp.array(self.covars.index) if self.covars is not None else None,
            covars=jnp.array(self.covars) if self.covars is not None else None,
            M=M,
            a=a,
            J=J,
            thresh=thresh,
            verbose=verbose,
            key=new_key,
            n_monitors=n_monitors,
            particle_thetas=False,
        )

        self.theta = dict(zip(theta.keys(), np.mean(theta_ests[-1], axis=0).tolist()))

        self.results.append(
            {
                "logLik": xr.DataArray(-nLLs, dims=["iteration"]),
                "thetas_out": xr.DataArray(
                    theta_ests,
                    dims=["iteration", "particle", "theta"],
                    coords={
                        "iteration": range(0, M + 1),
                        "particle": range(1, J + 1),
                        "theta": list(theta.keys()),
                    },
                ),
                "theta": theta,
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
        key: jax.Array | None = None,
        theta: dict | None = None,
        method: str = "Newton",
        itns: int = 20,
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
            key (jax.Array, optional): The random key for reproducibility.
                Defaults to self.fresh_key.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            method (str, optional): The gradient-based iterative optimization method to use.
                Options include "Newton", "weighted Newton", "BFGS", "gradient descent".
                Defaults to "Newton".
            itns (int, optional): Maximum iteration for the gradient descent optimization.
                Defaults to 20.
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
            None
        """
        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0")
        if Jh < 1:
            raise ValueError("Jh should be greater than 0")

        if not isinstance(theta, dict):
            raise TypeError("theta must be a dictionary")
        if not all(isinstance(val, float) for val in theta.values()):
            raise TypeError("Each value of theta must be a float")

        nLLs, theta_ests = _train_internal(
            theta_ests=jnp.array(list(theta.values())),
            t0=self.rinit.t0,
            times=jnp.array(self.ys.index),
            ys=jnp.array(self.ys),
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            J=J,
            Jh=Jh,
            ctimes=jnp.array(self.covars.index) if self.covars is not None else None,
            covars=jnp.array(self.covars) if self.covars is not None else None,
            method=method,
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
            key=new_key,
            n_monitors=n_monitors,
        )

        self.results.append(
            {
                "logLik": xr.DataArray(-nLLs, dims=["iteration"]),
                "thetas_out": xr.DataArray(
                    theta_ests,
                    dims=["iteration", "theta"],
                    coords={
                        "iteration": range(0, itns + 1),
                        "theta": list(theta.keys()),
                    },
                ),
                "theta": theta,
                "J": J,
                "Jh": Jh,
                "method": method,
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
        theta: dict | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
    ) -> dict:
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
            dict: A dictionary containing:
                - 'X' (jax.Array): Unobserved state values with shape (n_times, n_states, nsim)
                - 'Y' (jax.Array): Observed values with shape (n_times, n_obs, nsim)
        """
        theta = theta or self.theta
        new_key, old_key = self._update_fresh_key(key)
        times = jnp.array(self.ys.index) if times is None else times

        if self.rmeas is None:
            raise ValueError(
                "self.rmeas cannot be None. Did you forget to supply it to the object or method?"
            )

        if not isinstance(theta, dict):
            raise TypeError("theta must be a dictionary")
        if not all(isinstance(val, float) for val in theta.values()):
            raise TypeError("Each value of theta must be a float")

        X_sims, Y_sims = _simulate_internal(
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            rmeasure=self.rmeas.struct_pf,
            theta=jnp.array(list(theta.values())),
            t0=self.rinit.t0,
            times=jnp.array(times),
            ydim=self.rmeas.ydim,
            covars=jnp.array(self.covars) if self.covars is not None else None,
            ctimes=jnp.array(self.covars.index) if self.covars is not None else None,
            nsim=nsim,
            key=new_key,
        )

        X_sims = xr.DataArray(
            X_sims,
            dims=["time", "element", "sim"],
            coords={
                "time": jnp.concatenate([jnp.array([self.rinit.t0]), jnp.array(times)])
            },
        )
        Y_sims = xr.DataArray(
            Y_sims, dims=["time", "element", "sim"], coords={"time": times}
        )
        return {"X_sims": X_sims, "Y_sims": Y_sims}

"""
This module implements the OOP structure for POMP models.
"""

import jax
import jax.numpy as jnp
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
            ys (DataFrame): The measurement data frame. The row index should contain the
                observation times.
            rinit (RInit): Simulator for the process model.
            rproc (RProc): Basic component of the simulator for the process
                model.
            dmeas (DMeas): Basic component of the density evaluation for the
                measurement model.
            rmeas (RMeas): Measurement simulator.
            theta (dict): Parameters involved in the POMP model. Each value should be a
                float.
            covars (array-like, optional): Covariates or None if not applicable.
                 Defaults to None.
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
            raise TypeError("covars must be a pandas DataFrame if provided")

        self.ys = ys
        self.theta = theta
        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rmeas = rmeas
        self.covars = covars

    def mop(
        self,
        J: int,
        key: jax.Array,
        theta: dict | None = None,
        ys: pd.DataFrame | None = None,
        covars: pd.DataFrame | None = None,
        alpha: float = 0.97,
    ) -> jax.Array:
        """
        Instance method for MOP algorithm.

        Args:
            J (int): The number of particles.
            key (jax.Array): The random key for reproducibility.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            ys (pd.DataFrame, optional): The measurement array.
                Defaults to self.ys.
            covars (pd.DataFrame, optional): Covariates or None if not applicable.
                Defaults to self.covars.
            alpha (float, optional): Discount factor. Defaults to 0.97.

        Returns:
            jax.Array: The estimated log-likelihood of the observed data given the model parameters.
        """
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        covars = self.covars if covars is None else covars

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
            times=jnp.array(ys.index),
            ys=jnp.array(ys),
            J=J,
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            ctimes=jnp.array(covars.index) if covars is not None else None,
            covars=jnp.array(covars) if covars is not None else None,
            alpha=alpha,
            key=key,
        )

    def pfilter(
        self,
        J: int,
        key: jax.Array,
        theta: dict | None = None,
        ys: pd.DataFrame | None = None,
        covars: pd.DataFrame | None = None,
        thresh: float = 0,
        reps: int = 1,
    ) -> jax.Array:
        """
        Instance method for particle filtering algorithm.

        Args:
            J (int): The number of particles
            key (jax.random.PRNGKey, optional): The random key.
            theta (dict, optional): Parameters involved in the POMP model.
                Each value must be a float. Replaced with Pomp.theta if None.
            ys (array-like, optional): The measurement array. Replaced with
                Pomp.ys if None.
            covars (array-like, optional): Covariates or None if not applicable.
                Replaced with Pomp.covars if None.
            thresh (float, optional): Threshold value to determine whether to
                resample particles. Defaults to 0.
            reps (int, optional): Number of replicates to run. Defaults to 1.

        Returns:
            jax.Array: The log-likelihood estimate(s).
        """
        # Use arguments instead of attributes if given
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        covars = self.covars if covars is None else covars

        if self.dmeas is None:
            raise ValueError("self.dmeas cannot be None")

        if J < 1:
            raise ValueError("J should be greater than 0.")

        if not isinstance(theta, dict):
            raise TypeError("theta must be a dictionary")
        if not all(isinstance(val, float) for val in theta.values()):
            raise TypeError("Each value of theta must be a float")

        # Generate keys for each replicate
        keys = jax.random.split(key, reps)
        # Run multiple replicates using a simple for loop
        results = []
        for k in keys:
            results.append(
                _pfilter_internal(
                    theta=jnp.array(list(theta.values())),
                    t0=self.rinit.t0,
                    times=jnp.array(ys.index),
                    ys=jnp.array(ys),
                    J=J,
                    rinitializer=self.rinit.struct_pf,
                    rprocess=self.rproc.struct_pf,
                    dmeasure=self.dmeas.struct_pf,
                    ctimes=jnp.array(covars.index) if covars is not None else None,
                    covars=jnp.array(covars) if covars is not None else None,
                    thresh=thresh,
                    key=k,
                )
            )

        return -jnp.array(results)

    def mif(
        self,
        sigmas: float | jax.Array,
        sigmas_init: float | jax.Array,
        M: int,
        a: float,
        J: int,
        key: jax.Array,
        ys: pd.DataFrame | None = None,
        theta: dict | None = None,
        covars: pd.DataFrame | None = None,
        thresh: float = 0,
        verbose: bool = False,
        n_monitors: int = 1,
    ) -> dict:
        """
        Instance method for conducting the iterated filtering (IF2) algorithm,
        which uses the initialized instance parameters and calls the 'mif'
        function.

        Args:
            sigmas (float): Perturbation factor for parameters.
            sigmas_init (float): Initial perturbation factor for parameters.
            M (int): Number of algorithm iterations.
            a (float): Decay factor for sigmas.
            J (int): The number of particles.
            key (jax.random.PRNGKey): The random key for reproducibility.
            ys (array-like, optional): The measurement array. Defaults to self.ys.
            theta (dict, optional): Initial parameters for the POMP model.
                Defaults to self.theta.
            covars (array-like, optional): Covariates or None if not applicable.
                Defaults to self.covars.
            thresh (float, optional): Resampling threshold. Defaults to 0.
            verbose (bool, optional): Flag to print log-likelihood and parameter information. Defaults to False.
            n_monitors (int, optional): Number of particle filter runs to average for log-likelihood estimation.
                Defaults to 1.

        Returns:
            dict: A dictionary containing:
                - An xarray of log-likelihood estimates through the iterations.
                - An xarray of parameters through the iterations.
        """
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        covars = self.covars if covars is None else covars

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
            times=jnp.array(ys.index),
            ys=jnp.array(ys),
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            rinitializers=self.rinit.struct_per,
            rprocesses=self.rproc.struct_per,
            dmeasures=self.dmeas.struct_per,
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            ctimes=jnp.array(covars.index) if covars is not None else None,
            covars=jnp.array(covars) if covars is not None else None,
            M=M,
            a=a,
            J=J,
            thresh=thresh,
            verbose=verbose,
            key=key,
            n_monitors=n_monitors,
            particle_thetas=False,
        )

        return {
            "logLik": xr.DataArray(-nLLs, dims=["iteration"]),
            "thetas": xr.DataArray(
                theta_ests,
                dims=["iteration", "particle", "theta"],
                coords={
                    "iteration": range(0, M + 1),
                    "particle": range(1, J + 1),
                    "theta": list(theta.keys()),
                },
            ),
        }

    def train(
        self,
        J: int,
        Jh: int,
        key: jax.Array,
        ys: pd.DataFrame | None = None,
        theta: dict | None = None,
        covars: pd.DataFrame | None = None,
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
    ) -> dict:
        """
        Instance method for conducting the MOP gradient-based iterative optimization method.

        Args:
            J (int): The number of particles in the MOP objective for obtaining the gradient.
            Jh (int): The number of particles in the MOP objective for obtaining the Hessian matrix.
            key (jax.Array): The random key for reproducibility.
            ys (pd.DataFrame, optional): The measurement array.
                Defaults to self.ys.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            covars (pd.DataFrame, optional): Covariates or None if not applicable.
                Defaults to self.covars.
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
            dict: A dictionary containing:
                - 'loglik' (xarray.DataArray): Log-likelihood values through iterations
                - 'params' (xarray.DataArray): Parameter values through iterations
        """
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        covars = self.covars if covars is None else covars

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
            times=jnp.array(ys.index),
            ys=jnp.array(ys),
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            J=J,
            Jh=Jh,
            ctimes=jnp.array(covars.index) if covars is not None else None,
            covars=jnp.array(covars) if covars is not None else None,
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
            key=key,
            n_monitors=n_monitors,
        )
        return {
            "logLik": xr.DataArray(-nLLs, dims=["iteration"]),
            "thetas": xr.DataArray(
                theta_ests,
                dims=["iteration", "theta"],
                coords={
                    "iteration": range(0, itns + 1),
                    "theta": list(theta.keys()),
                },
            ),
        }

    def simulate(
        self,
        key: jax.Array,
        theta: dict | None = None,
        times: jax.Array | None = None,
        covars: pd.DataFrame | None = None,
        nsim: int = 1,
    ) -> dict:
        """
        Simulates the evolution of a system over time using a Partially Observed
        Markov Process (POMP) model.

        Args:
            key (jax.Array): The random key for random number generation.
            theta (dict, optional): Parameters involved in the POMP model.
                If provided, overrides the unit-specific parameters.
            times (jax.Array, optional): Times at which to generate observations.
                If provided, overrides the unit-specific times.
            covars (pd.DataFrame, optional): Covariates for the process.
                If provided, overrides the unit-specific covariates.
            nsim (int, optional): The number of simulations to perform. Defaults to 1.

        Returns:
            dict: A dictionary containing:
                - 'X' (jax.Array): Unobserved state values with shape (n_times, n_states, nsim)
                - 'Y' (jax.Array): Observed values with shape (n_times, n_obs, nsim)
        """
        # Use arguments instead of attributes if given
        theta = self.theta if theta is None else theta
        times = jnp.array(self.ys.index) if times is None else times
        covars = self.covars if covars is None else covars

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
            covars=jnp.array(covars) if covars is not None else None,
            ctimes=jnp.array(covars.index) if covars is not None else None,
            nsim=nsim,
            key=key,
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

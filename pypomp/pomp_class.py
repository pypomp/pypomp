"""
This module implements the OOP structure for POMP models.
"""

import jax
import jax.numpy as jnp
import pandas as pd
from .simulate import simulate
from .mop import mop
from .pfilter import pfilter
from .mif import mif
from .train import train
from .model_struct import RInit
from .model_struct import RProc
from .model_struct import DMeas
from .model_struct import RMeas


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
        rinit: RInit | None = None,
        rproc: RProc | None = None,
        dmeas: DMeas | None = None,
        theta: dict | None = None,
        ys: pd.DataFrame | None = None,
        covars: pd.DataFrame | None = None,
        alpha: float = 0.97,
    ) -> float:
        """
        Instance method for MOP algorithm.

        Args:
            J (int): The number of particles.
            key (jax.Array): The random key for reproducibility.
            rinit (RInit, optional): Simulator for the initial-state distribution.
                Defaults to self.rinit.
            rproc (RProc, optional): Simulator for the process model.
                Defaults to self.rproc.
            dmeas (DMeas, optional): Density evaluation for the measurement model.
                Defaults to self.dmeas.
            theta (dict, optional): Parameters involved in the POMP model.
                Defaults to self.theta.
            ys (pd.DataFrame, optional): The measurement array.
                Defaults to self.ys.
            covars (pd.DataFrame, optional): Covariates or None if not applicable.
                Defaults to self.covars.
            alpha (float, optional): Discount factor. Defaults to 0.97.

        Returns:
            float: The estimated log-likelihood of the observed data given the model parameters.
        """
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        dmeas = self.dmeas if dmeas is None else dmeas
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        covars = self.covars if covars is None else covars

        if dmeas is None:
            raise ValueError("dmeas cannot be None")

        return mop(
            J=J,
            rinit=rinit,
            rproc=rproc,
            dmeas=dmeas,
            theta=theta,
            ys=ys,
            key=key,
            covars=covars,
            alpha=alpha,
        )

    def pfilter(
        self,
        J: int,
        key: jax.Array,
        theta: dict | None = None,
        ys: pd.DataFrame | None = None,
        rinit: RInit | None = None,
        rproc: RProc | None = None,
        dmeas: DMeas | None = None,
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
            rinit (RInit, optional): Simulator for the initial-state
                distribution. Replaced with Pomp.rinit if None.
            rproc (RProc, optional): Simulator for the process model.
                Replaced with Pomp.rproc if None.
            dmeas (DMeas, optional): Density evaluation for the measurement
                model. Replaced with Pomp.dmeas if None.
            covars (array-like, optional): Covariates or None if not applicable.
                Replaced with Pomp.covars if None.
            thresh (float, optional): Threshold value to determine whether to
                resample particles. Defaults to 0.
            reps (int, optional): Number of replicates to run. Defaults to 1.

        Returns:
            jax.Array: The log-likelihood estimate(s).
        """
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        dmeas = self.dmeas if dmeas is None else dmeas
        covars = self.covars if covars is None else covars

        if dmeas is None:
            raise ValueError("dmeas cannot be None")

        return pfilter(
            theta=theta,
            ys=ys,
            J=J,
            rinit=rinit,
            rproc=rproc,
            dmeas=dmeas,
            covars=covars,
            thresh=thresh,
            key=key,
            reps=reps,
        )

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
        rinit: RInit | None = None,
        rproc: RProc | None = None,
        dmeas: DMeas | None = None,
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
            rinit (RInit, optional): Simulator for the initial-state distribution.
                Defaults to self.rinit.
            rproc (RProc, optional): Simulator for the process model.
                Defaults to self.rproc.
            dmeas (DMeas, optional): Simulator for the measurement model.
                Defaults to self.dmeas.
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
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        dmeas = self.dmeas if dmeas is None else dmeas
        covars = self.covars if covars is None else covars

        if dmeas is None:
            raise ValueError("dmeas cannot be None")

        return mif(
            rinit=rinit,
            rproc=rproc,
            dmeas=dmeas,
            theta=theta,
            ys=ys,
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            covars=covars,
            M=M,
            a=a,
            J=J,
            thresh=thresh,
            verbose=verbose,
            key=key,
            n_monitors=n_monitors,
        )

    def train(
        self,
        J: int,
        Jh: int,
        key: jax.Array,
        rinit: RInit | None = None,
        rproc: RProc | None = None,
        dmeas: DMeas | None = None,
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
            rinit (RInit, optional): Simulator for the initial-state distribution.
                Defaults to self.rinit.
            rproc (RProc, optional): Simulator for the process model.
                Defaults to self.rproc.
            dmeas (DMeas, optional): Density evaluation for the measurement model.
                Defaults to self.dmeas.
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
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        dmeas = self.dmeas if dmeas is None else dmeas
        covars = self.covars if covars is None else covars

        if dmeas is None:
            raise ValueError("dmeas cannot be None")

        return train(
            theta=theta,
            ys=ys,
            rinit=rinit,
            rproc=rproc,
            dmeas=dmeas,
            covars=covars,
            J=J,
            Jh=Jh,
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

    def simulate(
        self,
        key: jax.Array,
        rinit: RInit | None = None,
        rproc: RProc | None = None,
        rmeas: RMeas | None = None,
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
            rinit (RInit, optional): Simulator for the initial-state distribution.
                If provided, overrides the unit-specific simulator.
            rproc (RProc, optional): Simulator for the process model.
                If provided, overrides the unit-specific simulator.
            rmeas (RMeas, optional): Simulator for the measurement model.
                If provided, overrides the unit-specific simulator.
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
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        rmeas = self.rmeas if rmeas is None else rmeas
        theta = self.theta if theta is None else theta
        times = jnp.array(self.ys.index) if times is None else times
        covars = self.covars if covars is None else covars

        if rmeas is None:
            raise ValueError(
                "rmeas cannot be None. Did you forget to supply it to the object or method?"
            )

        return simulate(
            rinit=rinit,
            rproc=rproc,
            rmeas=rmeas,
            theta=theta,
            times=times,
            key=key,
            covars=covars,
            nsim=nsim,
        )

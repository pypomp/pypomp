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
    MONITORS = 1

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
            J (int): The number of particles
            alpha (float, optional): Discount factor. Defaults to 0.97.
            key (jax.random.PRNGKey, optional): The random key. Defaults to
                None.

        Returns:
            float: The log-likelihood estimate
        """
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        dmeas = self.dmeas if dmeas is None else dmeas
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        covars = self.covars if covars is None else covars

        if self.dmeas is None:
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
    ) -> float:
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

        Returns:
            float: The log-likelihood estimate
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
        )

    def mif(
        self,
        sigmas: float,
        sigmas_init: float,
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
        monitor: bool = False,
        verbose: bool = False,
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
            monitor (bool, optional): Flag to monitor log-likelihood values. Defaults to False.
            verbose (bool, optional): Flag to print log-likelihood and parameter information. Defaults to False.

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
            monitor=monitor,
            verbose=verbose,
            key=key,
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
    ) -> dict:
        """
        Instance method for conducting the MOP gradient-based iterative
        optimization method.

        Args:
            theta (dict): Starting parameter values. Each value should be a float.
            J (int, optional): The number of particles in the MOP objective for
                obtaining the gradient.
            Jh (int, optional): The number of particles in the MOP objective for
                obtaining the Hessian matrix.
            method (str, optional): The gradient-based iterative optimization
                method to use, including Newton method, weighted Newton method,
                BFGS method, gradient descent.
            itns (int, optional): Maximum iteration for the gradient descent
                optimization.
            beta (float, optional): Initial step size for the line search
                algorithm.
            eta (float, optional): Initial step size.
            c (float, optional): The user-defined Armijo condition constant.
            max_ls_itn (int, optional): The maximum number of iterations for the
                line search algorithm.
            thresh (int, optional): Threshold value to determine whether to
                resample particles in pfilter function.
            verbose (bool, optional): Boolean flag controlling whether to print
                out the log-likelihood and parameter information.
            scale (bool, optional): Boolean flag controlling normalizing the
                direction or not.
            ls (bool, optional): Boolean flag controlling using the line search
                or not.
            alpha (int, optional): Discount factor.

        Returns:
            dict: a dictionary containing:
                - xarray of log-likelihood values through iterations.
                - xarray of parameters through iterations.
        """
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        dmeas = self.dmeas if dmeas is None else dmeas
        covars = self.covars if covars is None else covars

        if self.dmeas is None:
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
        Nsim: int = 1,
    ) -> dict:
        """
        Instance method for simulating a POMP model. By default, it uses this object’s
        attributes, but these can be overridden by providing them as arguments.

        Args:
            rinit (RInit, optional): Simulator for the initial-state distribution.
            rproc (RProc, optional): Simulator for the process model.
            rmeas (RMeas, optional): Simulator for the measurement model.            theta (array-like, optional): Parameters involved in the POMP model.
            times (array-like, optional): Times of the simulated observations.
            covars (array-like, optional): Covariates for the process, or None if not
                applicable.
            Nsim (int, optional): The number of simulations to perform.
            key (jax.random.PRNGKey): The random key for random number
                generation.

        Returns:
            dict: A dictionary of simulated values. 'X' contains the unobserved values
            whereas 'Y' contains the observed values as JAX arrays. In each case, the
            first dimension is the observation index, the second indexes the element of
            the observation vector, and the third is the simulation number.
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
            covars=covars,
            Nsim=Nsim,
            key=key,
        )

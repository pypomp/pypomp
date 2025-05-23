"""
This module implements the OOP structure for POMP models.
"""

import jax.numpy as jnp
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

    def __init__(self, ys, theta, rinit, rproc, dmeas=None, rmeas=None, covars=None):
        """
        Initializes the necessary components for a specific POMP model.

        Args:
            rinit (RInit): Simulator for the process model.
            rproc (RProc): Basic component of the simulator for the process
                model.
            dmeas (DMeas): Basic component of the density evaluation for the
                measurement model.
            rmeas (RMeas): Measurement simulator.
            ys (array-like): The measurement array.
            theta (dict): Parameters involved in the POMP model. Each value should be a
                float.
            covars (array-like, optional): Covariates or None if not applicable.
                 Defaults to None.

        Raises:
            TypeError: The required argument 'rinit' is not an RInit.
            TypeError: The required argument 'rproc' is not an RProc.
            ValueError: 'dmeas' and 'rmeas' are both None.
            TypeError: The required argument 'dmeas' is not a DMeas.
            TypeError: The required argument 'rmeas' is not an RMeas.
            TypeError: The required argument 'ys' is None.
            TypeError: The required argument 'theta' is None.
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

        try:
            self.ys = jnp.array(ys)
        except Exception as e:
            raise ValueError("Invalid 'ys': {}. Use an array-like.".format(e))
        try:
            self.covars = jnp.array(covars) if covars is not None else None
        except Exception as e:
            raise ValueError("Invalid 'covars': {}. Use an array-like.".format(e))
        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rmeas = rmeas
        self.theta = theta

    def mop(
        self,
        J,
        key,
        rinit=None,
        rproc=None,
        dmeas=None,
        theta=None,
        ys=None,
        covars=None,
        alpha=0.97,
    ):
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
        J,
        key,
        theta=None,
        ys=None,
        rinit=None,
        rproc=None,
        dmeas=None,
        covars=None,
        thresh=0,
    ):
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

        if self.dmeas is None:
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
        sigmas,
        sigmas_init,
        M,
        a,
        J,
        key,
        ys=None,
        theta=None,
        rinit=None,
        rproc=None,
        dmeas=None,
        covars=None,
        thresh=0,
        monitor=False,
        verbose=False,
    ):
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

        if self.dmeas is None:
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
        J,
        Jh,
        key,
        rinit=None,
        rproc=None,
        dmeas=None,
        ys=None,
        theta=None,
        covars=None,
        method="Newton",
        itns=20,
        beta=0.9,
        eta=0.0025,
        c=0.1,
        max_ls_itn=10,
        thresh=0,
        verbose=False,
        scale=False,
        ls=False,
        alpha=0.97,
    ):
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
        key,
        rinit=None,
        rproc=None,
        rmeas=None,
        theta=None,
        ylen=None,
        covars=None,
        Nsim=1,
    ):
        """
        Instance method for simulating a POMP model. By default, it uses this object’s
        attributes, but these can be overridden by providing them as arguments.

        Args:
            rinit (RInit, optional): Simulator for the initial-state distribution.
            rproc (RProc, optional): Simulator for the process model.
            rmeas (RMeas, optional): Simulator for the measurement model.            theta (array-like, optional): Parameters involved in the POMP model.
            ylen (int, optional): The number of observations to generate in one time
                series. Defaults to None, in which case simulate uses the length of the
                time series stored in the Pomp object.
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
        rproc = self.rproc if rproc is None else rinit
        rmeas = self.rmeas if rmeas is None else rinit
        theta = self.theta if theta is None else rinit
        ylen = len(self.ys) if ylen is None else ylen
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
            ylen=ylen,
            covars=covars,
            Nsim=Nsim,
            key=key,
        )

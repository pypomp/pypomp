"""
This module implements the OOP structure for POMP models.
"""

from .simulate import simulate
from .mop import _mop_internal
from .pfilter import pfilter
from .mif import mif
from .train import _train_internal
from .fit import _fit_internal
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
            theta (array-like): Parameters involved in the POMP model.
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

        if ys is None:
            raise TypeError("ys cannot be None")
        if theta is None:
            raise TypeError("theta cannot be None")

        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.rmeas = rmeas
        self.ys = ys
        self.theta = theta
        self.covars = covars

    def mop(self, J, alpha=0.97, key=None):
        """
        Instance method for MOP algorithm, which uses the initialized instance
            parameters and calls 'mop_internal' function.

        Args:
            J (int): The number of particles
            alpha (float, optional): Discount factor. Defaults to 0.97.
            key (jax.random.PRNGKey, optional): The random key. Defaults to
                None.

        Returns:
            float: Negative log-likelihood value
        """
        if J < 1:
            raise ValueError("J should be greater than 0")
        if self.dmeas is None:
            raise ValueError("dmeas cannot be None")
        return _mop_internal(
            theta=self.theta,
            ys=self.ys,
            J=J,
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            covars=self.covars,
            alpha=alpha,
            key=key,
        )

    def pfilter(
        self,
        J,
        theta=None,
        ys=None,
        rinit=None,
        rproc=None,
        dmeas=None,
        covars=None,
        thresh=0,
        key=None,
    ):
        """
        Instance method for particle filtering algorithm, which uses the
        initialized instance parameters and calls 'pfilter_internal' function.

        Args:
            J (int): The number of particles
            thresh (float, optional): Threshold value to determine whether to
                resample particles. Defaults to 0.
            key (jax.random.PRNGKey, optional): The random key. Defaults to
                None.

        Returns:
            float: Negative log-likelihood value
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
        ys=None,
        theta=None,
        rinit=None,
        rproc=None,
        dmeas=None,
        covars=None,
        thresh=0,
        monitor=False,
        verbose=False,
        key=None,
    ):
        """
        Instance method for conducting iterated filtering (IF2) algorith, which
        uses the initialized instance parameters and calls 'mif_internal'
        function.

        Args:
            sigmas (float): Perturbed factor
            sigmas_init (float): Initial perturbed factor
            M (int, optional): Algorithm Iteration.
            a (float, optional): Decay factor for sigmas.
            J (int, optional): The number of particles. Defaults to 100.
            thresh (float, optional): Threshold value to determine whether to
                resample particles.
            monitor (bool, optional): Boolean flag controlling whether to
                monitor the log-likelihood value.
            verbose (bool, optional): Boolean flag controlling whether to print
                out the log-likelihood and parameter information.
        Returns:
            tuple: A tuple containing:
            - An array of negative log-likelihood through the iterations
            - An array of parameters through the iterations
        """
        theta = self.theta if theta is None else theta
        ys = self.ys if ys is None else ys
        rinit = self.rinit if rinit is None else rinit
        rproc = self.rproc if rproc is None else rproc
        dmeas = self.dmeas if dmeas is None else dmeas
        covars = self.covars if covars is None else covars
        if J < 1:
            raise ValueError("J should be greater than 0")
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
        theta_ests,
        J=5000,
        Jh=1000,
        method="Newton",
        itns=20,
        beta=0.9,
        eta=0.0025,
        c=0.1,
        max_ls_itn=10,
        thresh=100,
        verbose=False,
        scale=False,
        ls=False,
        alpha=1,
        key=None,
    ):
        """
        Instance method for conducting the MOP gradient-based iterative
        optimization method, which uses the initialized instance parameters and
        calls 'train_internal' function.

        Args:
            theta_ests (array-like): Initial value of parameter values before
                gradient descent.
            J (int, optional): The number of particles in the MOP objective for
                obtaining the gradient. Defaults to 5000.
            Jh (int, optional): The number of particles in the MOP objective for
                obtaining the Hessian matrix. Defaults to 1000.
            method (str, optional): The gradient-based iterative optimization
                method to use, including Newton method, weighted Newton method
                BFGS method, gradient descent. Defaults to 'Newton'.
            itns (int, optional): Maximum iteration for the gradient descent
                optimization. Defaults to 20.
            beta (float, optional): Initial step size for the line search
                algorithm. Defaults to 0.9.
            eta (float, optional): Initial step size. Defaults to 0.0025.
            c (float, optional): The user-defined Armijo condition constant.
                Defaults to 0.1.
            max_ls_itn (int, optional): The maximum number of iterations for the
                line search algorithm. Defaults to 10.
            thresh (int, optional): Threshold value to determine whether to
                resample particles in pfilter function. Defaults to 100.
            verbose (bool, optional): Boolean flag controlling whether to print
                out the log-likelihood and parameter information. Defaults to
                False.
            scale (bool, optional): Boolean flag controlling normalizing the
                direction or not. Defaults to False.
            ls (bool, optional): Boolean flag controlling using the line search
                or not. Defaults to False.
            alpha (int, optional): Discount factor. Defaults to 1.

        Returns:
            tuple: A tuple containing:
            - An array of negative log-likelihood through the iterations
            - An array of parameters through the iterations
        """
        if J < 1:
            raise ValueError("J should be greater than 0")
        if Jh < 1:
            raise ValueError("Jh should be greater than 0")
        if self.dmeas is None:
            raise ValueError("dmeas cannot be None")
        return _train_internal(
            theta_ests=theta_ests,
            ys=self.ys,
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            covars=self.covars,
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

    def fit(
        self,
        sigmas=None,
        sigmas_init=None,
        M=10,
        a=0.9,
        J=100,
        Jh=1000,
        method="Newton",
        itns=20,
        beta=0.9,
        eta=0.0025,
        c=0.1,
        max_ls_itn=10,
        thresh_mif=100,
        thresh_tr=100,
        verbose=False,
        scale=False,
        ls=False,
        alpha=0.1,
        monitor=True,
        mode="IFAD",
        key=None,
    ):
        """
        Instance method for executing the iterated filtering (IF2), MOP
        gradient-based iterative optimization method (GD), and iterated
        filtering with automatic differentiation (IFAD), which uses the
        initialized instance parameters and calls 'fit_internal' function.

        Args:
            sigmas (float, optional): Perturbed factor. Defaults to None.
            sigmas_init (float, optional): Initial perturbed factor. Defaults to
                None.
            M (int, optional): Maximum algorithm iteration for iterated
                filtering. Defaults to 10.
            a (float, optional): Decay factor for sigmas. Defaults to 0.9.
            J (int, optional): The number of particles in iterated filtering and
                the number of particles in the MOP objective for obtaining the
                gradient in gradient-based optimization procedure. Defaults to
                100.
            Jh (int, optional): The number of particles in the MOP objective for
                obtaining the Hessian matrix. Defaults to 1000.
            method (str, optional): The gradient-based iterative optimization
                method to use, including Newton method, weighted Newton method,
                BFGS method and gradient descent. Defaults to 'Newton'.
            itns (int, optional): Maximum iteration for the gradient
                optimization. Defaults to 20.
            beta (float, optional): Initial step size. Defaults to 0.9.
            eta (float, optional): Initial step size. Defaults to 0.0025.
            c (float, optional): The user-defined Armijo condition constant.
                Defaults to 0.1.
            max_ls_itn (int, optional): The maximum number of iterations for the
                line search algorithm. Defaults to 10.
            thresh_mif (int, optional): Threshold value to determine whether to
                resample particles in iterated filtering. Defaults to 100.
            thresh_tr (int, optional): Threshold value to determine whether to
                resample particles in gradient optimization. Defaults to 100.
            verbose (bool, optional):  Boolean flag controlling whether to print
                out the log-likelihood and parameter information. Defaults to
                False.
            scale (bool, optional): Boolean flag controlling normalizing the
                direction or not. Defaults to False.
            ls (bool, optional): Boolean flag controlling using the line search
                or not. Defaults to False.
            alpha (float, optional): Discount factor. Defaults to 0.1.
            monitor (bool, optional): Boolean flag controlling whether to
                monitor the log-likelihood value. Defaults to True.
            mode (str, optional): The optimization algorithm to use, including
                'IF2', 'GD', and 'IFAD'. Defaults to "IFAD".

        Returns:
            tuple: A tuple containing:
            - An array of negative log-likelihood through the iterations
            - An array of parameters through the iterations
        """
        if J < 1:
            raise ValueError("J should be greater than 0")
        if Jh < 1:
            raise ValueError("Jh should be greater than 0")
        if self.dmeas is None:
            raise ValueError("dmeas cannot be None")
        return _fit_internal(
            theta=self.theta,
            ys=self.ys,
            rinitializer=self.rinit.struct_pf,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            rinitializers=self.rinit.struct_per,
            rprocesses=self.rproc.struct_per,
            dmeasures=self.dmeas.struct_per,
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            covars=self.covars,
            M=M,
            a=a,
            J=J,
            Jh=Jh,
            method=method,
            itns=itns,
            beta=beta,
            eta=eta,
            c=c,
            max_ls_itn=max_ls_itn,
            thresh_mif=thresh_mif,
            thresh_tr=thresh_tr,
            verbose=verbose,
            scale=scale,
            ls=ls,
            alpha=alpha,
            monitor=monitor,
            mode=mode,
            key=key,
        )

    def simulate(
        self,
        rinit=None,
        rproc=None,
        rmeas=None,
        theta=None,
        ylen=None,
        covars=None,
        Nsim=1,
        key=None,
    ):
        """
        Instance method for simulating a POMP model. By default, it uses this object’s
        attributes, but these can be overridden by providing them as arguments.

        Args:
            rinit (RInit, optional): Simulator for the initial-state distribution.
                Defaults to None.
            rproc (RProc, optional): Simulator for the process model. Defaults to None.
            rmeas (RMeas, optional): Simulator for the measurement model. Defaults to
                None.
            theta (array-like, optional): Parameters involved in the POMP model.
                Defaults to None.
            ylen (int, optional): The number of observations to generate in one time
                series. Defaults to None, in which case simulate uses the length of the
                time series stored in the Pomp object.
            covars (array-like, optional): Covariates for the process, or None if not
                applicable. Defaults to None.
            Nsim (int, optional): The number of simulations to perform. Defaults to 1.
            key (jax.random.PRNGKey, optional): The random key for random number
                generation.

        Returns:
            dict: A dictionary of simulated values. 'X' contains the unobserved values
                whereas 'Y' contains the observed values.
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

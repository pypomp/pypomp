"""
This module implements the OOP structure for POMP models.
"""

from .internal_functions import _mop_internal
from .internal_functions import _pfilter_internal
from .internal_functions import _mif_internal
from .internal_functions import _train_internal
from .internal_functions import _fit_internal
from .model_struct import RInit
from .model_struct import RProc
from .model_struct import DMeas


class Pomp:
    MONITORS = 1

    def __init__(self, rinit, rproc, dmeas, ys, theta, covars=None):
        """
        Initializes the necessary components for a specific POMP model.

        Args:
            rinit (RInit): Simulator for the process model.
            rproc (RProc): Basic component of the simulator for the process
                model.
            dmeas (DMeas): Basic component of the density evaluation for the
                measurement model.
            ys (array-like): The measurement array.
            theta (array-like): Parameters involved in the POMP model.
            covars (array-like, optional): Covariates or None if not applicable.
                 Defaults to None.

        Raises:
            TypeError: The required argument 'rinit' is not an RInit.
            TypeError: The required argument 'rproc' is not an RProc.
            TypeError: The required argument 'dmeas' is not a DMeas.
            TypeError: The required argument 'ys' is None.
            TypeError: The required argument 'theta' is None.
        """
        if not isinstance(rinit, RInit):
            raise TypeError("rinit must be an instance of the class RInit")
        if not isinstance(rproc, RProc):
            raise TypeError("rproc must be an instance of the class RProc")
        if not isinstance(dmeas, DMeas):
            raise TypeError("dmeas must be an instance of the class DMeas")
        if ys is None:
            raise TypeError("ys cannot be None")
        if theta is None:
            raise TypeError("theta cannot be None")

        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.ys = ys
        self.theta = theta
        self.covars = covars

    # def rinits(self, thetas, J, covars):
    #     return rinits_internal(self.rinit, thetas, J, covars)

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

        return _mop_internal(
            theta = self.theta,
            ys = self.ys,
            J = J,
            rinit = self.rinit.struct,
            rprocess = self.rproc.struct_pf,
            dmeasure = self.dmeas.struct_pf,
            covars = self.covars,
            alpha = alpha,
            key=key,
        )

    def pfilter(self, J, thresh=100, key=None):
        """
        Instance method for particle filtering algorithm, which uses the
        initialized instance parameters and calls 'pfilter_internal' function.

        Args:
            J (int): The number of particles
            thresh (float, optional): Threshold value to determine whether to
                resample particles. Defaults to 100.
            key (jax.random.PRNGKey, optional): The random key. Defaults to
                None.

        Returns:
            float: Negative log-likelihood value
        """

        return _pfilter_internal(
            theta = self.theta,
            ys = self.ys,
            J = J,
            rinit = self.rinit.struct,
            rprocess = self.rproc.struct_pf,
            dmeasure = self.dmeas.struct_pf,
            covars = self.covars,
            thresh = thresh,
            key = key,
        )

    def mif(
        self,
        sigmas,
        sigmas_init,
        M=10,
        a=0.9,
        J=100,
        thresh=100,
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
            M (int, optional): Algorithm Iteration. Defaults to 10.
            a (float, optional): Decay factor for sigmas. Defaults to 0.95.
            J (int, optional): The number of particles. Defaults to 100.
            thresh (float, optional): Threshold value to determine whether to
                resample particles. Defaults to 100.
            monitor (bool, optional): Boolean flag controlling whether to
                monitor the log-likelihood value. Defaults to False.
            verbose (bool, optional): Boolean flag controlling whether to print
                out the log-likelihood and parameter information. Defaults to
                False.
        Returns:
            tuple: A tuple containing:
            - An array of negative log-likelihood through the iterations
            - An array of parameters through the iterations
        """

        return _mif_internal(
            theta=self.theta,
            ys=self.ys,
            rinit=self.rinit.struct,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
            rprocesses=self.rproc.struct_per,
            dmeasures=self.dmeas.struct_per,
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            covars=self.covars,
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

        return _train_internal(
            theta_ests=theta_ests,
            ys=self.ys,
            rinit=self.rinit.struct,
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

        return _fit_internal(
            theta=self.theta,
            ys=self.ys,
            rinit=self.rinit.struct,
            rprocess=self.rproc.struct_pf,
            dmeasure=self.dmeas.struct_pf,
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

"""
This module implements the OOP structure for PanelPOMP models.
"""

from .pomp_class import Pomp
import jax


class PanelPomp:
    def __init__(self, Pomp_dict):
        """
        Initializes the necessary components for a PanelPOMP model.

        Args:
            Pomp_dict: A dictionary of Pomp objects. Each key is used as the name of
                the corresponding unit.
        """
        if not isinstance(Pomp_dict, dict):
            raise TypeError("Pomp_dict must be an instance of the class RInit")
        for value in Pomp_dict.values():
            if not isinstance(value, Pomp):
                raise TypeError(
                    "Every element of Pomp_dict must be an instance of the class Pomp"
                )

        self.unit_objects = Pomp_dict
        # self.shared =
        # self.specific =

    def simulate(
        self,
        key,
        rinit=None,
        rproc=None,
        rmeas=None,
        theta=None,
        times=None,
        covars=None,
        Nsim=1,
    ):
        """
        Simulate the PanelPOMP model.

        This method applies the simulate method to each of the individual Pomp
        models. Returns a dictionary of simulated values. Each key is the name of
        a unit, and the value is the output of the simulate method for that unit.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The random key for random number generation.
        rinit : RInit, optional
            Simulator for the initial-state distribution. Defaults to None.
        rproc : RProc, optional
            Simulator for the process model. Defaults to None.
        rmeas : RMeas, optional
            Simulator for the measurement model. Defaults to None.
        theta : array-like, optional
            Parameters involved in the POMP model. Defaults to None.
        times : jax.Array, optional
            The times at which to generate observations. Defaults to None, in which
            case simulate uses the times stored in the Pomp object.
        covars : array-like, optional
            Covariates for the process, or None if not applicable. Defaults to
            None.
        Nsim : int, optional
            The number of simulations to perform. Defaults to 1.

        Returns
        -------
        dict
            A dictionary of simulated values. Each key is the name of a unit,
            and the value is the output of the simulate method for that unit.
        """
        results = {}
        for unit, obj in self.unit_objects.items():
            key, subkey = jax.random.split(key)
            results[unit] = obj.simulate(
                key=subkey,
                rinit=rinit,
                rproc=rproc,
                rmeas=rmeas,
                theta=theta,
                times=times,
                covars=covars,
                Nsim=Nsim,
            )
        return results

    def pfilter(
        self,
        J,
        key=None,
        theta=None,
        ys=None,
        rinit=None,
        rproc=None,
        dmeas=None,
        covars=None,
        thresh=0,
    ):
        """
        Run the pfilter method on the individual Pomp models.

        Parameters
        ----------
        J : int
            The number of particles
        key : jax.random.PRNGKey, optional
            The random key for random number generation. Defaults to None.
        theta : array-like, optional
            Parameters involved in the POMP model. Defaults to None.
        ys : array-like, optional
            The measurement array. Defaults to None.
        rinit : RInit, optional
            Simulator for the initial-state distribution. Defaults to None.
        rproc : RProc, optional
            Simulator for the process model. Defaults to None.
        dmeas : DMeas, optional
            Simulator for the measurement model. Defaults to None.
        covars : array-like, optional
            Covariates for the process, or None if not applicable. Defaults to
            None.
        thresh : float, optional
            Threshold value to determine whether to resample particles.
            Defaults to 0.

        Returns
        -------
        dict
            A dictionary of particle filter results. Each key is the name of a
            unit, and the value is the output of the pfilter method for that
            unit.
        """
        results = {}
        for unit, obj in self.unit_objects.items():
            key, subkey = jax.random.split(key)
            results[unit] = obj.pfilter(
                J=J,
                key=subkey,
                theta=theta,
                ys=ys,
                rinit=rinit,
                rproc=rproc,
                dmeas=dmeas,
                covars=covars,
                thresh=thresh,
            )
        return results

    def mif(
        self,
        J,
        key,
        sigmas,
        sigmas_init,
        M,
        a,
        theta=None,
        ys=None,
        rinit=None,
        rproc=None,
        dmeas=None,
        covars=None,
        thresh=0,
        monitor=False,
        verbose=False,
    ):
        """
        Run the mif method on the individual Pomp models.

        Parameters
        ----------
        J : int
            The number of particles.
        key : jax.random.PRNGKey
            The random key for random number generation.
        sigmas : float
            Perturbation factor for parameters.
        sigmas_init : float
            Initial perturbation factor for parameters.
        M : int
            Number of algorithm iterations.
        a : float
            Decay factor for sigmas.
        theta : array-like, optional
            Initial parameters for the POMP model.
        ys : array-like, optional
            The measurement array.
        rinit : RInit, optional
            Simulator for the initial-state distribution.
        rproc : RProc, optional
            Simulator for the process model.
        dmeas : DMeas, optional
            Simulator for the measurement model.
        covars : array-like, optional
            Covariates or None if not applicable.
        thresh : float, optional
            Resampling threshold.
        monitor : bool, optional
            Flag to monitor log-likelihood values.
        verbose : bool, optional
            Flag to print log-likelihood and parameter information.

        Returns
        -------
        dict
            A dictionary of mif results. Each key is the name of a unit, and the value is the output of the mif method for that unit.
        """
        results = {}
        for unit, obj in self.unit_objects.items():
            key, subkey = jax.random.split(key)
            results[unit] = obj.mif(
                J=J,
                key=subkey,
                theta=theta,
                ys=ys,
                rinit=rinit,
                rproc=rproc,
                dmeas=dmeas,
                covars=covars,
                sigmas=sigmas,
                sigmas_init=sigmas_init,
                M=M,
                a=a,
                thresh=thresh,
                monitor=monitor,
                verbose=verbose,
            )
        return results

"""
This module implements the OOP structure for PanelPOMP models.
"""

import jax
import pandas as pd
from .pomp_class import Pomp
from .model_struct import RInit, RProc, DMeas, RMeas


class PanelPomp:
    def __init__(self, Pomp_dict):
        """
        Initializes a PanelPOMP model, which consists of multiple POMP models
        (units) that share the same structure but may have different parameters
        and observations.

        Args:
            Pomp_dict (dict): A dictionary mapping unit names to Pomp objects.
                Each Pomp object represents a single unit in the panel data.
                The keys are used as unit identifiers.
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
        Simulates the PanelPOMP model by applying the simulate method to each unit.

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
            dict: A dictionary mapping unit names to simulation results. Each result
                is a dictionary containing:
                - 'X' (jax.Array): Unobserved state values with shape (n_times, n_states, nsim)
                - 'Y' (jax.Array): Observed values with shape (n_times, n_obs, nsim)
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
                nsim=nsim,
            )
        return results

    def pfilter(
        self,
        J: int,
        key: jax.Array,
        theta: dict | None = None,
        ys: jax.Array | None = None,
        rinit: RInit | None = None,
        rproc: RProc | None = None,
        dmeas: DMeas | None = None,
        covars: pd.DataFrame | None = None,
        thresh: float = 0,
    ) -> dict:
        """
        Runs the particle filter on each unit in the panel.

        Args:
            J (int): The number of particles to use in the filter.
            key (jax.Array): The random key for reproducibility.
            theta (dict, optional): Parameters involved in the POMP model.
                If provided, overrides the unit-specific parameters.
            ys (pd.DataFrame, optional): The measurement array.
                If provided, overrides the unit-specific measurements.
            rinit (RInit, optional): Simulator for the initial-state distribution.
                If provided, overrides the unit-specific simulator.
            rproc (RProc, optional): Simulator for the process model.
                If provided, overrides the unit-specific simulator.
            dmeas (DMeas, optional): Density evaluation for the measurement model.
                If provided, overrides the unit-specific evaluator.
            covars (pd.DataFrame, optional): Covariates for the process.
                If provided, overrides the unit-specific covariates.
            thresh (float, optional): Threshold value to determine whether to resample particles.
                Defaults to 0.

        Returns:
            dict: A dictionary mapping unit names to log-likelihood estimates.
                Each value is the estimated log-likelihood for that unit.
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
        J: int,
        key: jax.Array,
        sigmas: float,
        sigmas_init: float,
        M: int,
        a: float,
        theta: dict | None = None,
        ys: jax.Array | None = None,
        rinit: RInit | None = None,
        rproc: RProc | None = None,
        dmeas: DMeas | None = None,
        covars: pd.DataFrame | None = None,
        thresh: float = 0,
        monitor: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Runs the iterated filtering (IF2) algorithm on each unit in the panel.

        Args:
            J (int): The number of particles.
            key (jax.Array): The random key for reproducibility.
            sigmas (float): Perturbation factor for parameters.
            sigmas_init (float): Initial perturbation factor for parameters.
            M (int): Number of algorithm iterations.
            a (float): Decay factor for sigmas.
            theta (dict, optional): Initial parameters for the POMP model.
                If provided, overrides the unit-specific parameters.
            ys (pd.DataFrame, optional): The measurement array.
                If provided, overrides the unit-specific measurements.
            rinit (RInit, optional): Simulator for the initial-state distribution.
                If provided, overrides the unit-specific simulator.
            rproc (RProc, optional): Simulator for the process model.
                If provided, overrides the unit-specific simulator.
            dmeas (DMeas, optional): Density evaluation for the measurement model.
                If provided, overrides the unit-specific evaluator.
            covars (pd.DataFrame, optional): Covariates for the process.
                If provided, overrides the unit-specific covariates.
            thresh (float, optional): Resampling threshold. Defaults to 0.
            monitor (bool, optional): Flag to monitor log-likelihood values. Defaults to False.
            verbose (bool, optional): Flag to print log-likelihood and parameter information.
                Defaults to False.

        Returns:
            dict: A dictionary mapping unit names to MIF results. Each result is a dictionary
                containing:
                - 'loglik' (xarray.DataArray): Log-likelihood values through iterations
                - 'params' (xarray.DataArray): Parameter values through iterations
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

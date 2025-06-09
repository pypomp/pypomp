"""
This module implements the OOP structure for PanelPOMP models.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
from tqdm import tqdm
from .pomp_class import Pomp
from .model_struct import RInit, RProc, DMeas, RMeas
from .mif import _mif_internal


class PanelPomp:
    def __init__(
        self,
        Pomp_dict: dict[str, Pomp],
        shared: pd.DataFrame,
        unit_specific: pd.DataFrame,
    ):
        """
        Initializes a PanelPOMP model, which consists of multiple POMP models
        (units) that share the same structure but may have different parameters
        and observations.

        Args:
            Pomp_dict (dict[str, Pomp]): A dictionary mapping unit names to Pomp objects.
                Each Pomp object represents a single unit in the panel data.
                The keys are used as unit identifiers.
            shared (pd.DataFrame): A (d,1) DataFrame containing shared parameters.
                The index should be parameter names and the single column should be named 'shared'.
            unit_specific (pd.DataFrame): A (d,U) DataFrame containing unit-specific parameters.
                The index should be parameter names and columns should be unit names.
        """
        if not isinstance(Pomp_dict, dict):
            raise TypeError("Pomp_dict must be a dictionary")
        for value in Pomp_dict.values():
            if not isinstance(value, Pomp):
                raise TypeError(
                    "Every element of Pomp_dict must be an instance of the class Pomp"
                )

        # Validate shared parameters DataFrame
        if not isinstance(shared, pd.DataFrame):
            raise TypeError("shared must be a pandas DataFrame")
        if shared.shape[1] != 1 or shared.columns[0] != "shared":
            raise ValueError(
                "shared must be a (d,1) DataFrame with column name 'shared'"
            )

        # Validate unit-specific parameters DataFrame
        if not isinstance(unit_specific, pd.DataFrame):
            raise TypeError("unit_specific must be a pandas DataFrame")
        if not all(unit in Pomp_dict for unit in unit_specific.columns):
            raise ValueError("unit_specific columns must match Pomp_dict keys")

        self.unit_objects = Pomp_dict
        self.shared = shared
        self.unit_specific = unit_specific

        # Store original parameter order for each unit
        self._unit_param_order = {}
        for unit, obj in self.unit_objects.items():
            self._unit_param_order[unit] = (
                list(obj.theta.keys()) if obj.theta is not None else []
            )
            obj.theta = None  # type: ignore

    def get_unit_parameters(self, unit: str) -> dict:
        """
        Get the complete parameter set for a specific unit, combining shared and unit-specific parameters.

        Args:
            unit (str): The unit identifier.

        Returns:
            dict: Combined parameters for the specified unit.
        """
        params = {}

        # Add shared parameters
        if not self.shared.empty:
            params.update(self.shared["shared"].to_dict())

        # Add unit-specific parameters
        if not self.unit_specific.empty:
            params.update(self.unit_specific[unit].to_dict())

        return params

    def simulate(
        self,
        key: jax.Array,
        theta: dict | None = None,
        times: jax.Array | None = None,
        covars: pd.DataFrame | None = None,
        nsim: int = 1,
    ) -> dict:
        """
        Simulates the PanelPOMP model by applying the simulate method to each unit.

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
            dict: A dictionary mapping unit names to simulation results. Each result
                is a dictionary containing:
                - 'X' (jax.Array): Unobserved state values with shape (n_times, n_states, nsim)
                - 'Y' (jax.Array): Observed values with shape (n_times, n_obs, nsim)
        """
        results = {}
        for unit, obj in self.unit_objects.items():
            key, subkey = jax.random.split(key)
            unit_theta = theta if theta is not None else self.get_unit_parameters(unit)
            results[unit] = obj.simulate(
                key=subkey,
                theta=unit_theta,
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
        ys: pd.DataFrame | None = None,
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
            unit_theta = theta if theta is not None else self.get_unit_parameters(unit)
            results[unit] = obj.pfilter(
                J=J,
                key=subkey,
                theta=unit_theta,
                ys=ys,
                covars=covars,
                thresh=thresh,
            )
        return results

    def _initialize_parameters(
        self, J: int
    ) -> tuple[jax.Array | None, jax.Array | None, list[str], list[str], list[str]]:
        """Initialize parameter matrices for MIF algorithm."""
        shared_params = list(self.shared.index)
        unit_specific_params = list(self.unit_specific.index)
        unit_names = list(self.unit_objects.keys())

        # Initialize shared parameters
        shared_values = (
            self.shared["shared"].values if not self.shared.empty else jnp.array([])
        )
        shared_thetas = (
            jnp.tile(shared_values, (J, 1)).T if len(shared_values) > 0 else None
        )

        # Initialize unit-specific parameters
        unit_specific_values = (
            self.unit_specific.values if not self.unit_specific.empty else jnp.array([])
        )
        unit_specific_thetas = (
            jnp.tile(unit_specific_values[:, None, :], (1, J, 1))
            if len(unit_specific_values) > 0
            else None
        )

        return (
            shared_thetas,
            unit_specific_thetas,
            shared_params,
            unit_specific_params,
            unit_names,
        )

    def _process_unit(
        self,
        unit: str,
        unit_idx: int,
        shared_thetas: jax.Array | None,
        unit_specific_thetas: jax.Array | None,
        shared_params: list[str],
        unit_specific_params: list[str],
        J: int,
        ys: dict[str, pd.DataFrame] | None,
        covars: dict[str, pd.DataFrame] | None,
        sigmas: float | jax.Array,
        sigmas_init: float | jax.Array,
        thresh: float,
        n_monitors: int,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Process a single unit in the MIF algorithm."""
        # Get unit-specific data
        unit_ys = ys[unit] if ys is not None else self.unit_objects[unit].ys
        unit_covars = (
            covars[unit] if covars is not None else self.unit_objects[unit].covars
        )

        # Get unit-specific components
        unit_rinit = self.unit_objects[unit].rinit
        unit_rproc = self.unit_objects[unit].rproc
        unit_dmeas = self.unit_objects[unit].dmeas

        if unit_rinit is None or unit_rproc is None or unit_dmeas is None:
            raise ValueError(f"Missing required components for unit {unit}")

        # Combine parameters for this unit
        unit_theta = {}
        if shared_thetas is not None:
            unit_theta.update(dict(zip(shared_params, shared_thetas)))
        if unit_specific_thetas is not None:
            unit_theta.update(
                dict(zip(unit_specific_params, unit_specific_thetas[:, :, unit_idx]))
            )

        # Validate parameters
        expected_param_order = self._unit_param_order[unit]
        for param in expected_param_order:
            if param not in unit_theta:
                raise KeyError(
                    f"Parameter '{param}' missing for unit '{unit}'. Check shared/unit-specific DataFrames."
                )

        # Build parameter array for particles
        theta_values = jnp.stack(
            [
                jnp.stack([unit_theta[param][j] for param in expected_param_order])
                for j in range(J)
            ]
        )

        # Run perturbed particle filter
        key, subkey = jax.random.split(key)
        unit_loglik, unit_thetas = _mif_internal(
            theta=theta_values,
            t0=unit_rinit.t0,
            times=jnp.array(unit_ys.index),
            ys=jnp.array(unit_ys),
            rinitializer=unit_rinit.struct_pf,
            rprocess=unit_rproc.struct_pf,
            dmeasure=unit_dmeas.struct_pf,
            rinitializers=unit_rinit.struct_per,
            rprocesses=unit_rproc.struct_per,
            dmeasures=unit_dmeas.struct_per,
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            ctimes=jnp.array(unit_covars.index) if unit_covars is not None else None,
            covars=jnp.array(unit_covars) if unit_covars is not None else None,
            M=1,  # Single iteration for each unit
            a=1.0,  # Not used in single iteration
            J=J,
            thresh=thresh,
            verbose=False,
            key=subkey,
            n_monitors=n_monitors,
            particle_thetas=True,
        )

        return unit_loglik[1], unit_thetas[1], key

    def _update_parameters(
        self,
        unit_thetas: jax.Array,
        expected_param_order: list[str],
        shared_params: list[str],
        unit_specific_params: list[str],
        shared_thetas: jax.Array | None,
        unit_specific_thetas: jax.Array | None,
        unit_idx: int,
        unit_names: list[str],
        J: int,
        block: bool,
    ) -> tuple[jax.Array | None, jax.Array | None]:
        """Update parameters based on particle filter results."""
        updated_shared_thetas = shared_thetas
        updated_unit_specific_thetas = unit_specific_thetas

        if shared_thetas is not None:
            shared_indices = [
                i for i, p in enumerate(expected_param_order) if p in shared_params
            ]
            updated_shared_thetas = unit_thetas[:, shared_indices].T

        if unit_specific_thetas is not None:
            unit_specific_indices = [
                i
                for i, p in enumerate(expected_param_order)
                if p in unit_specific_params
            ]
            updated_unit_specific_thetas = unit_specific_thetas.at[:, :, unit_idx].set(
                unit_thetas[:, unit_specific_indices].T
            )

            # Resample other units if not blocked
            if not block:
                for other_idx in range(len(unit_names)):
                    if other_idx != unit_idx:
                        updated_unit_specific_thetas = updated_unit_specific_thetas.at[
                            :, :, other_idx
                        ].set(updated_unit_specific_thetas[:, jnp.arange(J), other_idx])

        return updated_shared_thetas, updated_unit_specific_thetas

    def _create_results(
        self,
        logliks: jax.Array,
        shared_params_history: list[jax.Array],
        unit_specific_params_history: list[jax.Array],
        unit_logliks: dict[str, list[float]],
        shared_params: list[str],
        unit_specific_params: list[str],
        unit_names: list[str],
        J: int,
    ) -> dict[str, xr.DataArray]:
        """Create results dictionary in xarray format."""
        # Handle empty logliks
        if logliks.shape == (0,):
            logliks = jnp.zeros(len(unit_logliks[unit_names[0]]))

        results = {
            "logLiks": xr.DataArray(
                logliks,
                dims=["iteration"],
                coords={"iteration": range(1, len(logliks) + 1)},
            ),
        }

        if shared_params_history:
            shared_params_history_array = jnp.stack(shared_params_history)
            results["shared_thetas"] = xr.DataArray(
                shared_params_history_array,
                dims=["iteration", "param", "particle"],
                coords={
                    "iteration": range(len(shared_params_history_array)),
                    "param": shared_params,
                    "particle": range(J),
                },
            )

        if unit_specific_params_history:
            unit_specific_params_history_array = jnp.stack(unit_specific_params_history)
            results["unit_specific_thetas"] = xr.DataArray(
                unit_specific_params_history_array,
                dims=["iteration", "param", "particle", "unit"],
                coords={
                    "iteration": range(len(unit_specific_params_history_array)),
                    "param": unit_specific_params,
                    "particle": range(J),
                    "unit": unit_names,
                },
            )

        # Create unit_logLiks
        results["unit_logLiks"] = xr.DataArray(
            jnp.array([unit_logliks[unit] for unit in unit_names]).T,
            dims=["iteration", "unit"],
            coords={
                "iteration": range(len(unit_logliks[unit_names[0]])),
                "unit": unit_names,
            },
        )

        return results

    # TODO: quant test this function
    def mif(
        self,
        J: int,
        key: jax.Array,
        sigmas: float | jax.Array,
        sigmas_init: float | jax.Array,
        M: int,
        a: float,
        theta: dict | None = None,
        ys: dict | None = None,
        covars: dict | None = None,
        thresh: float = 0,
        verbose: bool = False,
        n_monitors: int = 1,
        block: bool = True,
    ) -> dict:
        """
        Runs the panel iterated filtering (PIF) algorithm, which estimates parameters
        by maximizing the combined log-likelihood across all units.

        Args:
            J (int): The number of particles.
            key (jax.Array): The random key for reproducibility.
            sigmas (float | jax.Array): Perturbation factor for parameters.
            sigmas_init (float | jax.Array): Initial perturbation factor for parameters.
            M (int): Number of algorithm iterations.
            a (float): Decay factor for sigmas.
            theta (dict, optional): Initial parameters for the POMP model.
                If provided, overrides the unit-specific parameters.
            ys (dict, optional): Dictionary mapping unit names to measurement arrays.
                If provided, overrides the unit-specific measurements.
            covars (dict, optional): Dictionary mapping unit names to covariate arrays.
                If provided, overrides the unit-specific covariates.
            thresh (float, optional): Resampling threshold. Defaults to 0.
            verbose (bool, optional): Flag to print log-likelihood and parameter information.
                Defaults to False.
            n_monitors (int, optional): Number of particle filter runs to average for log-likelihood estimation.
                Defaults to 1.
            block (bool, optional): Whether to block resampling of unit-specific parameters.
                Defaults to True.

        Returns:
            dict: A dictionary containing:
                - 'logLik' (xarray.DataArray): Combined log-likelihood values through iterations
                - 'shared_thetas' (xarray.DataArray): Shared parameter values through iterations
                - 'unit_specific_thetas' (xarray.DataArray): Unit-specific parameter values through iterations
                - 'unit_logLiks' (xarray.DataArray): Dictionary mapping unit names to their individual log-likelihoods
        """
        if J < 1:
            raise ValueError("J should be greater than 0.")

        # Initialize parameters
        (
            shared_thetas,
            unit_specific_thetas,
            shared_params,
            unit_specific_params,
            unit_names,
        ) = self._initialize_parameters(J)

        # Initialize arrays to store results
        logliks = []
        shared_params_history = []
        unit_specific_params_history = []
        unit_logliks = {unit: [] for unit in unit_names}

        # Store initial parameter state (iteration 0)
        if shared_thetas is not None:
            shared_params_history.append(shared_thetas.copy())
        if unit_specific_thetas is not None:
            unit_specific_params_history.append(unit_specific_thetas.copy())

        # Main MIF iterations
        for m in tqdm(range(M)):
            # Cool sigmas
            sigmas = a * sigmas
            sigmas_init = a * sigmas_init

            # Perturb parameters
            key, *subkeys = jax.random.split(key, num=3)
            if shared_thetas is not None:
                shared_thetas = shared_thetas + sigmas_init * jax.random.normal(
                    shape=shared_thetas.shape, key=subkeys[0]
                )
            if unit_specific_thetas is not None:
                unit_specific_thetas = (
                    unit_specific_thetas
                    + sigmas_init
                    * jax.random.normal(
                        shape=unit_specific_thetas.shape, key=subkeys[1]
                    )
                )

            # Process each unit
            total_loglik = 0
            for unit_idx, unit in enumerate(unit_names):
                # Process unit
                unit_loglik, unit_thetas, key = self._process_unit(
                    unit,
                    unit_idx,
                    shared_thetas,
                    unit_specific_thetas,
                    shared_params,
                    unit_specific_params,
                    J,
                    ys,
                    covars,
                    sigmas,
                    sigmas_init,
                    thresh,
                    n_monitors,
                    key,
                )

                # Update total log-likelihood
                total_loglik += unit_loglik
                unit_logliks[unit].append(unit_loglik)

                # Update parameters
                shared_thetas, unit_specific_thetas = self._update_parameters(
                    unit_thetas,
                    self._unit_param_order[unit],
                    shared_params,
                    unit_specific_params,
                    shared_thetas,
                    unit_specific_thetas,
                    unit_idx,
                    unit_names,
                    J,
                    block,
                )

            # Store parameter history
            if shared_thetas is not None:
                shared_params_history.append(shared_thetas.copy())
            if unit_specific_thetas is not None:
                unit_specific_params_history.append(unit_specific_thetas.copy())
            logliks.append(total_loglik)

            # Print verbose information
            if verbose:
                print(f"Iteration {m + 1}:")
                print(f"Total log-likelihood: {total_loglik}")
                if shared_thetas is not None:
                    print(
                        "Shared parameters:",
                        dict(zip(shared_params, shared_thetas.mean(1))),
                    )
                if unit_specific_thetas is not None:
                    print(
                        "Unit-specific parameters:",
                        {
                            unit: dict(
                                zip(
                                    unit_specific_params,
                                    unit_specific_thetas[:, :, i].mean(1),
                                )
                            )
                            for i, unit in enumerate(unit_names)
                        },
                    )

        # Create results
        return self._create_results(
            jnp.array(logliks),
            shared_params_history,
            unit_specific_params_history,
            unit_logliks,
            shared_params,
            unit_specific_params,
            unit_names,
            J,
        )

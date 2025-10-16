"""
This module implements the OOP structure for PanelPOMP models.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
from .pomp_class import Pomp
from .mif import _jv_panel_mif_internal
import numpy as np


class PanelPomp:
    def __init__(
        self,
        Pomp_dict: dict[str, Pomp],
        shared: pd.DataFrame | list[pd.DataFrame] | None = None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None = None,
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
        shared, unit_specific, unit_objects = self._validate_params_and_units(
            shared, unit_specific, Pomp_dict
        )

        self.unit_objects = unit_objects
        self.shared = shared
        self.unit_specific = unit_specific
        self.results_history = []

        for unit in self.unit_objects.keys():
            self.unit_objects[unit].theta = None  # type: ignore

    def _validate_unit_objects(self, unit_objects: dict[str, Pomp]) -> dict[str, Pomp]:
        if not isinstance(unit_objects, dict):
            raise TypeError("unit_objects must be a dictionary")
        for value in unit_objects.values():
            if not isinstance(value, Pomp):
                raise TypeError(
                    "Every element of unit_objects must be an instance of the class Pomp"
                )
        return unit_objects

    def _validate_shared(
        self, shared: pd.DataFrame | list[pd.DataFrame] | None
    ) -> list[pd.DataFrame] | None:
        if not isinstance(shared, (pd.DataFrame, list)) and shared is not None:
            raise TypeError(
                "shared must be a pandas DataFrame, a list of pandas DataFrames, or None"
            )
        if shared is None:
            return None
        if isinstance(shared, pd.DataFrame):
            shared = [shared]
        if not all(shared_i.shape[1] == 1 for shared_i in shared):
            raise ValueError("Data frames in shared must have shape (d,1)")
        for shared_i in shared:
            shared_i.columns = ["shared"]
        return shared

    def _validate_unit_specific(
        self, unit_specific: pd.DataFrame | list[pd.DataFrame] | None, units: list[str]
    ) -> list[pd.DataFrame] | None:
        if (
            not isinstance(unit_specific, (pd.DataFrame, list))
            and unit_specific is not None
        ):
            raise TypeError(
                "unit_specific must be a pandas DataFrame, a list of pandas DataFrames, or None"
            )
        if unit_specific is None:
            return None
        if isinstance(unit_specific, pd.DataFrame):
            unit_specific = [unit_specific]
        for unit_specific_i in unit_specific:
            if not all(unit_specific_i.columns == units):
                raise ValueError(
                    "unit_specific columns must match unit_objects keys in content and order"
                )

        return unit_specific

    def _validate_params_and_units(
        self,
        shared: pd.DataFrame | list[pd.DataFrame] | None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None,
        unit_objects: dict[str, Pomp],
    ) -> tuple[
        list[pd.DataFrame] | None,
        list[pd.DataFrame] | None,
        dict[str, Pomp],
    ]:
        unit_objects = self._validate_unit_objects(unit_objects)
        shared = self._validate_shared(shared)
        units = list(unit_objects.keys())
        unit_specific = self._validate_unit_specific(unit_specific, units)
        if shared is not None and unit_specific is not None:
            assert len(shared) == len(unit_specific)
        return shared, unit_specific, unit_objects

    def get_unit_parameters(
        self,
        unit: str,
        shared: list[pd.DataFrame] | None = None,
        unit_specific: list[pd.DataFrame] | None = None,
    ) -> list[dict]:
        """
        Get the complete parameter set for a specific unit, combining shared and
        unit-specific parameters.

        Args:
            unit (str): The unit identifier.
            shared (list[pd.DataFrame], optional): Shared parameters involved in the POMP model. If provided,
                overrides the shared parameter attribute.
            unit_specific (list[pd.DataFrame], optional): Unit-specific parameters involved in the POMP model.
                If provided, overrides the unit-specific parameter attribute.

        Returns:
            list[dict]: List of dictionaries with combined parameters for the specified unit.
        """
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        tll = self._get_theta_list_len(shared, unit_specific)
        params = [{}] * tll

        for i in range(tll):
            if shared is not None:
                params[i].update(shared[i]["shared"].to_dict())
            if unit_specific is not None:
                params[i].update(unit_specific[i][unit].to_dict())

        return params

    def _get_theta_list_len(
        self,
        shared: list[pd.DataFrame] | None,
        unit_specific: list[pd.DataFrame] | None,
    ) -> int:
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        return (
            len(shared)
            if shared is not None
            else len(unit_specific)
            if unit_specific is not None
            else 0
        )

    def simulate(
        self,
        key: jax.Array,
        shared: pd.DataFrame | list[pd.DataFrame] | None = None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulates the PanelPOMP model by applying the simulate method to each unit.

        Args:
            key (jax.Array): The random key for random number generation.
            shared (pd.DataFrame, optional): Parameters involved in the POMP model.
                If provided, overrides the shared parameters.
            unit_specific (pd.DataFrame, optional): Parameters involved in the POMP model.
                If provided, overrides the unit-specific parameters.
            times (jax.Array, optional): Times at which to generate observations.
                If provided, overrides the unit-specific times.
            nsim (int, optional): The number of simulations to perform. Defaults to 1.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the simulated unobserved state values and the simulated observed values in dataframes.
            The columns are as follows:
            - replicate: The index of the parameter set.
            - sim: The index of the simulation.
            - time: The time points at which the observations were made.
            - Remaining columns contain the features of the state and observation processes.
        """
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        shared, unit_specific, _ = self._validate_params_and_units(
            shared, unit_specific, self.unit_objects
        )

        X_sims_list = []
        Y_sims_list = []
        for unit, obj in self.unit_objects.items():
            theta_list = self.get_unit_parameters(unit, shared, unit_specific)
            key, subkey = jax.random.split(key)
            X_sims, Y_sims = obj.simulate(
                key=subkey,
                theta=theta_list,
                times=times,
                nsim=nsim,
            )
            X_sims.insert(0, "unit", unit)
            Y_sims.insert(0, "unit", unit)
            X_sims_list.append(X_sims)
            Y_sims_list.append(Y_sims)
        X_sims_long = pd.concat(X_sims_list)
        Y_sims_long = pd.concat(Y_sims_list)
        return X_sims_long, Y_sims_long

    def pfilter(
        self,
        J: int,
        key: jax.Array,
        shared: pd.DataFrame | list[pd.DataFrame] | None = None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None = None,
        thresh: float = 0,
        reps: int = 1,
    ) -> None:
        """
        Runs the particle filter on each unit in the panel.

        Args:
            J (int): The number of particles to use in the filter.
            key (jax.Array): The random key for reproducibility.
            shared (pd.DataFrame | list[pd.DataFrame], optional): Parameters involved
                in the POMP model. If provided, overrides the shared parameters.
            unit_specific (pd.DataFrame | list[pd.DataFrame], optional): Parameters
                involved in the POMP model. If provided, overrides the unit-specific
                parameters.
            thresh (float, optional): Threshold value to determine whether to resample
                particles.
            reps (int): Number of replicates to run.
        Returns:
            None. Updates self.results_history with the results of the particle filter
            algorithm for each unit in the panel.
        """
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        shared, unit_specific, _ = self._validate_params_and_units(
            shared, unit_specific, self.unit_objects
        )
        old_key = key

        tll = self._get_theta_list_len(shared, unit_specific)
        results = xr.DataArray(
            np.zeros((tll, len(self.unit_objects), reps)),
            dims=["theta", "unit", "replicate"],
            coords={"unit": list(self.unit_objects.keys()), "replicate": range(reps)},
        )
        for unit, obj in self.unit_objects.items():
            theta_list = self.get_unit_parameters(unit, shared, unit_specific)
            key, subkey = jax.random.split(key)
            obj.pfilter(
                J=J,
                key=subkey,
                theta=theta_list,
                thresh=thresh,
                reps=reps,
            )
            results.loc[:, unit, :] = obj.results_history[-1]["logLiks"]
            obj.results_history = []
        self.results_history.append(
            {
                "method": "pfilter",
                "logLiks": results,
                "shared": shared,
                "unit_specific": unit_specific,
                "J": J,
                "reps": reps,
                "thresh": thresh,
                "key": old_key,
            }
        )

    # TODO: quant test this function
    def mif(
        self,
        J: int,
        key: jax.Array,
        sigmas: float | jax.Array,
        sigmas_init: float | jax.Array,
        M: int,
        a: float,
        shared: pd.DataFrame | list[pd.DataFrame] | None = None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None = None,
        thresh: float = 0,
        block: bool = True,
    ) -> None:
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
            shared (pd.DataFrame | list[pd.DataFrame], optional): Shared parameters
                involved in the POMP model. If provided, overrides the shared
                parameter attribute.
            unit_specific (pd.DataFrame | list[pd.DataFrame], optional): Parameters
                involved in the POMP model for each unit. If provided, overrides the
                unit-specific parameters.
            thresh (float, optional): Resampling threshold.
            block (bool, optional): Whether to block resampling of unit-specific
                parameters.

        Returns:
            None. Updates self.results_history with the results of the MIF algorithm.
        """
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        shared, unit_specific, _ = self._validate_params_and_units(
            shared, unit_specific, self.unit_objects
        )
        if J < 1:
            raise ValueError("J should be greater than 0.")
        if M < 1:
            raise ValueError("M should be greater than 0.")
        if a < 0 or a > 1:
            raise ValueError("a should be between 0 and 1.")
        if block is False:
            raise NotImplementedError("block=False is not supported yet.")

        unit_names = list(self.unit_objects.keys())
        U = len(unit_names)
        # Use a representative unit for structural arrays and static callables
        rep_unit = self.unit_objects[unit_names[0]]

        # TODO: make this more flexible
        # Assume all units share the same dt_array_extended, nstep_array, t0, and times
        dt_array_extended = rep_unit._dt_array_extended
        nstep_array = rep_unit._nstep_array
        t0 = rep_unit.rinit.t0
        times = jnp.array(rep_unit.ys.index)

        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

        # TODO: make this more flexible
        # Assume all units share the same rinitializers, rprocesses_interp, dmeasures, accumvars
        rinitializers = rep_unit.rinit.struct_per
        rprocesses_interp = rep_unit.rproc.struct_per_interp
        dmeasures = rep_unit.dmeas.struct_per
        accumvars = rep_unit.rproc.accumvars

        has_covars = [
            self.unit_objects[u]._covars_extended is not None for u in unit_names
        ]
        if all(has_covars):
            covars_per_unit = jnp.stack(
                [jnp.array(self.unit_objects[u]._covars_extended) for u in unit_names],
                axis=0,
            )
        elif any(has_covars):
            raise NotImplementedError(
                "Some units have covariates, but not all units have covariates. This is not supported yet."
            )
        else:
            covars_per_unit = None

        shared_list = shared if isinstance(shared, list) else []
        spec_list = unit_specific if isinstance(unit_specific, list) else []
        n_reps = self._get_theta_list_len(shared, unit_specific)

        # Shared parameters
        if len(shared_list) == 0:
            n_shared = 0
            shared_array = jnp.zeros((n_reps, 0, J))
            shared_index: list[str] = []
        else:
            shared_index = list(shared_list[0].index)
            n_shared = len(shared_index)
            shared_array = jnp.stack(
                [
                    jnp.tile(
                        jnp.array(df["shared"].to_numpy().astype(float)).reshape(
                            n_shared, 1
                        ),
                        (1, J),
                    )
                    for df in shared_list
                ],
                axis=0,
            )

        # Unit-specific parameters
        if len(spec_list) == 0:
            n_spec = 0
            unit_array = jnp.zeros((n_reps, 0, J, U))
            spec_index: list[str] = []
        else:
            spec_index = list(spec_list[0].index)
            n_spec = len(spec_index)
            unit_array = jnp.stack(
                [
                    jnp.stack(
                        [
                            jnp.tile(
                                jnp.array(df[unit].to_numpy().astype(float)).reshape(
                                    n_spec, 1
                                ),
                                (1, J),
                            )
                            for unit in unit_names
                        ],
                        axis=2,
                    )  # shape: (n_spec, J, U)
                    for df in spec_list
                ],
                axis=0,
            )  # shape: (R, n_spec, J, U)

        # Stack ys per unit (units assumed to share time grid and dt/nstep arrays)
        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )

        old_key = key
        keys = jax.random.split(key, n_reps)
        (
            shared_array_f,
            unit_array_f,
            shared_traces,
            unit_traces,
            unit_logliks,
        ) = _jv_panel_mif_internal(
            shared_array,
            unit_array,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            ys_per_unit,
            rinitializers,
            rprocesses_interp,
            dmeasures,
            sigmas,
            sigmas_init,
            accumvars,
            covars_per_unit,
            M,
            a,
            J,
            U,
            thresh,
            keys,
        )

        shared_vars = ["logLik"] + shared_index
        unit_vars = ["unitLogLik"] + spec_index

        shared_da = xr.DataArray(
            shared_traces,
            dims=["replicate", "iteration", "variable"],
            coords={
                "replicate": jnp.arange(shared_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": shared_vars,
            },
        )
        unit_da = xr.DataArray(
            unit_traces,
            dims=["replicate", "iteration", "variable", "unit"],
            coords={
                "replicate": jnp.arange(unit_traces.shape[0]),
                "iteration": jnp.arange(M + 1),
                "variable": unit_vars,
                "unit": unit_names,
            },
        )

        full_logliks = xr.DataArray(
            jnp.concatenate(
                [np.sum(unit_logliks, axis=1).reshape(-1, 1), unit_logliks], axis=1
            ),
            dims=["replicate", "unit"],
            coords={"replicate": jnp.arange(n_reps), "unit": ["shared"] + unit_names},
        )

        if shared is not None:
            self.shared = [
                pd.DataFrame(
                    shared_traces[rep, -1, 1:].reshape(-1, 1),
                    index=pd.Index(shared_index),
                    columns=pd.Index(["shared"]),
                )
                for rep in range(shared_traces.shape[0])
            ]
        else:
            self.shared = None

        if unit_specific is not None:
            self.unit_specific = [
                pd.DataFrame(
                    unit_traces[rep, -1, 1:, :],
                    index=pd.Index(spec_index),
                    columns=pd.Index(unit_names),
                )
                for rep in range(unit_traces.shape[0])
            ]
        else:
            self.unit_specific = None

        self.results_history.append(
            {
                "method": "mif",
                "shared_traces": shared_da,
                "unit_traces": unit_da,
                "logLiks": full_logliks,
                "shared": shared,
                "unit_specific": unit_specific,
                "J": J,
                "thresh": thresh,
                "sigmas": sigmas,
                "sigmas_init": sigmas_init,
                "M": M,
                "a": a,
                "block": block,
                "key": old_key,
            }
        )

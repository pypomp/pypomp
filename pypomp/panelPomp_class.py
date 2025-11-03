"""
This module implements the OOP structure for PanelPOMP models.
"""

import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
from .pomp_class import Pomp
import matplotlib.pyplot as plt
import seaborn as sns
from .mif import _jv_panel_mif_internal
from .util import logmeanexp
import numpy as np
import time
from .RWSigma_class import RWSigma


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

        self.unit_objects: dict[str, Pomp] = unit_objects
        self.shared: list[pd.DataFrame] | None = shared
        self.unit_specific: list[pd.DataFrame] | None = unit_specific
        self.results_history = []
        self.fresh_key: jax.Array | None = None
        canonical_shared_param_names, canonical_unit_param_names = (
            self._get_param_names(shared, unit_specific)
        )
        self.canonical_shared_param_names: list[str] = canonical_shared_param_names
        self.canonical_unit_param_names: list[str] = canonical_unit_param_names
        self.canonical_param_names: list[str] = (
            canonical_shared_param_names + canonical_unit_param_names
        )

        for unit in self.unit_objects.keys():
            self.unit_objects[unit].theta = None  # type: ignore

    def _validate_unit_objects(self, unit_objects: dict[str, Pomp]) -> dict[str, Pomp]:
        if not isinstance(unit_objects, dict):
            raise TypeError("unit_objects must be a dictionary")
        unit_objs = list(unit_objects.values())
        for unit_obj in unit_objs:
            if not isinstance(unit_obj, Pomp):
                raise TypeError(
                    "Every element of unit_objects must be an instance of the class Pomp"
                )
            # TODO: loosen these constraints
            if unit_obj.t0 != unit_objs[0].t0:
                raise ValueError("All units must have the same t0")
            if any(unit_obj._dt_array_extended != unit_objs[0]._dt_array_extended):
                raise ValueError("All units must have the same _dt_array_extended")
            if any(unit_obj._nstep_array != unit_objs[0]._nstep_array):
                raise ValueError("All units must have the same _nstep_array")
            if any(unit_obj.ys.index != unit_objs[0].ys.index):
                raise ValueError("All units must have the same ys index")
            if any(unit_obj.ys.columns != unit_objs[0].ys.columns):
                raise ValueError("All units must have the same ys columns")

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
            if not shared_i.index.equals(shared[0].index):
                raise ValueError("shared index must match for all shared DataFrames")
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
            if not unit_specific_i.index.equals(unit_specific[0].index):
                raise ValueError(
                    "unit_specific index must match for all unit_specific DataFrames"
                )

        return unit_specific

    def _validate_params_and_units(
        self,
        shared: pd.DataFrame | list[pd.DataFrame] | None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None,
        unit_objects: dict[str, Pomp],
    ) -> tuple[list[pd.DataFrame] | None, list[pd.DataFrame] | None, dict[str, Pomp]]:
        unit_objects = self._validate_unit_objects(unit_objects)
        shared = self._validate_shared(shared)
        units = list(unit_objects.keys())
        unit_specific = self._validate_unit_specific(unit_specific, units)
        if shared is not None and unit_specific is not None:
            if len(shared) != len(unit_specific):
                raise ValueError(
                    "shared and unit_specific lists must have the same length if both are provided. "
                    f"shared length: {len(shared)}, unit_specific length: {len(unit_specific)}"
                )
        return shared, unit_specific, unit_objects

    def _update_fresh_key(
        self, key: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """
        Updates the fresh_key attribute and returns a new key along with the old key.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple containing the new key and the old key.
                The old key is the key that was used to update the fresh_key attribute.
                The new key is the key that should be used for the next method call.
        """
        old_key = self.fresh_key if key is None else key
        if old_key is None:
            raise ValueError(
                "Both the key argument and the fresh_key attribute are None. At least one key must be given."
            )
        self.fresh_key, new_key = jax.random.split(old_key)
        return new_key, old_key

    def _get_param_names(
        self,
        shared: list[pd.DataFrame] | None = None,
        unit_specific: list[pd.DataFrame] | None = None,
    ) -> tuple[list[str], list[str]]:
        shared = shared or self.shared
        shared_lst = [] if shared is None else list(shared[0].index)
        unit_specific = unit_specific or self.unit_specific
        unit_specific_lst = (
            [] if unit_specific is None else list(unit_specific[0].index)
        )
        return shared_lst, unit_specific_lst

    def _get_unit_param_permutation(self, unit_name: str) -> jax.Array:
        """
        Get permutation indices to reorder from PanelPomp canonical order
        to a specific unit's canonical order.

        Args:
            unit_name: Name of the unit

        Returns:
            Array of indices for reordering parameters
        """
        unit_canonical = self.unit_objects[unit_name].canonical_param_names
        panel_canonical = self.canonical_param_names

        # Create mapping from panel order to unit order
        # All unit parameters should be present in the panel
        try:
            permutation = [panel_canonical.index(name) for name in unit_canonical]
        except ValueError as e:
            missing = set(unit_canonical) - set(panel_canonical)
            raise ValueError(
                f"Unit '{unit_name}' has parameters {missing} that are not in the panel's parameter list. "
                f"Panel parameters: {panel_canonical}, Unit parameters: {unit_canonical}"
            ) from e
        return jnp.array(permutation, dtype=jnp.int32)

    def _dataframe_to_array_canonical(
        self, df: pd.DataFrame, param_names: list[str], column_name: str
    ) -> jnp.ndarray:
        """
        Convert a DataFrame column to a JAX array using canonical parameter ordering.

        Args:
            df: DataFrame with parameter names as index
            param_names: List of parameter names in canonical order
            column_name: Name of the column to extract values from

        Returns:
            JAX array with parameter values in canonical order
        """
        # Reorder DataFrame to match canonical order
        ordered_values = [df.loc[name, column_name] for name in param_names]
        return jnp.array(ordered_values, dtype=float)

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
        params = [{} for _ in range(tll)]

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
        """
        Get the number of parameter sets (i.e., replicates) from the two lists of parameters.
        """
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        if shared is None and unit_specific is not None:
            return len(unit_specific)
        elif shared is not None and unit_specific is None:
            return len(shared)
        elif shared is not None and unit_specific is not None:
            if len(shared) == len(unit_specific):
                return len(shared)
            else:
                raise ValueError(
                    "shared and unit_specific must have the same length if both are provided"
                )
        else:
            raise ValueError("At least one of shared or unit_specific must be provided")

    @staticmethod
    def sample_params(
        param_bounds: dict,
        units: list[str],
        n: int,
        key: jax.Array,
        shared_names: list[str] | None = None,
    ) -> tuple[list[pd.DataFrame] | None, list[pd.DataFrame] | None]:
        """
        Sample n sets of parameters from uniform distributions.

        Args:
            param_bounds (dict): Dictionary mapping parameter names to (lower, upper) bounds
            units (list[str]): List of unit names to sample parameters for.
            n (int): Number of parameter sets to sample
            key (jax.Array): JAX random key for reproducibility
            shared_names (list[str] | None): Names of shared parameters. Remaining
                parameters are unit-specific. If None, all parameters are unit-specific.


        Returns:
            tuple[list[pd.DataFrame] | None, list[pd.DataFrame] | None]: Two lists of length n.
              - First list: shared parameter DataFrames (S,1) with column 'shared' and
                index shared parameter names; None if no shared parameters were specified.
              - Second list: unit-specific parameter DataFrames (U*, len(units)) with
                columns equal to `units` and index unit-specific parameter names; None
                if no unit-specific parameters were specified.
        """
        param_keys = jax.random.split(key, n)
        param_names = list(param_bounds.keys())
        if shared_names is not None:
            unit_specific_names = [
                name for name in param_names if name not in set(shared_names)
            ]
        else:
            unit_specific_names = list(param_names)
        shared_param_sets: list[pd.DataFrame] | None = (
            [] if shared_names is not None and len(shared_names) > 0 else None
        )
        unit_specific_param_sets: list[pd.DataFrame] | None = (
            []
            if unit_specific_names is not None and len(unit_specific_names) > 0
            else None
        )

        for i in range(n):
            if (
                shared_names is not None
                and len(shared_names) > 0
                and shared_param_sets is not None
            ):
                shared_keys = jax.random.split(param_keys[i], len(shared_names))
                shared_values: list[float] = []
                for j_idx, param_name in enumerate(shared_names):
                    lower, upper = param_bounds[param_name]
                    val = float(
                        jax.random.uniform(
                            shared_keys[j_idx], shape=(), minval=lower, maxval=upper
                        )
                    )
                    shared_values.append(val)
                shared_df = pd.DataFrame(
                    shared_values,
                    index=pd.Index(shared_names),
                    columns=pd.Index(["shared"]),
                )
                shared_param_sets.append(shared_df)

            if (
                unit_specific_names is not None
                and len(unit_specific_names) > 0
                and unit_specific_param_sets is not None
            ):
                total_needed = len(unit_specific_names) * len(units)
                unit_keys = jax.random.split(param_keys[i], total_needed)
                values_by_param: dict[str, list[float]] = {
                    name: [] for name in unit_specific_names
                }
                k = 0
                for param_name in unit_specific_names:
                    lower, upper = param_bounds[param_name]
                    col_values: list[float] = []
                    for _unit in units:
                        val = float(
                            jax.random.uniform(
                                unit_keys[k], shape=(), minval=lower, maxval=upper
                            )
                        )
                        col_values.append(val)
                        k += 1
                    values_by_param[param_name] = col_values

                unit_df: pd.DataFrame | None = pd.DataFrame(
                    data={
                        u: [values_by_param[p][ui] for p in unit_specific_names]
                        for ui, u in enumerate(units)
                    },
                    index=pd.Index(unit_specific_names),
                    columns=pd.Index(units),
                )
                unit_specific_param_sets.append(unit_df)

        return shared_param_sets, unit_specific_param_sets

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
        shared, unit_specific, *_ = self._validate_params_and_units(
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
        key: jax.Array | None = None,
        shared: pd.DataFrame | list[pd.DataFrame] | None = None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None = None,
        thresh: float = 0.0,
        reps: int = 1,
    ) -> None:
        """
        Runs the particle filter on each unit in the panel.

        Args:
            J (int): The number of particles to use in the filter.
            key (jax.Array, optional): The random key for reproducibility. Defaults to
                self.fresh_key attribute.
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
        start_time = time.time()
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        shared, unit_specific, *_ = self._validate_params_and_units(
            shared, unit_specific, self.unit_objects
        )
        key, old_key = self._update_fresh_key(key)

        tll = self._get_theta_list_len(shared, unit_specific)
        results = xr.DataArray(
            np.zeros((tll, len(self.unit_objects), reps)),
            dims=["theta", "unit", "replicate"],
            coords={"unit": list(self.unit_objects.keys()), "replicate": range(reps)},
        )
        for unit, obj in self.unit_objects.items():
            theta_list = self.get_unit_parameters(unit, shared, unit_specific)
            key, subkey = jax.random.split(key)  # pyright: ignore[reportArgumentType]
            obj.pfilter(
                J=J,
                key=subkey,
                theta=theta_list,
                thresh=thresh,
                reps=reps,
            )
            results.loc[:, unit, :] = obj.results_history[-1]["logLiks"]
            obj.results_history = []

        execution_time = time.time() - start_time

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
                "execution_time": execution_time,
            }
        )

    # TODO: quant test this function
    def mif(
        self,
        J: int,
        M: int,
        rw_sd: RWSigma,
        a: float,
        key: jax.Array | None = None,
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
            M (int): Number of algorithm iterations.
            rw_sd (RWSigma): Random walk sigma object.
            a (float): Decay factor for RWSigma over 50 iterations.
            key (jax.Array): The random key for reproducibility.
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
        start_time = time.time()
        shared = shared or self.shared
        unit_specific = unit_specific or self.unit_specific
        shared, unit_specific, *_ = self._validate_params_and_units(
            shared, unit_specific, self.unit_objects
        )
        sigmas_array, sigmas_init_array = rw_sd._return_arrays(
            param_names=self.canonical_param_names
        )
        key, old_key = self._update_fresh_key(key)
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

        # Compute permutation indices to reorder from PanelPomp canonical order
        # to each unit's canonical order
        unit_param_permutations = jnp.stack(
            [self._get_unit_param_permutation(u) for u in unit_names], axis=0
        )  # shape: (U, n_params)

        # TODO: make this more flexible
        # Assume all units share the same dt_array_extended, nstep_array, t0, and times
        dt_array_extended = rep_unit._dt_array_extended
        nstep_array = rep_unit._nstep_array
        t0 = rep_unit.t0
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

        n_reps = self._get_theta_list_len(shared, unit_specific)

        shared_list = shared if isinstance(shared, list) else None
        spec_list = unit_specific if isinstance(unit_specific, list) else None
        shared_trans_list, spec_trans_list = rep_unit.par_trans.panel_transform_list(
            shared_list, spec_list, direction="to_est"
        )

        # Shared parameters
        if len(shared_trans_list) == 0:
            n_shared = 0
            shared_array = jnp.zeros((n_reps, 0, J))
            shared_index: list[str] = []
        else:
            shared_index = self.canonical_shared_param_names
            n_shared = len(shared_index)
            shared_array = jnp.stack(
                [
                    jnp.tile(
                        self._dataframe_to_array_canonical(
                            df, self.canonical_shared_param_names, "shared"
                        ).reshape(n_shared, 1),
                        (1, J),
                    )
                    for df in shared_trans_list
                ],
                axis=0,
            )

        # Unit-specific parameters
        if len(spec_trans_list) == 0:
            n_spec = 0
            unit_array = jnp.zeros((n_reps, 0, J, U))
            spec_index: list[str] = []
        else:
            spec_index = self.canonical_unit_param_names
            n_spec = len(spec_index)
            unit_array = jnp.stack(
                [
                    jnp.stack(
                        [
                            jnp.tile(
                                self._dataframe_to_array_canonical(
                                    df, self.canonical_unit_param_names, unit
                                ).reshape(n_spec, 1),
                                (1, J),
                            )
                            for unit in unit_names
                        ],
                        axis=2,
                    )  # shape: (n_spec, J, U)
                    for df in spec_trans_list
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
            sigmas_array,
            sigmas_init_array,
            accumvars,
            covars_per_unit,
            unit_param_permutations,
            M,
            a,
            J,
            U,
            thresh,
            keys,
        )

        shared_traces, unit_traces = rep_unit.par_trans.transform_panel_traces(
            shared_traces=np.array(shared_traces),
            unit_traces=np.array(unit_traces),
            shared_param_names=shared_index,
            unit_param_names=spec_index,
            unit_names=unit_names,
            direction="from_est",
        )

        if shared_traces is None:
            # unit_traces shape: (R, M+1, n_spec+1, U) where [:, :, 0, :] is per-unit loglik
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            n_reps = unit_traces.shape[0]
            shared_ll = np.sum(
                unit_traces[:, :, 0, :], axis=-1, keepdims=True
            )  # (R, M+1, 1)
            shared_traces = shared_ll  # only 'logLik' column when no shared params
            shared_index = []
        if unit_traces is None:
            # Construct empty unit traces with just unit logliks if missing
            n_reps = shared_traces.shape[0]
            unit_ll = np.zeros((n_reps, M + 1, 1, U), dtype=float)
            unit_traces = unit_ll

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

        shared_final_logliks = shared_traces[:, -1, 0]  # shape: (n_reps,)
        unit_final_logliks = unit_traces[:, -1, 0, :]  # shape: (n_reps, n_units)

        full_logliks = xr.DataArray(
            jnp.concatenate(
                [shared_final_logliks.reshape(-1, 1), unit_final_logliks], axis=1
            ),
            dims=["replicate", "unit"],
            coords={"replicate": jnp.arange(n_reps), "unit": ["shared"] + unit_names},
        )

        if shared is not None:
            shared_list_out = [
                pd.DataFrame(
                    shared_traces[rep, -1, 1:].reshape(-1, 1),
                    index=pd.Index(shared_index),
                    columns=pd.Index(["shared"]),
                )
                for rep in range(shared_traces.shape[0])
            ]
        else:
            shared_list_out = None

        if unit_specific is not None:
            specific_list_out = [
                pd.DataFrame(
                    unit_traces[rep, -1, 1:, :],
                    index=pd.Index(spec_index),
                    columns=pd.Index(unit_names),
                )
                for rep in range(unit_traces.shape[0])
            ]
        else:
            specific_list_out = None

        self.shared = shared_list_out
        self.unit_specific = specific_list_out

        execution_time = time.time() - start_time
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
                "rw_sd": rw_sd,
                "M": M,
                "a": a,
                "block": block,
                "key": old_key,
                "execution_time": execution_time,
            }
        )

    # TODO: clean up results functions
    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame with results for each replicate and unit for the given method.

        Columns:
            - replicate: index of the replicate
            - unit: the unit name
            - shared log-likelihood: log-likelihood contribution of shared parameters
            - unit log-likelihood: log-likelihood contribution of unit parameters
            - <shared_param>: columns for each shared parameter
            - <unit_param>: columns for each unit parameter
        """
        res = self.results_history[index]
        method = res.get("method", None)
        if method == "mif":
            return self._results_from_mif(res)
        elif method == "pfilter":
            return self._results_from_pfilter(res, ignore_nan=ignore_nan)
        else:
            raise ValueError(f"Unknown method '{method}' for results()")

    def _extract_parameter_dict(self, values, names, skip_first=False):
        """Helper to extract parameter dictionary from values and names."""
        if skip_first and len(values) > 1:
            values = values[1:]
            names = names[1:] if len(names) > 1 else []

        return {str(name): float(val.item()) for name, val in zip(names, values)}

    def _build_result_row(
        self, rep, unit, shared_loglik, unit_loglik, shared_dict, unit_dict
    ):
        """Helper to build a result row."""
        return {
            "replicate": rep,
            "unit": unit,
            "shared logLik": shared_loglik,
            "unit logLik": unit_loglik,
            **shared_dict,
            **unit_dict,
        }

    def _results_from_mif(self, res) -> pd.DataFrame:
        """
        Helper to process results from the panel mif method.
        """
        shared_da = res["shared_traces"]
        unit_da = res["unit_traces"]
        full_logliks = res["logLiks"]

        # Get parameter names, skipping loglikelihood entries
        all_shared_vars = list(shared_da.coords["variable"].values)
        shared_names = all_shared_vars[1:] if len(all_shared_vars) > 1 else []

        all_unit_vars = list(unit_da.coords["variable"].values)
        unit_names = list(unit_da.coords["unit"].values)
        n_reps = shared_da.sizes["replicate"]

        shared_final_values = shared_da.isel(
            iteration=-1
        ).values  # shape: (n_reps, n_vars)
        unit_final_values = unit_da.isel(
            iteration=-1
        ).values  # shape: (n_reps, n_vars, n_units)

        # Extract loglikelihoods
        shared_logliks = full_logliks[:, 0].values  # shape: (n_reps,)
        unit_logliks = full_logliks[:, 1:].values  # shape: (n_reps, n_units)

        rep_indices = np.repeat(np.arange(n_reps), len(unit_names))
        unit_indices = np.tile(np.arange(len(unit_names)), n_reps)

        data = {
            "replicate": rep_indices,
            "unit": [unit_names[i] for i in unit_indices],
            "shared logLik": shared_logliks[rep_indices],
            "unit logLik": unit_logliks[rep_indices, unit_indices],
        }

        if shared_names:
            shared_param_values = shared_final_values[:, 1:]  # Skip logLik column
            for i, param_name in enumerate(shared_names):
                data[param_name] = shared_param_values[rep_indices, i]

        unit_param_values = unit_final_values[
            :, 1:, :
        ]  # Skip logLik column, shape: (n_reps, n_unit_params, n_units)
        for i, param_name in enumerate(all_unit_vars[1:]):  # Skip logLik
            data[param_name] = unit_param_values[rep_indices, i, unit_indices]

        return pd.DataFrame(data)

    def _results_from_pfilter(self, res, ignore_nan) -> pd.DataFrame:
        """
        Helper to process results from the panel pfilter method.
        """
        logLiks = res["logLiks"]
        shared_params = res.get("shared", [])
        unit_specific_params = res.get("unit_specific", [])

        # Get unit names from coords (no "shared" unit in pfilter results)
        unit_names = list(logLiks.coords["unit"].values)
        n_reps = logLiks.sizes["theta"]

        logliks_array = logLiks.values  # shape: (n_reps, n_units, n_replicates)

        unit_logliks = np.array(
            [
                [
                    logmeanexp(logliks_array[rep, unit_idx, :], ignore_nan=ignore_nan)
                    for unit_idx in range(len(unit_names))
                ]
                for rep in range(n_reps)
            ]
        )  # shape: (n_reps, n_units)

        shared_logliks = np.sum(unit_logliks, axis=1)  # shape: (n_reps,)

        rep_indices = np.repeat(np.arange(n_reps), len(unit_names))
        unit_indices = np.tile(np.arange(len(unit_names)), n_reps)

        data = {
            "replicate": rep_indices,
            "unit": [unit_names[i] for i in unit_indices],
            "shared logLik": shared_logliks[rep_indices],
            "unit logLik": unit_logliks[rep_indices, unit_indices],
        }

        if shared_params and len(shared_params) > 0:
            shared_param_data = {}
            for rep in range(min(n_reps, len(shared_params))):
                shared_df = shared_params[rep]
                if hasattr(shared_df, "values") and shared_df.shape[1] >= 1:
                    shared_vals = shared_df.iloc[:, 0].values
                    shared_names = (
                        list(shared_df.columns) if hasattr(shared_df, "columns") else []
                    )
                    for i, param_name in enumerate(shared_names):
                        if param_name not in shared_param_data:
                            shared_param_data[param_name] = np.full(n_reps, np.nan)
                        shared_param_data[param_name][rep] = (
                            shared_vals[i] if i < len(shared_vals) else np.nan
                        )

            for param_name, values in shared_param_data.items():
                data[param_name] = values[rep_indices]

        if unit_specific_params and len(unit_specific_params) > 0:
            unit_param_data = {}
            for rep in range(min(n_reps, len(unit_specific_params))):
                unit_df = unit_specific_params[rep]
                if hasattr(unit_df, "columns"):
                    unit_param_names = (
                        list(unit_df.index) if hasattr(unit_df, "index") else []
                    )
                    for param_name in unit_param_names:
                        if param_name not in unit_param_data:
                            unit_param_data[param_name] = np.full(
                                (n_reps, len(unit_names)), np.nan
                            )
                        for unit_idx, unit in enumerate(unit_names):
                            if unit in unit_df.columns:
                                unit_param_data[param_name][rep, unit_idx] = (
                                    unit_df.loc[param_name, unit]
                                )

            for param_name, values in unit_param_data.items():
                data[param_name] = values[rep_indices, unit_indices]

        return pd.DataFrame(data)

    def time(self):
        """
        Return a DataFrame summarizing the execution times of methods run.

        Returns:
            pd.DataFrame: A DataFrame where each row contains:
                - 'method': The name of the method run.
                - 'time': The execution time in seconds.
        """
        rows = []
        for idx, res in enumerate(self.results_history):
            method = res.get("method", None)
            exec_time = res.get("execution_time", None)
            rows.append({"method": method, "time": exec_time})
        df = pd.DataFrame(rows)
        df.index.name = "history_index"
        return df

    def traces(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the full trace of log-likelihoods and parameters from the entire result history.
        Columns:
            - replicate: The index of the parameter set (for all methods)
            - unit: The unit name (includes 'shared' for shared parameters/log-likelihood)
            - iteration: The global iteration number for that parameter set (increments over all mif calls for that set; for pfilter, the last iteration for that set)
            - method: 'pfilter' or 'mif'
            - logLik: The log-likelihood estimate (for mif: shared total on unit='shared' rows and per-unit unitLogLik on unit rows; for pfilter: averaged over reps; the shared row is the sum over units)
            - <param>: One column for each parameter (shared parameters are repeated on all rows; unit-specific parameters appear on their corresponding unit rows; others are NaN)
        """
        if not self.results_history:
            return pd.DataFrame()

        shared_param_names, unit_param_names = self._get_param_names()
        all_param_names = shared_param_names + unit_param_names

        all_data = []
        global_iters: dict[int, int] = {}

        for res in self.results_history:
            method = res.get("method")
            if method == "mif":
                shared_da = res["shared_traces"]  # dims: replicate, iteration, variable
                unit_da = res[
                    "unit_traces"
                ]  # dims: replicate, iteration, variable, unit
                unit_names = list(unit_da.coords["unit"].values)
                shared_vars = list(shared_da.coords["variable"].values)
                unit_vars = list(unit_da.coords["variable"].values)

                n_rep = shared_da.sizes["replicate"]
                n_iter = shared_da.sizes["iteration"]

                shared_values = shared_da.values  # shape: (n_rep, n_iter, n_vars)
                unit_values = unit_da.values  # shape: (n_rep, n_iter, n_vars, n_units)

                shared_param_indices = {
                    name: i for i, name in enumerate(shared_vars[1:])
                }
                unit_param_indices = {name: i for i, name in enumerate(unit_vars[1:])}

                for rep_idx in range(n_rep):
                    if rep_idx not in global_iters:
                        global_iters[rep_idx] = 0

                    for iter_idx in range(n_iter):
                        shared_loglik = float(
                            shared_values[rep_idx, iter_idx, 0]
                        )  # logLik is first
                        shared_params = {
                            name: float(
                                shared_values[
                                    rep_idx, iter_idx, shared_param_indices[name] + 1
                                ]
                            )
                            for name in shared_param_indices
                        }

                        shared_row = {
                            "replicate": rep_idx,
                            "unit": "shared",
                            "iteration": global_iters[rep_idx],
                            "method": "mif",
                            "logLik": shared_loglik,
                        }
                        for name in all_param_names:
                            shared_row[name] = shared_params.get(name, float("nan"))
                        all_data.append(shared_row)

                        for unit_idx, unit in enumerate(unit_names):
                            unit_loglik = float(
                                unit_values[rep_idx, iter_idx, 0, unit_idx]
                            )  # unitLogLik is first
                            unit_params = {
                                name: float(
                                    unit_values[
                                        rep_idx,
                                        iter_idx,
                                        unit_param_indices[name] + 1,
                                        unit_idx,
                                    ]
                                )
                                for name in unit_param_indices
                            }

                            unit_row = {
                                "replicate": rep_idx,
                                "unit": str(unit),
                                "iteration": global_iters[rep_idx],
                                "method": "mif",
                                "logLik": unit_loglik,
                            }
                            for name in all_param_names:
                                if name in unit_params:
                                    unit_row[name] = unit_params[name]
                                elif name in shared_params:
                                    unit_row[name] = shared_params[name]
                                else:
                                    unit_row[name] = float("nan")
                            all_data.append(unit_row)

                        global_iters[rep_idx] += 1

            elif method == "pfilter":
                logLiks = res["logLiks"]  # dims: theta, unit, replicate
                unit_names = list(logLiks.coords["unit"].values)
                shared_list = res.get("shared")
                unit_list = res.get("unit_specific")

                n_theta = logLiks.sizes["theta"]

                logliks_array = (
                    logLiks.values
                )  # shape: (n_theta, n_units, n_replicates)
                unit_avgs = np.array(
                    [
                        [
                            logmeanexp(logliks_array[rep_idx, u_i, :])
                            for u_i in range(len(unit_names))
                        ]
                        for rep_idx in range(n_theta)
                    ]
                )  # shape: (n_theta, n_units)
                shared_totals = np.sum(unit_avgs, axis=1)  # shape: (n_theta,)

                shared_param_data = {}
                if isinstance(shared_list, list):
                    for rep_idx in range(min(n_theta, len(shared_list))):
                        df = shared_list[rep_idx]
                        if hasattr(df, "index") and df.shape[1] >= 1:
                            for name in df.index:
                                if name not in shared_param_data:
                                    shared_param_data[name] = np.full(n_theta, np.nan)
                                shared_param_data[name][rep_idx] = float(
                                    df.loc[name, df.columns[0]]
                                )

                unit_param_data = {}
                if isinstance(unit_list, list):
                    for rep_idx in range(min(n_theta, len(unit_list))):
                        df = unit_list[rep_idx]
                        if hasattr(df, "columns"):
                            for name in df.index:
                                if name not in unit_param_data:
                                    unit_param_data[name] = np.full(
                                        (n_theta, len(unit_names)), np.nan
                                    )
                                for unit_idx, unit in enumerate(unit_names):
                                    if str(unit) in df.columns:
                                        unit_param_data[name][rep_idx, unit_idx] = (
                                            float(df.loc[name, str(unit)])
                                        )

                for rep_idx in range(n_theta):
                    iter_val = global_iters.get(rep_idx, 1) - 1
                    iter_val = iter_val if iter_val > 0 else 1

                    shared_row = {
                        "replicate": rep_idx,
                        "unit": "shared",
                        "iteration": iter_val,
                        "method": "pfilter",
                        "logLik": float(shared_totals[rep_idx]),
                    }
                    for name in all_param_names:
                        if name in shared_param_data:
                            shared_row[name] = shared_param_data[name][rep_idx]
                        else:
                            shared_row[name] = float("nan")
                    all_data.append(shared_row)

                    for u_i, unit in enumerate(unit_names):
                        unit_row = {
                            "replicate": rep_idx,
                            "unit": str(unit),
                            "iteration": iter_val,
                            "method": "pfilter",
                            "logLik": float(unit_avgs[rep_idx, u_i]),
                        }
                        for name in all_param_names:
                            if name in unit_param_data:
                                unit_row[name] = unit_param_data[name][rep_idx, u_i]
                            elif name in shared_param_data:
                                unit_row[name] = shared_param_data[name][rep_idx]
                            else:
                                unit_row[name] = float("nan")
                        all_data.append(unit_row)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        # Sort for readability
        if not df.empty:
            df = df.sort_values(["replicate", "unit", "iteration"]).reset_index(
                drop=True
            )
        return df

    # TODO: clean up plot_traces function
    def plot_traces(self, which: str = "shared", show: bool = True):
        """
        Plot traces using PanelPomp.traces(). Produces a single figure per call.

        Args:
            which (str): One of:
                - "shared": plot all shared values (including shared logLik) faceted by variable
                - "unitLogLik": plot per-unit logLik (faceted by unit, unit != 'shared')
                - <unit-specific parameter name>: plot that parameter across units (faceted by unit)
            show (bool): Whether to display the plot.
        """

        traces = self.traces()
        assert isinstance(traces, pd.DataFrame)
        if traces.empty:
            print("No trace data to plot.")
            return

        value_cols = [
            c
            for c in traces.columns
            if c not in ["replicate", "iteration", "method", "unit"]
        ]

        # Determine shared vs unit-specific parameters using presence on the shared row
        has_shared_rows = bool((traces["unit"] == "shared").any())
        shared_params = []
        if has_shared_rows:
            shared_rows = traces.loc[traces["unit"] == "shared"]
            for c in value_cols:
                if c != "logLik" and pd.notna(shared_rows[c]).any():
                    shared_params.append(c)
        else:
            shared_params = []

        unit_params = [
            c for c in value_cols if c != "logLik" and c not in shared_params
        ]

        if which == "shared":
            if not has_shared_rows:
                print("No shared rows to plot.")
                return None
            shared_vars = (["logLik"] if "logLik" in value_cols else []) + shared_params
            if len(shared_vars) == 0:
                print("No shared parameters or logLik to plot.")
                return None

            df_shared = traces.loc[
                traces["unit"] == "shared",
                ["replicate", "iteration", "method", *shared_vars],
            ]
            assert isinstance(df_shared, pd.DataFrame)
            df_shared_long: pd.DataFrame = df_shared.melt(
                id_vars=["replicate", "iteration", "method"],
                value_vars=shared_vars,
                var_name="variable",
                value_name="value",
            )

            g = sns.FacetGrid(
                df_shared_long,
                col="variable",
                sharex=True,
                sharey=False,
                hue="replicate",
                col_wrap=3,
                height=3.5,
                aspect=1.2,
                palette="tab10",
            )

            def facet_plot_shared(data, color, **kwargs):
                for rep, group in data.groupby("replicate"):
                    for method in ["mif", "train"]:
                        sub = group[group["method"] == method]
                        if len(sub) > 1:
                            plt.plot(
                                sub["iteration"],
                                sub["value"],
                                "-",
                                color=color,
                                alpha=0.8,
                            )
                        elif len(sub) == 1:
                            plt.scatter(
                                sub["iteration"],
                                sub["value"],
                                color=color,
                                marker="o",
                                alpha=0.8,
                            )
                    sub = group[group["method"] == "pfilter"]
                    if not sub.empty:
                        plt.scatter(
                            sub["iteration"],
                            sub["value"],
                            color=color,
                            marker="o",
                            edgecolor="k",
                            zorder=3,
                        )

            g.map_dataframe(facet_plot_shared)
            g.add_legend(title="Replicate")
            g.set_axis_labels("Iteration", "Value")
            g.set_titles(col_template="{col_name}")
            plt.tight_layout()
            if show:
                plt.show()
            return g

        if which == "unitLogLik":
            # Plot per-unit logLik, exclude the shared row
            df_ul = traces.loc[
                traces["unit"] != "shared",
                ["replicate", "iteration", "method", "unit", "logLik"],
            ].rename(columns={"logLik": "value"})
            df_ul = df_ul.loc[pd.notna(df_ul["value"])]
            if bool(df_ul.empty):
                print("No unit-specific logLik data to plot.")
                return None

            g = sns.FacetGrid(
                df_ul,
                col="unit",
                sharex=True,
                sharey=False,
                hue="replicate",
                col_wrap=4,
                height=3.2,
                aspect=1.1,
                palette="tab10",
            )

            def facet_plot_units_ll(data, color, **kwargs):
                for rep, group in data.groupby("replicate"):
                    for method in ["mif", "train"]:
                        sub = group[group["method"] == method]
                        if len(sub) > 1:
                            plt.plot(
                                sub["iteration"],
                                sub["value"],
                                "-",
                                color=color,
                                alpha=0.8,
                            )
                        elif len(sub) == 1:
                            plt.scatter(
                                sub["iteration"],
                                sub["value"],
                                color=color,
                                marker="o",
                                alpha=0.8,
                            )
                    sub = group[group["method"] == "pfilter"]
                    if not sub.empty:
                        plt.scatter(
                            sub["iteration"],
                            sub["value"],
                            color=color,
                            marker="o",
                            edgecolor="k",
                            zorder=3,
                        )

            g.map_dataframe(facet_plot_units_ll)
            g.add_legend(title="Replicate")
            g.set_axis_labels("Iteration", "logLik")
            g.set_titles(col_template="{col_name}")
            plt.tight_layout()
            if show:
                plt.show()
            return g

        # Otherwise, treat 'which' as a unit-specific parameter name
        if which not in unit_params:
            raise ValueError(
                f"'{which}' not found among unit-specific parameters: {unit_params}"
            )

        df_param = traces.loc[
            :, ["replicate", "iteration", "method", "unit", which]
        ].copy()
        assert isinstance(df_param, pd.DataFrame)
        df_param = df_param.loc[pd.notna(df_param[which])]
        if bool(df_param.empty):
            print(f"No data to plot for unit-specific parameter '{which}'.")
            return None
        df_param = df_param.rename(columns={which: "value"})

        g = sns.FacetGrid(
            df_param,
            col="unit",
            sharex=True,
            sharey=False,
            hue="replicate",
            col_wrap=4,
            height=3.2,
            aspect=1.1,
            palette="tab10",
        )

        def facet_plot_units(data, color, **kwargs):
            for rep, group in data.groupby("replicate"):
                for method in ["mif", "train"]:
                    sub = group[group["method"] == method]
                    if len(sub) > 1:
                        plt.plot(
                            sub["iteration"], sub["value"], "-", color=color, alpha=0.8
                        )
                    elif len(sub) == 1:
                        plt.scatter(
                            sub["iteration"],
                            sub["value"],
                            color=color,
                            marker="o",
                            alpha=0.8,
                        )
                sub = group[group["method"] == "pfilter"]
                if not sub.empty:
                    plt.scatter(
                        sub["iteration"],
                        sub["value"],
                        color=color,
                        marker="o",
                        edgecolor="k",
                        zorder=3,
                    )

        g.map_dataframe(facet_plot_units)
        g.add_legend(title="Replicate")
        g.set_axis_labels("Iteration", which)
        g.set_titles(col_template="{col_name}")
        plt.tight_layout()
        if show:
            plt.show()
        return g

    def __getstate__(self):
        """
        Custom pickling method to handle wrapped function objects. This is
        necessary because the JAX-wrapped functions in the Pomp objects are not picklable.
        """
        state = self.__dict__.copy()

        # Handle unit_objects by storing their state information
        if hasattr(self, "unit_objects") and self.unit_objects is not None:
            unit_objects_state = {}
            for unit_name, pomp_obj in self.unit_objects.items():
                # Get the state of each Pomp object
                unit_objects_state[unit_name] = pomp_obj.__getstate__()
            state["_unit_objects_state"] = unit_objects_state
            # Remove the original unit_objects from state
            state.pop("unit_objects", None)

        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to reconstruct wrapped function objects. This is
        necessary because the JAX-wrapped functions in the Pomp objects are not picklable.
        """
        # Restore basic attributes
        self.__dict__.update(state)

        # Reconstruct unit_objects
        if "_unit_objects_state" in state:
            unit_objects = {}
            for unit_name, pomp_state in state["_unit_objects_state"].items():
                # Create a new Pomp object and restore its state
                pomp_obj = Pomp.__new__(Pomp)
                pomp_obj.__setstate__(pomp_state)
                unit_objects[unit_name] = pomp_obj
            self.unit_objects = unit_objects
            # Clean up temporary state
            del self.__dict__["_unit_objects_state"]
        else:
            self.unit_objects = {}

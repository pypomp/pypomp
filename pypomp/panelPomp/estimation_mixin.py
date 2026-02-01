import gc
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
import numpy as np
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Union, cast

from ..mif import _jv_panel_mif_internal
from ..RWSigma_class import RWSigma
from ..results import PanelPompPFilterResult, PanelPompMIFResult, ResultsHistory
from ..parameters import PanelParameters
from ..util import logmeanexp

if TYPE_CHECKING:
    from .interfaces import PanelPompInterface as Base
else:
    Base = object  # At runtime, this is just a normal class


class PanelEstimationMixin(Base):
    """
    Handles Simulation, Particle Filtering, and MIF algorithms.
    """

    def _update_fresh_key(
        self, key: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]:
        old_key = self.fresh_key if key is None else key
        if old_key is None:
            raise ValueError(
                "Both the key argument and the fresh_key attribute are None. At least one key must be given."
            )
        self.fresh_key, new_key = jax.random.split(old_key)
        return new_key, old_key

    def _prepare_theta_input(
        self,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ],
    ) -> PanelParameters:
        """Convert various theta inputs to PanelParameters."""
        if isinstance(theta, PanelParameters):
            return theta
        if theta is None:
            if self.theta is None:
                raise ValueError("theta must be provided or self.theta must exist")
            return self.theta
        return PanelParameters(theta=theta)

    def _get_unit_param_permutation(self, unit_name: str) -> jax.Array:
        unit_canonical = self.unit_objects[unit_name].canonical_param_names
        panel_canonical = self.canonical_param_names

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
        ordered_values = [df.loc[name, column_name] for name in param_names]
        return jnp.array(ordered_values, dtype=float)

    def get_unit_parameters(
        self,
        unit: str,
        theta: PanelParameters | None = None,
    ) -> list[dict[str, float]]:
        theta = self._prepare_theta_input(theta)

        tll = theta.num_replicates()
        params = [{} for _ in range(tll)]

        for i in range(tll):
            theta_dict = theta.theta[i]
            if theta_dict["shared"] is not None:
                params[i].update(theta_dict["shared"].iloc[:, 0].to_dict())
            if theta_dict["unit_specific"] is not None:
                params[i].update(theta_dict["unit_specific"][unit].to_dict())

        return params

    @staticmethod
    def sample_params(
        param_bounds: dict,
        units: list[str],
        n: int,
        key: jax.Array,
        shared_names: list[str] | None = None,
    ) -> list[dict[str, pd.DataFrame | None]]:
        """
        Sample parameters for PanelPomp models.

        Args:
            param_bounds (dict): Dictionary mapping parameter names to (lower, upper) bounds.
            units (list[str]): List of unit names.
            n (int): Number of parameter sets to sample.
            key (jax.Array): JAX random key for reproducibility.
            shared_names (list[str], optional): List of shared parameter names. If None, all parameters are considered unit-specific.

        Returns:
            list[dict[str, pd.DataFrame | None]]: List of n dictionaries containing sampled parameters. Each dictionary contains "shared" and "unit_specific" keys mapping to DataFrames or None.
        """
        shared = shared_names or []
        specific = [k for k in param_bounds if k not in shared]
        keys = jax.random.split(key, n)

        def _sample(k, names, n_cols):
            if not names:
                return None
            # Create arrays of shape (n_params, 1) for broadcasting
            low = jnp.array([param_bounds[p][0] for p in names])[:, None]
            high = jnp.array([param_bounds[p][1] for p in names])[:, None]
            # Sample (n_params, n_cols)
            return low + jax.random.uniform(k, (len(names), n_cols)) * (high - low)

        results = []
        for i in range(n):
            k_s, k_u = jax.random.split(keys[i])
            s_arr = _sample(k_s, shared, 1)
            u_arr = _sample(k_u, specific, len(units))

            results.append(
                {
                    "shared": pd.DataFrame(
                        s_arr, index=pd.Index(shared), columns=pd.Index(["shared"])
                    )
                    if shared
                    else None,
                    "unit_specific": pd.DataFrame(
                        u_arr, index=pd.Index(specific), columns=pd.Index(units)
                    )
                    if specific
                    else None,
                }
            )

        return results

    def simulate(
        self,
        key: jax.Array,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
        times: jax.Array | None = None,
        nsim: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        theta = self._prepare_theta_input(theta)

        X_sims_list = []
        Y_sims_list = []
        for unit, obj in self.unit_objects.items():
            theta_list = self.get_unit_parameters(unit, theta=theta)
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
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
        thresh: float = 0.0,
        reps: int = 1,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
    ) -> None:
        start_time = time.time()
        theta_obj_in = deepcopy(self._prepare_theta_input(theta))

        key, old_key = self._update_fresh_key(key)

        n_theta_reps = theta_obj_in.num_replicates()
        unit_names = list(self.unit_objects.keys())
        n_units = len(unit_names)

        results = xr.DataArray(
            np.zeros((n_theta_reps, n_units, reps)),
            dims=["theta", "unit", "replicate"],
            coords={"unit": unit_names, "replicate": range(reps)},
        )

        CLL_list = []
        ESS_list = []
        filter_mean_list = []
        prediction_mean_list = []

        for unit, obj in self.unit_objects.items():
            theta_list = self.get_unit_parameters(unit, theta=theta_obj_in)
            key, subkey = jax.random.split(key)  # pyright: ignore[reportArgumentType]
            obj.pfilter(
                J=J,
                key=subkey,
                theta=theta_list,
                thresh=thresh,
                reps=reps,
                CLL=CLL,
                ESS=ESS,
                filter_mean=filter_mean,
                prediction_mean=prediction_mean,
            )

            # Clear JAX caches and collect garbage to avoid memory leaks.
            # Each time pfilter is run for a different unit, compiled code is cached and accumulated.
            # This happens even if rproc, dmeas, etc. are functionally the same for all units, because the current implementation of this stuff causes JAX to treat the code as being different for each unit.
            # Clearing the caches is a bandaid, but also helpful when the functions really are different.
            # The runtime cost of clearing the caches is negligible, and recompilation times shouldn't be too much of an issue either.
            # There are probably better ways to deal with this which could be explored in the future, but this is a simple workaround for now.
            jax.clear_caches()
            gc.collect()

            unit_result = obj.results_history[-1]
            results.loc[:, unit, :] = unit_result.logLiks

            if CLL and unit_result.CLL is not None:
                CLL_list.append(unit_result.CLL)
            if ESS and unit_result.ESS is not None:
                ESS_list.append(unit_result.ESS)
            if filter_mean and unit_result.filter_mean is not None:
                filter_mean_list.append(unit_result.filter_mean)
            if prediction_mean and unit_result.prediction_mean is not None:
                prediction_mean_list.append(unit_result.prediction_mean)

            obj.results_history = ResultsHistory()

        # results has shape (n_theta_reps, len(self.unit_objects), reps)
        results_np = np.array(results.values)
        logLik_unit = np.apply_along_axis(
            logmeanexp, -1, results_np, ignore_nan=False
        )  # shape: (n_theta_reps, len(self.unit_objects))

        self.theta.logLik_unit = logLik_unit

        def _stack_diagnostics(diag_list, dims, coord_names):
            """Stack unit diagnostics and create DataArray with unit dimension."""
            if not diag_list:
                return None
            arrays = [np.array(arr.values) for arr in diag_list]
            stacked = np.stack(arrays, axis=1)
            coords = {"unit": unit_names, "replicate": range(reps)}
            for i, coord_name in enumerate(coord_names):
                coord_idx = -(len(coord_names) - i)
                if coord_name in diag_list[0].coords:
                    coords[coord_name] = diag_list[0].coords[coord_name]
                else:
                    coords[coord_name] = range(stacked.shape[coord_idx])
            return xr.DataArray(stacked, dims=dims, coords=coords)

        CLL_da = (
            _stack_diagnostics(
                CLL_list, ["theta", "unit", "replicate", "time"], ["time"]
            )
            if CLL
            else None
        )
        ESS_da = (
            _stack_diagnostics(
                ESS_list, ["theta", "unit", "replicate", "time"], ["time"]
            )
            if ESS
            else None
        )
        filter_mean_da = (
            _stack_diagnostics(
                filter_mean_list,
                ["theta", "unit", "replicate", "time", "state"],
                ["time", "state"],
            )
            if filter_mean
            else None
        )
        prediction_mean_da = (
            _stack_diagnostics(
                prediction_mean_list,
                ["theta", "unit", "replicate", "time", "state"],
                ["time", "state"],
            )
            if prediction_mean
            else None
        )

        execution_time = time.time() - start_time

        result = PanelPompPFilterResult(
            method="pfilter",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_in,
            logLiks=results,
            J=J,
            reps=reps,
            thresh=thresh,
            CLL=CLL_da,
            ESS=ESS_da,
            filter_mean=filter_mean_da,
            prediction_mean=prediction_mean_da,
        )

        self.results_history.add(result)

    def mif(
        self,
        J: int,
        M: int,
        rw_sd: RWSigma,
        a: float,
        key: jax.Array | None = None,
        theta: Union[
            PanelParameters,
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            None,
        ] = None,
        thresh: float = 0,
        block: bool = True,
    ) -> None:
        start_time = time.time()
        theta_obj_in: PanelParameters = deepcopy(self._prepare_theta_input(theta))
        if theta_obj_in is None:
            raise ValueError("theta must be provided or self.theta must exist")

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

        unit_names = self.get_unit_names()
        U = len(unit_names)
        rep_unit = self.unit_objects[unit_names[0]]

        unit_param_permutations = jnp.stack(
            [self._get_unit_param_permutation(u) for u in unit_names], axis=0
        )

        dt_array_extended = rep_unit._dt_array_extended
        nstep_array = rep_unit._nstep_array
        t0 = rep_unit.t0
        times = jnp.array(rep_unit.ys.index)

        if rep_unit.dmeas is None:
            raise ValueError("dmeas cannot be None in PanelPomp units")

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

        n_reps = theta_obj_in.num_replicates()

        # Extract theta list from PanelParameters and transform to estimation scale if needed
        theta_list = theta_obj_in.theta
        if not theta_obj_in.estimation_scale:
            # Create a copy to avoid modifying the original
            theta_list = [
                {
                    "shared": t["shared"].copy() if t["shared"] is not None else None,
                    "unit_specific": (
                        t["unit_specific"].copy()
                        if t["unit_specific"] is not None
                        else None
                    ),
                }
                for t in theta_list
            ]
            theta_trans_list = rep_unit.par_trans.panel_transform_list(
                theta_list, direction="to_est"
            )
        else:
            theta_trans_list = theta_list

        # Extract shared and unit_specific DataFrames from transformed list
        shared_trans_list = [t.get("shared") for t in theta_trans_list]
        spec_trans_list = [t.get("unit_specific") for t in theta_trans_list]

        # Store original shared/unit_specific for later use
        shared = [t.get("shared") for t in theta_list]
        unit_specific = [t.get("unit_specific") for t in theta_list]

        if all(df is None for df in shared_trans_list):
            n_shared = 0
            shared_array = jnp.zeros((n_reps, 0, J))
            shared_index: list[str] = []
        else:
            shared_index = self.canonical_shared_param_names
            n_shared = len(shared_index)
            # PanelParameters ensures all shared are None or all are not None
            # Cast to list[pd.DataFrame] since we know all are not None here
            shared_trans_list_nonnull = cast(list[pd.DataFrame], shared_trans_list)
            shared_array = jnp.stack(
                [
                    jnp.tile(
                        self._dataframe_to_array_canonical(
                            df, self.canonical_shared_param_names, "shared"
                        ).reshape(n_shared, 1),
                        (1, J),
                    )
                    for df in shared_trans_list_nonnull
                ],
                axis=0,
            )

        if all(df is None for df in spec_trans_list):
            n_spec = 0
            unit_array = jnp.zeros((n_reps, 0, J, U))
            spec_index: list[str] = []
        else:
            spec_index = self.canonical_unit_param_names
            n_spec = len(spec_index)
            # PanelParameters ensures all unit_specific are None or all are not None
            # Cast to list[pd.DataFrame] since we know all are not None here
            spec_trans_list_nonnull = cast(list[pd.DataFrame], spec_trans_list)
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
                    )
                    for df in spec_trans_list_nonnull
                ],
                axis=0,
            )

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
            if unit_traces is None:
                raise ValueError(
                    "Both shared_traces and unit_traces are None; cannot build traces."
                )
            n_reps = unit_traces.shape[0]
            shared_ll = np.sum(unit_traces[:, :, 0, :], axis=-1, keepdims=True)
            shared_traces = shared_ll
            shared_index = []
        if unit_traces is None:
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

        shared_final_logliks = shared_traces[:, -1, 0]
        unit_final_logliks = unit_traces[:, -1, 0, :]

        full_logliks = xr.DataArray(
            jnp.concatenate(
                [shared_final_logliks.reshape(-1, 1), unit_final_logliks], axis=1
            ),
            dims=["replicate", "unit"],
            coords={"replicate": jnp.arange(n_reps), "unit": ["shared"] + unit_names},
        )

        # PanelParameters ensures all shared are None or all are not None
        if shared and shared[0] is not None:
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

        # PanelParameters ensures all unit_specific are None or all are not None
        if unit_specific and unit_specific[0] is not None:
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

        theta_list_out = [
            {
                "shared": shared_list_out[rep] if shared_list_out else None,
                "unit_specific": specific_list_out[rep] if specific_list_out else None,
            }
            for rep in range(n_reps)
        ]

        # unit_final_logliks has shape (n_reps, U)
        logLik_unit_out = np.array(unit_final_logliks)

        self.theta.theta = theta_list_out
        self.theta.logLik_unit = logLik_unit_out
        self.theta.estimation_scale = False

        execution_time = time.time() - start_time

        result = PanelPompMIFResult(
            method="mif",
            execution_time=execution_time,
            key=old_key,
            theta=theta_obj_in,
            shared_traces=shared_da,
            unit_traces=unit_da,
            logLiks=full_logliks,
            J=J,
            M=M,
            rw_sd=rw_sd,
            a=a,
            thresh=thresh,
            block=block,
        )

        self.results_history.add(result)

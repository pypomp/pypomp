import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
import numpy as np
import time
from typing import TYPE_CHECKING

from ..mif import _jv_panel_mif_internal
from ..internal_functions import _shard_rows
from ..RWSigma_class import RWSigma
from ..results import PanelPompPFilterResult, PanelPompMIFResult, ResultsHistory

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
        shared: list[pd.DataFrame] | None = None,
        unit_specific: list[pd.DataFrame] | None = None,
    ) -> list[dict]:
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

    @staticmethod
    def sample_params(
        param_bounds: dict,
        units: list[str],
        n: int,
        key: jax.Array,
        shared_names: list[str] | None = None,
    ) -> tuple[list[pd.DataFrame] | None, list[pd.DataFrame] | None]:
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
            results.loc[:, unit, :] = obj.results_history[-1].logLiks
            obj.results_history = ResultsHistory()

        execution_time = time.time() - start_time

        result = PanelPompPFilterResult(
            method="pfilter",
            execution_time=execution_time,
            key=old_key,
            shared=shared,
            unit_specific=unit_specific,
            logLiks=results,
            J=J,
            reps=reps,
            thresh=thresh,
        )

        self.results_history.add(result)

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

        n_reps = self._get_theta_list_len(shared, unit_specific)

        shared_list = shared if isinstance(shared, list) else None
        spec_list = unit_specific if isinstance(unit_specific, list) else None
        shared_trans_list, spec_trans_list = rep_unit.par_trans.panel_transform_list(
            shared_list, spec_list, direction="to_est"
        )

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
                    )
                    for df in spec_trans_list
                ],
                axis=0,
            )

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )

        old_key = key
        keys = jax.random.split(key, n_reps)

        shared_sharded = _shard_rows(shared_array)
        unit_sharded = _shard_rows(unit_array)
        (
            shared_array_f,
            unit_array_f,
            shared_traces,
            unit_traces,
        ) = _jv_panel_mif_internal(
            shared_sharded,
            unit_sharded,
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

        result = PanelPompMIFResult(
            method="mif",
            execution_time=execution_time,
            key=old_key,
            shared=shared,
            unit_specific=unit_specific,
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

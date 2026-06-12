from __future__ import annotations
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax
import xarray as xr
from typing import (
    Mapping,
    Sequence,
    Union,
    Literal,
    Any,
    cast,
    overload,
)

from .base import ParameterSet
from ..par_trans import ParTrans
from pypomp.types import Numeric


def _standardize_pomp_theta(
    theta: Mapping[str, Numeric]
    | Sequence[Mapping[str, Numeric]]
    | PompParameters
    | None,
) -> xr.DataArray:
    if isinstance(theta, xr.DataArray):
        return theta

    if theta is None:
        raise ValueError("theta cannot be None")

    theta_dicts: list[dict[str, Numeric]] = []
    if isinstance(theta, Mapping):
        theta_dicts = [dict(theta)]
    elif isinstance(theta, (list, tuple)):
        theta_dicts = [dict(t) for t in cast(Any, theta)]
    else:
        try:
            theta_dicts = [dict(t) for t in cast(Any, theta)]
        except (TypeError, ValueError):
            raise TypeError(
                "theta must be a Mapping, Sequence of Mappings, or PompParameters"
            )

    # Validate elements are dictionaries and not empty
    if len(theta_dicts) == 0:
        raise ValueError("theta cannot be empty")
    if not all(isinstance(t, dict) for t in theta_dicts):
        raise TypeError("All elements in theta must be dictionaries")

    # Cast to floats (making copy to prevent side-effects)
    clean_dicts = []
    for i, t in enumerate(theta_dicts):
        t_copy = {}
        for key, value in t.items():
            if isinstance(value, bool):
                raise TypeError(
                    f"Parameter '{key}' at index {i} is not a float: got bool"
                )
            try:
                t_copy[key] = float(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Parameter '{key}' at index {i} is not a float: got {type(value).__name__}"
                )
        clean_dicts.append(t_copy)

    # Ensure all dicts have identical keys
    first_keys = set(clean_dicts[0].keys())
    for i, t in enumerate(clean_dicts[1:]):
        if set(t.keys()) != first_keys:
            raise ValueError(
                f"Parameter set at index {i + 1} has different keys than the first set. "
                f"Expected {first_keys}, got {set(t.keys())}"
            )

    reps = len(clean_dicts)
    param_names = list(clean_dicts[0].keys())

    # Contiguous array of shape (J, 1, P)
    values = pd.DataFrame(clean_dicts)[param_names].values[:, np.newaxis, :]

    return xr.DataArray(
        values,
        dims=["theta_idx", "unit", "parameter"],
        coords={
            "theta_idx": np.arange(reps),
            "unit": ["shared"],
            "parameter": param_names,
        },
    )


class PompParameters(ParameterSet[xr.DataArray]):
    """
    Manages parameters for a standard Pomp model.

    Internal storage is a 3D ``xarray.DataArray`` with dimensions
    ``("theta_idx", "unit", "parameter")``, where ``"unit"`` is always ``"shared"``.

    Parameters
    ----------
    theta : Mapping[str, Numeric] | Sequence[Mapping[str, Numeric]] | PompParameters | xr.DataArray | None
        Parameters for the model. Accepts:

        - A single dictionary: ``dict[str, Numeric]``
        - A list of dictionaries: ``list[dict[str, Numeric]]``
        - An existing :class:`~pypomp.core.parameters.PompParameters` object
        - An ``xarray.DataArray`` with dimensions ``("theta_idx", "unit", "parameter")``
    logLik : np.ndarray, optional
        A numpy array of log-likelihoods associated with each parameter set.
    estimation_scale : bool, optional
        Whether the parameters are on the estimation scale. Defaults to False.
    """

    _data: xr.DataArray
    estimation_scale: bool
    _logLik: np.ndarray

    def __init__(
        self,
        theta: Mapping[str, Numeric]
        | Sequence[Mapping[str, Numeric]]
        | PompParameters
        | xr.DataArray
        | None,
        logLik: np.ndarray | None = None,
        estimation_scale: bool = False,
    ):
        if theta is None:
            self._data = xr.DataArray(
                np.empty((0, 1, 0)),
                dims=["theta_idx", "unit", "parameter"],
                coords={"theta_idx": [], "unit": ["shared"], "parameter": []},
            )
            self._logLik = np.full(0, np.nan)
            self.estimation_scale = False
            return

        if isinstance(theta, PompParameters):
            self._data = theta._data.copy(deep=True)
            self._logLik = (
                theta.logLik.copy()
                if logLik is None
                else self._format_logLik(logLik, self._data.sizes["theta_idx"])
            )
            self.estimation_scale = theta.estimation_scale
            return

        if isinstance(theta, xr.DataArray):
            theta = theta.astype(float)
            if theta.ndim == 1:
                if "parameter" not in theta.dims:
                    if len(theta.dims) == 1:
                        theta = theta.rename({theta.dims[0]: "parameter"})
                    else:
                        raise ValueError("1D DataArray must have 'parameter' dimension")
                theta_expanded = theta.expand_dims(dim={"theta_idx": [0]}, axis=0)
                self._data = theta_expanded.expand_dims(
                    dim={"unit": ["shared"]}, axis=1
                ).copy(deep=True)
            elif theta.ndim == 2:
                dims = list(theta.dims)
                if "parameter" not in dims:
                    raise ValueError("2D DataArray must have 'parameter' dimension")
                if "theta_idx" not in dims:
                    other_dim = [d for d in dims if d != "parameter"][0]
                    theta = theta.rename({other_dim: "theta_idx"})
                theta = theta.transpose("theta_idx", "parameter")
                self._data = theta.expand_dims(dim={"unit": ["shared"]}, axis=1).copy(
                    deep=True
                )
            elif theta.ndim == 3:
                dims = list(theta.dims)
                if set(dims) == {"theta_idx", "unit", "parameter"}:
                    self._data = theta.transpose("theta_idx", "unit", "parameter").copy(
                        deep=True
                    )
                else:
                    self._data = theta.copy(deep=True)
            else:
                raise ValueError("DataArray must be 1D, 2D, or 3D")
        else:
            self._data = _standardize_pomp_theta(theta)

        self.estimation_scale = estimation_scale
        self._logLik = self._format_logLik(logLik, self._data.sizes["theta_idx"])

    def _format_logLik(self, ll: np.ndarray | None, n_reps: int) -> np.ndarray:
        """Helper to standardize logLik input."""
        if ll is None:
            return np.full(n_reps, np.nan)

        ll = np.array(ll, dtype=float)

        if ll.ndim == 0:  # Handle single scalar input (broadcast)
            return np.full(n_reps, ll)

        if len(ll) != n_reps:
            raise ValueError(
                f"Length of logLik ({len(ll)}) must match parameters ({n_reps})"
            )
        return ll

    def to_jax_array(self, param_names: list[str] | None = None, **kwargs) -> jax.Array:
        """
        Convert to a JAX array matching the order of param_names.

        Parameters
        ----------
        param_names : list[str], optional
            A list of parameter names in the desired order. If None (default),
            returns the array matching the canonical order of parameters.

        Returns
        -------
        jax.Array
            A JAX array of shape (num_theta_idx, n_params).
        """
        if param_names is None:
            param_names = self.get_param_names()
        try:
            ordered_values = self._data.sel(parameter=param_names).values[:, 0, :]
        except KeyError as e:
            raise KeyError(
                f"Parameter {e} expected by model but missing from parameter set."
            )

        return jnp.array(ordered_values)

    @property
    def logLik(self) -> np.ndarray:
        """
        Get or set the log-likelihoods for each parameter set (theta_idx).
        """
        return self._logLik

    @logLik.setter
    def logLik(self, value):
        self._logLik = self._format_logLik(value, self.num_replicates())

    def subset(self, indices: Union[int, list[int], slice]) -> "PompParameters":
        """
        Return a new PompParameters object with the specified parameter set (theta_idx) indices.
        """
        if isinstance(indices, int):
            indices = [indices]

        sub_data = self._data.isel(theta_idx=indices)
        sub_data.coords["theta_idx"] = np.arange(sub_data.sizes["theta_idx"])
        sub_logLik = self._logLik[indices]

        return PompParameters(
            sub_data, logLik=sub_logLik, estimation_scale=self.estimation_scale
        )

    @overload
    def params(self, as_list: Literal[True] = True) -> list[dict[str, float]]: ...

    @overload
    def params(self, as_list: Literal[False]) -> xr.DataArray: ...

    @overload
    def params(self, as_list: bool = True) -> list[dict[str, float]] | xr.DataArray: ...

    def params(self, as_list: bool = True) -> list[dict[str, float]] | xr.DataArray:
        """
        Get the parameters in this set.

        Parameters
        ----------
        as_list : bool, default True
            If True, returns the parameters as a list of dictionaries mapping parameter names to floats.
            If False, returns the internal 3D xarray DataArray.

        Returns
        -------
        list[dict[str, float]] | xr.DataArray
            The parameters either as a list of dictionaries or as a DataArray.
        """
        return super().params(as_list)

    def set_params(
        self,
        value: Mapping[str, Numeric] | Sequence[Mapping[str, Numeric]] | xr.DataArray,
    ) -> None:
        """
        Set or overwrite the parameter values.

        Parameters
        ----------
        value : Mapping[str, Numeric] | Sequence[Mapping[str, Numeric]] | xr.DataArray
            The new parameter values. Accepts:
            - A single dictionary: ``dict[str, Numeric]``
            - A list of dictionaries: ``list[dict[str, Numeric]]`` (must have identical keys)
            - An ``xarray.DataArray`` of shape ``(theta_idx, unit, parameter)``
        """
        super().set_params(value)

    def _to_list(self) -> list[dict[str, float]]:
        """Return the parameter sets as a list of dictionaries."""
        return cast(
            list[dict[str, float]],
            pd.DataFrame(
                self._data.values[:, 0, :], columns=self.get_param_names()
            ).to_dict(orient="records"),
        )

    def _replicated_logLik(self, n: int) -> dict[str, np.ndarray]:
        return {"logLik": np.tile(self._logLik, n)}

    def _slice_logLik(self, indices: np.ndarray) -> None:
        self._logLik = self._logLik[indices]

    def _eq_logLik(self, other: "PompParameters") -> bool:
        return np.array_equal(self._logLik, other._logLik, equal_nan=True)

    def _getitem_int(self, index: int) -> dict[str, float]:
        return dict(zip(self.get_param_names(), self._data.values[index, 0]))

    def _transform_and_load(
        self,
        par_trans: ParTrans,
        param_list: list[Any],
        direction: Literal["to_est", "from_est"],
    ) -> None:
        transformed_list = [
            par_trans._to_floats(theta_i, direction) for theta_i in param_list
        ]
        self._data = _standardize_pomp_theta(transformed_list)

    def _set_theta(self, value: Any) -> None:
        if value is None:
            raise ValueError("theta cannot be None")
        self._data = _standardize_pomp_theta(value)
        self._logLik = self._format_logLik(None, self.num_replicates())

    @staticmethod
    def merge(*param_objs: "PompParameters") -> "PompParameters":
        """
        Merge replications from an arbitrary number of PompParameters objects.
        """
        if len(param_objs) == 0:
            raise ValueError("At least one PompParameters object must be provided.")
        first = param_objs[0]

        for obj in param_objs:
            if not isinstance(obj, PompParameters):
                raise TypeError("All merged objects must be of type PompParameters.")
            if obj.get_param_names() != first.get_param_names():
                raise ValueError(
                    "All PompParameters objects must have the same canonical parameter names."
                )
            if obj.estimation_scale != first.estimation_scale:
                raise ValueError(
                    "All PompParameters objects must have the same estimation scale."
                )

        merged_data = xr.concat([obj._data for obj in param_objs], dim="theta_idx")
        merged_data.coords["theta_idx"] = np.arange(merged_data.sizes["theta_idx"])

        all_logLik = [obj._logLik for obj in param_objs]
        merged_logLik = np.concatenate(all_logLik) if all_logLik else np.array([])
        return PompParameters(
            merged_data, logLik=merged_logLik, estimation_scale=first.estimation_scale
        )

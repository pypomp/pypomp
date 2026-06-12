"""
This module defines the parameter classes for Pomp and PanelPomp models.
It handles input validation, standardization, and conversion to JAX arrays.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax
import xarray as xr
from typing import (
    Union,
    Literal,
    Mapping,
    Sequence,
    cast,
    Iterator,
    Any,
    Generic,
    TypeVar,
    overload,
)
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from .par_trans import ParTrans
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
            if isinstance(value, (int, np.number, jax.Array)) and not isinstance(
                value, bool
            ):
                try:
                    t_copy[key] = float(value)
                except (TypeError, ValueError):
                    t_copy[key] = value
            else:
                t_copy[key] = value

            if not isinstance(t_copy[key], float):
                raise TypeError(
                    f"Parameter '{key}' at index {i} is not a float: got {type(t_copy[key]).__name__}"
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
    values = np.zeros((reps, 1, len(param_names)))
    for j, t in enumerate(clean_dicts):
        for p_idx, name in enumerate(param_names):
            values[j, 0, p_idx] = t[name]

    return xr.DataArray(
        values,
        dims=["theta_idx", "unit", "parameter"],
        coords={
            "theta_idx": np.arange(reps),
            "unit": ["shared"],
            "parameter": param_names,
        },
    )


def _standardize_panel_theta(
    theta: Union[
        dict[str, pd.DataFrame | None],
        list[dict[str, pd.DataFrame | None]],
        None,
    ],
) -> tuple[xr.Dataset, list[str], list[str]]:
    if theta is None:
        shared_da = xr.DataArray(
            np.empty((0, 0)),
            dims=["theta_idx", "parameter"],
            coords={"theta_idx": [], "parameter": []},
        )
        unit_specific_da = xr.DataArray(
            np.empty((0, 0, 0)),
            dims=["theta_idx", "unit", "parameter"],
            coords={"theta_idx": [], "unit": [], "parameter": []},
        )
        ds = xr.Dataset(
            data_vars={
                "shared": shared_da,
                "unit_specific": unit_specific_da,
            }
        )
        return ds, [], []

    if isinstance(theta, dict):
        theta_list = [theta]
    else:
        theta_list = list(theta)

    if not isinstance(theta_list, list):
        raise TypeError("theta must be a dictionary or a list of dictionaries")

    # Copy the structures, convert to floats, and validate keys to avoid side-effects
    clean_theta = []
    for i, t in enumerate(theta_list):
        keys = set(t.keys())
        if keys != {"shared", "unit_specific"}:
            raise ValueError(
                f"Each parameter dictionary must have exactly the keys 'shared' and 'unit_specific'. "
                f"Found keys {keys} in item {i}."
            )
        if not all(isinstance(v, (pd.DataFrame, type(None))) for v in t.values()):
            raise TypeError(
                f"All values in each dictionary must be None or pd.DataFrames. "
                f"Found values {t.values()} of type {type(t.values())} in item {i}."
            )
        t_copy = {"shared": t["shared"], "unit_specific": t["unit_specific"]}
        if t_copy["shared"] is not None:
            t_copy["shared"] = t_copy["shared"].astype(float)
        if t_copy["unit_specific"] is not None:
            t_copy["unit_specific"] = t_copy["unit_specific"].astype(float)
        clean_theta.append(t_copy)

    # Consistency checks
    shared_none = [t["shared"] is None for t in clean_theta]
    unit_none = [t["unit_specific"] is None for t in clean_theta]

    some_shared_none = any(shared_none) and not all(shared_none)
    some_unit_specific_none = any(unit_none) and not all(unit_none)
    if some_shared_none:
        raise ValueError(
            "Some, but not all, shared parameters are None. This is not supported."
        )
    if some_unit_specific_none:
        raise ValueError(
            "Some, but not all, unit-specific parameters are None. This is not supported."
        )

    # Check dataframe consistency
    ref = clean_theta[0]
    if ref["shared"] is not None:
        ref_s_idx = ref["shared"].index
        ref_s_cols = ref["shared"].columns
        if len(ref_s_cols) != 1:
            raise ValueError("Shared parameters must have exactly one column.")
    else:
        ref_s_idx = []

    if ref["unit_specific"] is not None:
        ref_u_idx = ref["unit_specific"].index
        ref_u_cols = ref["unit_specific"].columns
    else:
        ref_u_idx = []
        ref_u_cols = []

    shared_param_names = set(ref_s_idx)
    unit_param_names = set(ref_u_idx)
    overlap = shared_param_names.intersection(unit_param_names)
    if overlap:
        raise ValueError(
            f"Parameter name(s) found in both shared and unit-specific parameters: {sorted(overlap)}"
        )

    for i, t in enumerate(clean_theta[1:], 1):
        if t["shared"] is not None:
            if not t["shared"].index.equals(ref_s_idx):
                raise ValueError(f"Shared parameter index mismatch at replicate {i}.")
        if t["unit_specific"] is not None:
            if not t["unit_specific"].index.equals(ref_u_idx):
                raise ValueError(f"Unit parameter index mismatch at replicate {i}.")
            if not t["unit_specific"].columns.equals(ref_u_cols):
                raise ValueError(f"Unit columns mismatch at replicate {i}.")

    # Gather names
    shared_names_list = list(ref_s_idx)
    unit_specific_names_list = list(ref_u_idx)

    if ref["unit_specific"] is not None:
        unit_names = list(ref_u_cols)
    else:
        unit_names = []

    reps = len(clean_theta)

    shared_values = np.zeros((reps, len(shared_names_list)))
    unit_values = np.zeros((reps, len(unit_names), len(unit_specific_names_list)))

    shared_param_to_idx = {name: idx for idx, name in enumerate(shared_names_list)}
    specific_param_to_idx = {
        name: idx for idx, name in enumerate(unit_specific_names_list)
    }
    unit_to_idx = {name: idx for idx, name in enumerate(unit_names)}

    for j, t in enumerate(clean_theta):
        s_df = t["shared"]
        u_df = t["unit_specific"]

        if s_df is not None:
            s_col = s_df.columns[0]
            for p_name in shared_names_list:
                shared_values[j, shared_param_to_idx[p_name]] = float(
                    s_df.loc[p_name, s_col]
                )

        if u_df is not None:
            for p_name in unit_specific_names_list:
                p_idx = specific_param_to_idx[p_name]
                for u_name in unit_names:
                    unit_values[j, unit_to_idx[u_name], p_idx] = float(
                        u_df.loc[p_name, u_name]
                    )

    shared_da = xr.DataArray(
        shared_values,
        dims=["theta_idx", "parameter"],
        coords={
            "theta_idx": np.arange(reps),
            "parameter": shared_names_list,
        },
    )
    unit_specific_da = xr.DataArray(
        unit_values,
        dims=["theta_idx", "unit", "parameter"],
        coords={
            "theta_idx": np.arange(reps),
            "unit": unit_names,
            "parameter": unit_specific_names_list,
        },
    )

    ds = xr.Dataset(
        data_vars={
            "shared": shared_da,
            "unit_specific": unit_specific_da,
        }
    )
    ds.attrs["shared_names"] = shared_names_list
    ds.attrs["unit_specific_names"] = unit_specific_names_list
    return ds, shared_names_list, unit_specific_names_list


T_data = TypeVar("T_data", xr.DataArray, xr.Dataset)


class ParameterSet(ABC, Generic[T_data]):
    """
    Abstract base class for parameter sets used in POMP models.

    All parameter sets store parameters internally as a 3D ``xarray.DataArray``
    with dimensions ``("theta_idx", "unit", "parameter")``:

    - ``theta_idx``: Coordinate indexing each parameter set/replicate.
    - ``unit``: Coordinate indexing model units ("shared" or specific unit names).
    - ``parameter``: Coordinate indexing parameter names.
    """

    _data: T_data
    estimation_scale: bool

    @abstractmethod
    def to_jax_array(self, param_names: list[str], **kwargs) -> jax.Array:
        """
        Converts the parameters to a JAX array suitable for model functions.

        Args:
            param_names: A list of canonical parameter names expected by the model.
            **kwargs: Additional context required for conversion (e.g. unit names).

        Returns:
            A JAX array representing the parameters.
            - For Pomp: Shape (num_theta_idx, n_params)
            - For PanelPomp: Shape (num_theta_idx, n_units, n_params)
        """
        pass

    def num_replicates(self) -> int:
        """Returns the number of parameter sets/replicates."""
        return len(self)

    def num_params(self) -> int:
        """Return the number of canonical parameters."""
        return len(self.get_param_names())

    def get_param_names(self) -> list[str]:
        """Return the list of parameter names contained in this set."""
        if isinstance(self._data, xr.Dataset):
            shared = (
                list(self._data["shared"].coords["parameter"].values)
                if "shared" in self._data
                else []
            )
            unit_spec = (
                list(self._data["unit_specific"].coords["parameter"].values)
                if "unit_specific" in self._data
                else []
            )
            return sorted(list(set(shared + unit_spec)))
        return list(self._data.coords["parameter"].values)

    def __len__(self) -> int:
        """Return the number of parameter sets/replicates."""
        return self._data.sizes["theta_idx"]

    def __iter__(self) -> Iterator[Any]:
        """Support iteration over parameter sets."""
        return iter(self._to_list())

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k == "_data":
                setattr(new_obj, k, v.copy(deep=False))
            else:
                setattr(new_obj, k, v)
        return new_obj

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            if k == "_data":
                setattr(new_obj, k, v.copy(deep=True))
            else:
                setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj

    def __mul__(self, n: int) -> Self:
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("Multiplication factor must be non-negative")
        if n == 0:
            raise ValueError("Cannot create empty ParameterSet")

        if isinstance(self._data, xr.Dataset):
            new_data = xr.concat([self._data] * n, dim="theta_idx")
        else:
            new_data = xr.concat([self._data] * n, dim="theta_idx")
        new_data.coords["theta_idx"] = np.arange(new_data.sizes["theta_idx"])

        extra_kwargs = self._replicated_logLik(n)
        cls = cast(Any, self.__class__)
        return cls(new_data, estimation_scale=self.estimation_scale, **extra_kwargs)

    def __rmul__(self, n: int) -> Self:
        return self.__mul__(n)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n{self._data.__repr__()}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(\n{self._data.__str__()}\n)"

    def prune(self, n: int = 1, refill: bool = True) -> None:
        """
        Replace internal parameter sets with the top `n` based on stored log-likelihoods.

        Args:
            n: Number of top-performing parameter sets to keep.
            refill: If True, duplicate the top `n` sets to restore the original length.
        """
        n_reps = self.num_replicates()
        if n_reps == 0:
            raise ValueError("No parameter sets available to prune.")
        if n < 1:
            raise ValueError("n must be at least 1.")

        log_lik = self.logLik
        if log_lik is None or np.all(np.isnan(log_lik)):
            if isinstance(self, PompParameters):
                raise ValueError(
                    "No valid log-likelihoods available to prune (all nan)."
                )
            log_lik = np.zeros(n_reps)

        top_indices = log_lik.argsort()[-n:][::-1]

        if refill:
            prev_len = n_reps
            repeats = (prev_len + n - 1) // n
            new_indices = np.tile(top_indices, repeats)[:prev_len]
        else:
            new_indices = top_indices

        self._data = self._data.isel(theta_idx=new_indices)
        self._data.coords["theta_idx"] = np.arange(len(new_indices))
        self._slice_logLik(new_indices)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.estimation_scale != other.estimation_scale:
            return False
        if self.get_param_names() != other.get_param_names():
            return False
        if not self._data.equals(other._data):
            return False
        return self._eq_logLik(other)

    def __getitem__(self, index: int | slice | list[int]) -> Any:
        if isinstance(index, (slice, list, np.ndarray)):
            return self.subset(index)
        return self._getitem_int(int(index))

    def transform(
        self,
        par_trans: ParTrans,
        direction: Literal["to_est", "from_est"] | None = None,
    ) -> None:
        """
        Transform the parameters to or from the estimation parameter space.
        """
        auto = direction is None
        if auto:
            direction = "from_est" if self.estimation_scale else "to_est"

        if (direction == "to_est" and not self.estimation_scale) or (
            direction == "from_est" and self.estimation_scale
        ):
            param_list = self._to_list()
            self._transform_and_load(par_trans, param_list, direction)
            self.estimation_scale = not self.estimation_scale

    @overload
    def params(self, as_list: Literal[True] = True) -> list[Any]: ...

    @overload
    def params(self, as_list: Literal[False]) -> T_data: ...

    @overload
    def params(self, as_list: bool = True) -> list[Any] | T_data: ...

    def params(self, as_list: bool = True) -> list[Any] | T_data:
        """
        Get the parameter values in this parameter set.

        Parameters
        ----------
        as_list : bool, default True
            If True, returns the parameters as a list of Python dictionaries.
            If False, returns the internal xarray representation (DataArray or Dataset).

        Returns
        -------
        list[Any] | xr.DataArray | xr.Dataset
            The parameters either as a list of dictionaries or as an xarray object.
        """
        if as_list:
            return self._to_list()
        return self._data

    def set_params(self, value: Any) -> None:
        self._set_theta(value)

    @property
    @abstractmethod
    def logLik(self) -> np.ndarray:
        pass

    @abstractmethod
    def _to_list(self) -> list[Any]:
        pass

    @abstractmethod
    def subset(self, indices: Union[int, list[int], slice]) -> Self:
        pass

    @abstractmethod
    def _replicated_logLik(self, n: int) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def _slice_logLik(self, indices: np.ndarray) -> None:
        pass

    @abstractmethod
    def _eq_logLik(self, other: Any) -> bool:
        pass

    @abstractmethod
    def _getitem_int(self, index: int) -> Any:
        pass

    @abstractmethod
    def _transform_and_load(
        self,
        par_trans: ParTrans,
        param_list: list[Any],
        direction: Literal["to_est", "from_est"],
    ) -> None:
        pass

    @abstractmethod
    def _set_theta(self, value: Any) -> None:
        pass


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

    def to_jax_array(self, param_names: list[str], **kwargs) -> jax.Array:
        """
        Convert to JAX array matching the order of param_names.

        Returns shape (num_theta_idx, n_params).
        """
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

    def to_jax_array_canonical(self) -> jax.Array:
        """
        Convert to a JAX array matching the canonical parameter names order.
        """
        return self.to_jax_array(list(self._data.coords["parameter"].values))

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

    def _to_list(self) -> list[dict[str, float]]:
        """Return the parameter sets as a list of dictionaries."""
        param_names = self.get_param_names()
        values = self._data.values
        return [
            {name: float(values[j, 0, p_idx]) for p_idx, name in enumerate(param_names)}
            for j in range(self.num_replicates())
        ]

    @property
    def _params(self) -> list[dict[str, float]]:
        return self._to_list()

    @_params.setter
    def _params(self, value):
        self.set_params(value)

    def _replicated_logLik(self, n: int) -> dict[str, np.ndarray]:
        return {"logLik": np.tile(self._logLik, n)}

    def _slice_logLik(self, indices: np.ndarray) -> None:
        self._logLik = self._logLik[indices]

    def _eq_logLik(self, other: "PompParameters") -> bool:
        return np.array_equal(self._logLik, other._logLik, equal_nan=True)

    def _getitem_int(self, index: int) -> dict[str, float]:
        param_names = self.get_param_names()
        values = self._data.values
        return {
            name: float(values[index, 0, p_idx])
            for p_idx, name in enumerate(param_names)
        }

    def _transform_and_load(
        self,
        par_trans: ParTrans,
        param_list: list[Any],
        direction: Literal["to_est", "from_est"],
    ) -> None:
        transformed_list = [
            par_trans.to_floats(theta_i, direction) for theta_i in param_list
        ]
        self._data = _standardize_pomp_theta(transformed_list)

    def _set_theta(self, value: Any) -> None:
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


class PanelParameters(ParameterSet[xr.Dataset]):
    """
    Manages parameters for PanelPomp models.

    Internal storage is a 3D ``xarray.DataArray`` with dimensions
    ``("theta_idx", "unit", "parameter")``.

    Parameters
    ----------
    theta : PanelParameters | dict | list | xr.DataArray, optional
        Parameters for the panel model. Accepts:

        - A single dictionary with ``"shared"`` and ``"unit_specific"`` keys (each containing a DataFrame).
        - A list of such dictionaries.
        - An existing :class:`~pypomp.core.parameters.PanelParameters` object.
        - An existing ``xarray.DataArray`` with dimensions ``("theta_idx", "unit", "parameter")``.
    logLik_unit : np.ndarray, optional
        A numpy array of unit-specific log-likelihoods of shape ``(num_theta_idx, n_units)``.
    estimation_scale : bool, optional
        Whether the parameters are on the estimation scale. Defaults to False.
    """

    _data: xr.Dataset
    estimation_scale: bool
    _logLik_unit: np.ndarray
    _logLik: np.ndarray
    _canonical_shared_param_names: list[str]
    _canonical_unit_param_names: list[str]
    _canonical_param_names: list[str]

    def __init__(
        self,
        theta: Union[
            dict[str, pd.DataFrame | None],
            list[dict[str, pd.DataFrame | None]],
            "PanelParameters",
            xr.Dataset,
            None,
        ],
        logLik_unit: np.ndarray | None = None,
        estimation_scale: bool = False,
    ):
        if isinstance(theta, PanelParameters):
            self._data = theta._data.copy(deep=True)
            self.estimation_scale = theta.estimation_scale
            self._canonical_shared_param_names = list(
                theta._canonical_shared_param_names
            )
            self._canonical_unit_param_names = list(theta._canonical_unit_param_names)
            self._canonical_param_names = list(theta._canonical_param_names)
            self._logLik_unit = (
                theta.logLik_unit.copy()
                if logLik_unit is None
                else self._format_logLik_unit(
                    logLik_unit, self._data.sizes["theta_idx"]
                )
            )
            self._logLik = self._logLik_unit.sum(axis=1)
            return

        if isinstance(theta, xr.Dataset):
            self._data = theta.copy(deep=True)
            raw_s = self._data.attrs.get("shared_names")
            if raw_s is None:
                raw_s = (
                    list(self._data["shared"].coords["parameter"].values)
                    if "shared" in self._data
                    else []
                )
            self._canonical_shared_param_names = [str(x) for x in raw_s]

            raw_u = self._data.attrs.get("unit_specific_names")
            if raw_u is None:
                raw_u = (
                    list(self._data["unit_specific"].coords["parameter"].values)
                    if "unit_specific" in self._data
                    else []
                )
            self._canonical_unit_param_names = [str(x) for x in raw_u]
        else:
            ds, s_names, u_names = _standardize_panel_theta(theta)
            self._data = ds
            self._canonical_shared_param_names = [str(x) for x in s_names]
            self._canonical_unit_param_names = [str(x) for x in u_names]

        self.estimation_scale = estimation_scale
        self._logLik_unit = self._format_logLik_unit(
            logLik_unit, self._data.sizes["theta_idx"]
        )
        self._logLik = self._logLik_unit.sum(axis=1)
        self._canonical_param_names = list(
            set(self._canonical_shared_param_names + self._canonical_unit_param_names)
        )

    def _format_logLik_unit(
        self, ll_unit: np.ndarray | None, n_reps: int
    ) -> np.ndarray:
        """Standardize logLik dimensions."""
        n_units = 0
        if n_reps > 0:
            n_units = len(self.get_unit_names())

        if ll_unit is None:
            return np.full((n_reps, n_units), np.nan)

        ll_unit = np.array(ll_unit, dtype=float)
        if ll_unit.ndim == 1 and n_reps == 1:
            return ll_unit.reshape(1, -1)
        if ll_unit.shape != (n_reps, n_units):
            if n_units == 0 and ll_unit.size == 0:
                return np.empty((n_reps, 0))
            raise ValueError(
                f"logLik_unit shape mismatch: {ll_unit.shape} vs ({n_reps}, {n_units})"
            )
        return ll_unit

    @property
    def logLik(self) -> np.ndarray:
        """
        Get the overall log-likelihood for each parameter set (theta_idx).
        """
        return self._logLik

    @logLik.setter
    def logLik(self, value):
        # Read-only derived property
        pass

    @property
    def logLik_unit(self) -> np.ndarray:
        """
        Get or set the unit-specific log-likelihoods for each parameter set (theta_idx).
        """
        return self._logLik_unit

    @logLik_unit.setter
    def logLik_unit(self, value):
        self._logLik_unit = self._format_logLik_unit(value, self.num_replicates())
        self._logLik = self._logLik_unit.sum(axis=1)

    @property
    def _theta(self) -> list[dict[str, pd.DataFrame | None]]:
        return self._to_list()

    @_theta.setter
    def _theta(self, value):
        self.set_params(value)

    def get_shared_param_names(self) -> list[str]:
        """Return the list of shared parameter names."""
        return self._canonical_shared_param_names

    def get_unit_param_names(self) -> list[str]:
        """Return the list of unit-specific parameter names."""
        return self._canonical_unit_param_names

    def get_unit_names(self) -> list[str]:
        """Return the list of unit names."""
        if (
            "unit_specific" in self._data
            and "unit" in self._data["unit_specific"].coords
        ):
            return list(self._data["unit_specific"].coords["unit"].values)
        return []

    def subset(self, indices: Union[int, list[int], slice]) -> "PanelParameters":
        """
        Return a new PanelParameters object with the specified parameter set (theta_idx) indices.
        """
        if isinstance(indices, int):
            indices = [indices]

        sub_data = self._data.isel(theta_idx=indices)
        sub_data.coords["theta_idx"] = np.arange(sub_data.sizes["theta_idx"])

        sub_ll = self._logLik_unit[indices]
        return PanelParameters(
            sub_data, logLik_unit=sub_ll, estimation_scale=self.estimation_scale
        )

    def to_jax_array(
        self, param_names: list[str], unit_names: list[str] | None = None, **kwargs
    ) -> jax.Array:
        """
        Convert to a JAX array matching the order of param_names and unit_names.

        Returns shape (num_theta_idx, n_units, n_params).
        """
        reps = self.num_replicates()
        if reps == 0:
            return jnp.empty((0, 0, 0))

        if unit_names is None:
            existing_units = self.get_unit_names()
            if not existing_units:
                raise ValueError(
                    "unit_names required when no unit_specific parameters exist"
                )
            unit_names = existing_units

        n_units = len(unit_names)
        n_params = len(param_names)

        shared_keys = set(self._canonical_shared_param_names)
        specific_keys = set(self._canonical_unit_param_names)

        out_array = np.zeros((reps, n_units, n_params))

        for p_idx, p_name in enumerate(param_names):
            if p_name in specific_keys:
                try:
                    out_array[:, :, p_idx] = (
                        self._data["unit_specific"]
                        .sel(parameter=p_name, unit=unit_names)
                        .values
                    )
                except KeyError as e:
                    if (
                        "unit_specific" not in self._data
                        or p_name
                        not in self._data["unit_specific"].coords["parameter"].values
                    ):
                        raise KeyError(f"Parameter '{p_name}' not found.")
                    existing_units = list(
                        self._data["unit_specific"].coords["unit"].values
                    )
                    for u in unit_names:
                        if u not in existing_units:
                            raise KeyError(f"Unit mismatch for parameter {p_name}")
                    raise e
            elif p_name in shared_keys:
                shared_vals = self._data["shared"].sel(parameter=p_name).values
                out_array[:, :, p_idx] = np.broadcast_to(
                    shared_vals[:, None], (reps, n_units)
                )
            else:
                raise KeyError(f"Parameter '{p_name}' not found.")

        return jnp.array(out_array)

    def mix_and_match(self) -> None:
        """
        Mixes unit-specific and shared parameters independently by sorting each
        unit's unit-specific parameters and the shared parameters in descending
        order of their respective log-likelihood contribution.
        """
        unit_names = self.get_unit_names()
        if self.num_replicates() == 0:
            return

        shared_ranks = self._logLik.argsort()[::-1]

        unit_ranks = {}
        for u_idx, unit in enumerate(unit_names):
            unit_ranks[unit] = self._logLik_unit[:, u_idx].argsort()[::-1]

        reps = self.num_replicates()

        shared_keys = self._canonical_shared_param_names
        specific_keys = self._canonical_unit_param_names

        new_shared_values = np.zeros((reps, len(shared_keys)))
        new_unit_values = np.zeros((reps, len(unit_names), len(specific_keys)))
        new_ll_unit = np.zeros_like(self._logLik_unit)

        shared_param_to_idx = {name: idx for idx, name in enumerate(shared_keys)}
        specific_param_to_idx = {name: idx for idx, name in enumerate(specific_keys)}

        for i in range(reps):
            s_idx = shared_ranks[i]
            for p_name in shared_keys:
                new_shared_values[i, shared_param_to_idx[p_name]] = float(
                    self._data["shared"]
                    .isel(theta_idx=s_idx)
                    .sel(parameter=p_name)
                    .values
                )

            for u_idx, unit in enumerate(unit_names):
                u_best_idx = unit_ranks[unit][i]
                new_ll_unit[i, u_idx] = self._logLik_unit[u_best_idx, u_idx]

                for p_name in specific_keys:
                    p_idx = specific_param_to_idx[p_name]
                    new_unit_values[i, u_idx, p_idx] = float(
                        self._data["unit_specific"]
                        .isel(theta_idx=u_best_idx)
                        .sel(parameter=p_name, unit=unit)
                        .values
                    )

        shared_da = xr.DataArray(
            new_shared_values,
            dims=["theta_idx", "parameter"],
            coords={
                "theta_idx": np.arange(reps),
                "parameter": shared_keys,
            },
        )
        unit_specific_da = xr.DataArray(
            new_unit_values,
            dims=["theta_idx", "unit", "parameter"],
            coords={
                "theta_idx": np.arange(reps),
                "unit": unit_names,
                "parameter": specific_keys,
            },
        )
        self._data = xr.Dataset(
            data_vars={
                "shared": shared_da,
                "unit_specific": unit_specific_da,
            }
        )

        self._logLik_unit = new_ll_unit
        self._logLik = new_ll_unit.sum(axis=1)

    @overload
    def params(
        self, as_list: Literal[True] = True
    ) -> list[dict[str, pd.DataFrame | None]]: ...

    @overload
    def params(self, as_list: Literal[False]) -> xr.Dataset: ...

    @overload
    def params(
        self, as_list: bool = True
    ) -> list[dict[str, pd.DataFrame | None]] | xr.Dataset: ...

    def params(
        self, as_list: bool = True
    ) -> list[dict[str, pd.DataFrame | None]] | xr.Dataset:
        """
        Get the parameters in this set.

        Parameters
        ----------
        as_list : bool, default True
            If True, returns the parameters as a list of dictionaries with keys "shared" and "unit_specific".
            If False, returns the internal xarray Dataset.

        Returns
        -------
        list[dict[str, pd.DataFrame | None]] | xr.Dataset
            The parameters either as a list of dictionaries or as a Dataset.
        """
        return super().params(as_list)

    def _to_list(self) -> list[dict[str, pd.DataFrame | None]]:
        """Return the parameter sets as a list of dictionaries with 'shared' and 'unit_specific' DataFrames."""
        reps = self.num_replicates()
        if reps == 0:
            return []

        shared_names = self._canonical_shared_param_names
        unit_specific_names = self._canonical_unit_param_names
        unit_names = self.get_unit_names()

        out = []
        for j in range(reps):
            t_dict = {}

            if shared_names:
                s_vals = [
                    float(
                        self._data["shared"].isel(theta_idx=j).sel(parameter=p).values
                    )
                    for p in shared_names
                ]
                t_dict["shared"] = pd.DataFrame(
                    s_vals, index=pd.Index(shared_names), columns=["shared"]
                )
            else:
                t_dict["shared"] = None

            if unit_specific_names and unit_names:
                u_vals = np.zeros((len(unit_specific_names), len(unit_names)))
                for p_idx, p in enumerate(unit_specific_names):
                    for u_idx, u in enumerate(unit_names):
                        u_vals[p_idx, u_idx] = float(
                            self._data["unit_specific"]
                            .isel(theta_idx=j)
                            .sel(parameter=p, unit=u)
                            .values
                        )
                t_dict["unit_specific"] = pd.DataFrame(
                    u_vals,
                    index=pd.Index(unit_specific_names),
                    columns=pd.Index(unit_names),
                )
            else:
                t_dict["unit_specific"] = None

            out.append(t_dict)
        return out

    def _replicated_logLik(self, n: int) -> dict[str, np.ndarray]:
        if self._logLik_unit.size > 0:
            new_ll_unit = np.tile(self._logLik_unit, (n, 1))
        else:
            new_ll_unit = np.empty((n * self.num_replicates(), 0))
        return {"logLik_unit": new_ll_unit}

    def _slice_logLik(self, indices: np.ndarray) -> None:
        self._logLik_unit = self._logLik_unit[indices]
        self._logLik = self._logLik_unit.sum(axis=1)

    def _eq_logLik(self, other: "PanelParameters") -> bool:
        if self._canonical_shared_param_names != other._canonical_shared_param_names:
            return False
        if self._canonical_unit_param_names != other._canonical_unit_param_names:
            return False
        return np.array_equal(self._logLik_unit, other._logLik_unit, equal_nan=True)

    def _getitem_int(self, index: int) -> dict[str, pd.DataFrame | None]:
        return self._to_list()[index]

    def _transform_and_load(
        self,
        par_trans: ParTrans,
        param_list: list[Any],
        direction: Literal["to_est", "from_est"],
    ) -> None:
        transformed_list = par_trans.panel_transform_list(
            param_list, direction=direction
        )
        ds, s_names, u_names = _standardize_panel_theta(transformed_list)
        self._data = ds
        self._canonical_shared_param_names = [str(x) for x in s_names]
        self._canonical_unit_param_names = [str(x) for x in u_names]

    def _set_theta(self, value: Any) -> None:
        ds, s_names, u_names = _standardize_panel_theta(value)
        self._data = ds
        self._canonical_shared_param_names = [str(x) for x in s_names]
        self._canonical_unit_param_names = [str(x) for x in u_names]

        self._canonical_param_names = list(
            set(self._canonical_shared_param_names + self._canonical_unit_param_names)
        )
        self._logLik_unit = self._format_logLik_unit(None, self.num_replicates())
        self._logLik = self._logLik_unit.sum(axis=1)

    @staticmethod
    def merge(*param_objs: "PanelParameters") -> "PanelParameters":
        """Merge replications from multiple PanelParameters objects."""
        if len(param_objs) == 0:
            raise ValueError("At least one PanelParameters object must be provided.")
        first = param_objs[0]

        for obj in param_objs:
            if not isinstance(obj, PanelParameters):
                raise TypeError("All merged objects must be of type PanelParameters.")
            if obj._canonical_shared_param_names != first._canonical_shared_param_names:
                raise ValueError(
                    "All PanelParameters objects must have the same canonical shared parameter names."
                )
            if obj._canonical_unit_param_names != first._canonical_unit_param_names:
                raise ValueError(
                    "All PanelParameters objects must have the same canonical unit parameter names."
                )
            if obj.estimation_scale != first.estimation_scale:
                raise ValueError(
                    "All PanelParameters objects must have the same estimation scale."
                )
            if obj.get_unit_names() != first.get_unit_names():
                raise ValueError(
                    "All PanelParameters objects must have the same unit names."
                )

        merged_data = xr.concat([obj._data for obj in param_objs], dim="theta_idx")
        merged_data.coords["theta_idx"] = np.arange(merged_data.sizes["theta_idx"])
        merged_data.attrs["shared_names"] = first._canonical_shared_param_names
        merged_data.attrs["unit_specific_names"] = first._canonical_unit_param_names

        all_logLik_unit = [obj._logLik_unit for obj in param_objs]
        merged_logLik_unit = (
            np.concatenate(all_logLik_unit, axis=0) if all_logLik_unit else np.array([])
        )
        return PanelParameters(
            merged_data,
            logLik_unit=merged_logLik_unit,
            estimation_scale=first.estimation_scale,
        )

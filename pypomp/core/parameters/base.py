from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import numpy as np
import jax
import xarray as xr
from typing import (
    Union,
    Literal,
    Iterator,
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from ..par_trans import ParTrans

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
    def to_jax_array(self, param_names: list[str] | None = None, **kwargs) -> jax.Array:
        """
        Converts the parameters to a JAX array suitable for model functions.

        Args:
            param_names: A list of canonical parameter names expected by the model.
                If None, defaults to the canonical order of parameters in the set.
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

        new_data = xr.concat(cast(Any, [self._data] * n), dim="theta_idx")
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
            if self.__class__.__name__ == "PompParameters":
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

    @abstractmethod
    def set_params(self, value: Any) -> None:
        """
        Set or overwrite the parameter values.
        """
        pass

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

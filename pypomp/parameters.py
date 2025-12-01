"""
This module defines the parameter classes for Pomp and PanelPomp models.
It handles input validation, standardization, and conversion to JAX arrays.
"""

from abc import ABC, abstractmethod
import pandas as pd
import jax.numpy as jnp
import jax
from typing import Any, Union, Optional, Literal
from .ParTrans_class import ParTrans


class ParameterSet(ABC):
    """
    Abstract base class for parameter sets used in POMP models.
    """

    @abstractmethod
    def to_jax_array(self, param_names: list[str], **kwargs) -> jax.Array:
        """
        Converts the parameters to a JAX array suitable for model functions.

        Args:
            param_names: A list of canonical parameter names expected by the model.
            **kwargs: Additional context required for conversion (e.g. unit names).

        Returns:
            A JAX array representing the parameters.
            - For Pomp: Shape (reps, n_params)
            - For PanelPomp: Shape (reps, n_units, n_params)
        """
        pass

    @abstractmethod
    def num_replicates(self) -> int:
        """Returns the number of parameter replicates (J)."""
        pass

    @abstractmethod
    def subset(self, indices: Union[int, list[int], slice]) -> "ParameterSet":
        """
        Returns a new ParameterSet containing only the specified replicate indices.
        """
        pass

    @abstractmethod
    def get_param_names(self) -> list[str] | tuple[list[str], list[str]]:
        """Returns the list of parameter names contained in this set."""
        pass


class PompParameters(ParameterSet):
    """
    Manages parameters for a standard Pomp model.
    Internal storage is a list of dictionaries.
    """

    def __init__(
        self,
        theta: Union[dict, list[dict], "PompParameters"],
        estimation_scale: bool = False,
    ):
        """
        Args:
            theta: A single dictionary, a list of dictionaries, or an existing
                   PompParameters object.
        """
        if isinstance(theta, PompParameters):
            # Copy constructor behavior (shallow copy of list)
            self._params = list(theta._params)
            return

        # Normalize input to list of dicts
        if isinstance(theta, dict):
            theta = [theta]

        self._validate_raw(theta)
        self._params: list[dict] = theta
        self._canonical_param_names: list[str] = list(self._params[0].keys())
        self.estimation_scale: bool = estimation_scale

    def _validate_raw(self, theta: list[dict]):
        if not isinstance(theta, list):
            raise TypeError("theta must be a list of dictionaries")

        if len(theta) == 0:
            raise ValueError("theta cannot be empty")

        if not all(isinstance(t, dict) for t in theta):
            raise TypeError("All elements in theta must be dictionaries")

        # Check that all values of all dictionaries are single floats
        for i, t in enumerate(theta):
            for key, value in t.items():
                if not isinstance(value, float):
                    raise TypeError(
                        f"Parameter '{key}' at index {i} is not a float: got {type(value).__name__}"
                    )

        # Ensure all dicts have identical keys
        first_keys = set(theta[0].keys())
        for i, t in enumerate(theta[1:]):
            if set(t.keys()) != first_keys:
                raise ValueError(
                    f"Parameter set at index {i + 1} has different keys than the first set. "
                    f"Expected {first_keys}, got {set(t.keys())}"
                )

    def _child_PompParameters(
        self,
        theta: Union[dict, list[dict], "PompParameters"] | None = None,
        estimation_scale: bool | None = None,
    ):
        """
        Make a new PompParameters object with current attributes as the default.
        """
        theta_sel = theta or self._params
        estimation_scale_sel = estimation_scale or self.estimation_scale
        return PompParameters(theta=theta_sel, estimation_scale=estimation_scale_sel)

    def to_jax_array(self, param_names: list[str], **kwargs) -> jax.Array:
        """
        Convert to JAX array matching the order of param_names.
        Returns shape (n_reps, n_params).
        """
        # Logic formerly in _theta_dict_to_array
        try:
            ordered_values = [[t[name] for name in param_names] for t in self._params]
        except KeyError as e:
            raise KeyError(
                f"Parameter {e} expected by model but missing from parameter set."
            )

        return jnp.array(ordered_values)

    def to_jax_array_canonical(self) -> jax.Array:
        return self.to_jax_array(self._canonical_param_names)

    def num_replicates(self) -> int:
        return len(self._params)

    def num_params(self) -> int:
        return len(self._canonical_param_names)

    def subset(self, indices: Union[int, list[int], slice]) -> "PompParameters":
        if isinstance(indices, int):
            indices = [indices]

        # Determine subset based on type
        if isinstance(indices, slice):
            subset_params = self._params[indices]
        else:
            subset_params = [self._params[i] for i in indices]

        return self._child_PompParameters(subset_params)

    def get_param_names(self) -> list[str]:
        if not self._params:
            return []
        return list(self._canonical_param_names)

    def to_list(self) -> list[dict]:
        """Returns the internal list of dictionaries."""
        return self._params

    def transform(
        self,
        par_trans: ParTrans,
        direction: Literal["to_est", "from_est"] | None = None,
    ):
        """
        Transform the parameters to or from the estimation parameter space.

        Args:
            par_trans: The parameter transformation object.
            direction: The direction of transformation. If None, the direction is determined by the estimation_scale attribute.
        """
        auto = direction is None
        if auto:
            direction = "from_est" if self.estimation_scale else "to_est"
        if direction not in ["to_est", "from_est"]:
            raise ValueError(f"Invalid direction: {direction}")
        if direction == "to_est" and self.estimation_scale is False:
            self._params = [par_trans.to_est(theta_i) for theta_i in self._params]
            self.estimation_scale = True
        elif direction == "from_est" and self.estimation_scale is True:
            self._params = [par_trans.from_est(theta_i) for theta_i in self._params]
            self.estimation_scale = False
        else:
            # If this statement is reached, the parameters are already in the correct estimation space. Nothing needs to be done.
            pass

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[dict, "PompParameters", float]:
        """
        Support indexing like theta[0] or theta[0:2] or theta[0]["param_name"].
        - Integer index: returns the dict at that position
        - Slice: returns a new PompParameters object
        """
        if isinstance(index, int):
            # Integer index: return the dict directly
            return self._params[index]
        elif isinstance(index, slice):
            # Slice: return a new PompParameters object
            return self._child_PompParameters(self._params[index])
        else:
            raise TypeError(f"Invalid index: {index}. Must be an integer or slice.")

    def __iter__(self):
        """Support iteration over parameter sets."""
        return iter(self._params)

    def __len__(self) -> int:
        """Return the number of parameter replicates."""
        return len(self._params)

    def __mul__(self, n: int) -> "PompParameters":
        """
        Support replication like theta * 3.
        Returns a new PompParameters with n copies of the parameter sets.
        """
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("Multiplication factor must be non-negative")
        if n == 0:
            raise ValueError("Cannot create empty PompParameters")
        # Replicate the parameter sets n times
        replicated_params = self._params * n
        return self._child_PompParameters(replicated_params)

    def __rmul__(self, n: int) -> "PompParameters":
        """Support left multiplication like 3 * theta."""
        return self.__mul__(n)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PompParameters(n_replicates={len(self._params)}, n_params={len(self._canonical_param_names)})"

    def __eq__(self, other) -> bool:
        """
        Check equality with another PompParameters object.
        Two PompParameters are equal if their canonical parameter names and parameter sets are equal.
        """
        if not isinstance(other, type(self)):
            return False
        # Compare canonical parameter names
        if self._canonical_param_names != other._canonical_param_names:
            return False
        # Compare parameter lists
        if len(self._params) != len(other._params):
            return False
        for p1, p2 in zip(self._params, other._params):
            if p1 != p2:
                return False
        # Check same scale
        if self.estimation_scale != other.estimation_scale:
            return False
        return True


class PanelParameters(ParameterSet):
    """
    Manages parameters for PanelPomp models.
    Internal storage is lists of DataFrames for shared and unit-specific parameters.
    """

    def __init__(
        self,
        shared: Union[pd.DataFrame, list[pd.DataFrame], None] = None,
        unit_specific: Union[pd.DataFrame, list[pd.DataFrame], None] = None,
    ):
        self.shared = self._normalize_input(shared)
        self.unit_specific = self._normalize_input(unit_specific)

        self._validate_consistency()
        self._canonical_shared_param_names: list[str] = (
            list(self.shared[0].index) if self.shared else []
        )
        self._canonical_unit_specific_param_names: list[str] = (
            list(self.unit_specific[0].index) if self.unit_specific else []
        )

    def _normalize_input(self, data: Any) -> Optional[list[pd.DataFrame]]:
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return [data]
        if isinstance(data, list):
            if not all(isinstance(d, pd.DataFrame) for d in data):
                raise TypeError("Input lists must contain pandas DataFrames")
            return data
        raise TypeError(
            f"Invalid input type: {type(data)}. Expected DataFrame or list of DataFrames."
        )

    def _validate_consistency(self):
        """
        Ensures shared and unit_specific inputs have matching lengths (J) and structures.
        """
        if self.shared is None and self.unit_specific is None:
            # It is technically possible to initialize an empty shell,
            # but usually this implies no parameters.
            return

        # Check lengths match if both exist
        if self.shared is not None and self.unit_specific is not None:
            if len(self.shared) != len(self.unit_specific):
                raise ValueError(
                    f"shared and unit_specific must have the same number of replicates. "
                    f"Got shared={len(self.shared)}, unit_specific={len(self.unit_specific)}"
                )

        # Validate index consistency across replicates
        # (All shared DFs should have same index; All unit_specific DFs should have same index/columns)
        if self.shared:
            ref_idx = self.shared[0].index
            if not all(df.index.equals(ref_idx) for df in self.shared[1:]):
                raise ValueError(
                    "All shared DataFrames must have the same index (parameter names)"
                )
            # Check that all shared DataFrames have a single column named "shared"
            for i, df in enumerate(self.shared):
                if df.columns != ["shared"]:
                    raise ValueError(
                        f"All shared DataFrames must contain a single column named 'shared'. "
                        f"Missing in shared DataFrame at replicate {i}."
                    )

        if self.unit_specific:
            ref_idx = self.unit_specific[0].index
            ref_cols = self.unit_specific[0].columns
            for i, df in enumerate(self.unit_specific[1:]):
                if not df.index.equals(ref_idx):
                    raise ValueError(
                        "All unit_specific DataFrames must have the same index (parameter names)"
                    )
                if not df.columns.equals(ref_cols):
                    raise ValueError(
                        f"All unit_specific DataFrames must have the same columns (unit names). "
                        f"Mismatch found at replicate {i + 1}."
                    )

    def num_replicates(self) -> int:
        if self.shared is not None:
            return len(self.shared)
        if self.unit_specific is not None:
            return len(self.unit_specific)
        return 0

    def get_param_names(self) -> tuple[list[str], list[str]]:
        """
        Returns a tuple: (shared_param_names, unit_specific_param_names)
        """
        return (
            list(self._canonical_shared_param_names),
            list(self._canonical_unit_specific_param_names),
        )

    def subset(self, indices: Union[int, list[int], slice]) -> "PanelParameters":
        if isinstance(indices, int):
            indices = [indices]

        if isinstance(indices, slice):
            new_shared = self.shared[indices] if self.shared else None
            new_specific = self.unit_specific[indices] if self.unit_specific else None
        else:
            new_shared = [self.shared[i] for i in indices] if self.shared else None
            new_specific = (
                [self.unit_specific[i] for i in indices] if self.unit_specific else None
            )

        return PanelParameters(new_shared, new_specific)

    def to_jax_array(
        self, param_names: list[str], unit_names: list[str] | None = None
    ) -> jax.Array:
        """
        Constructs a combined parameter array for all units.

        Strategy:
        1. Iterate through replicates.
        2. For each replicate, build a (n_units, n_params) matrix.
        3. Stack to get (J, n_units, n_params).

        Args:
            param_names: List of all parameters required by the model logic.
            unit_names: List of unit names in the order expected by the model.
                        If None, inferred from unit_specific columns.
        """
        J = self.num_replicates()
        if J == 0:
            return jnp.empty((0, 0, 0))

        # Infer unit names if not provided
        if unit_names is None:
            if self.unit_specific:
                unit_names = list(self.unit_specific[0].columns)
            else:
                raise ValueError(
                    "unit_names must be provided if no unit_specific parameters exist"
                )

        n_units = len(unit_names)
        n_params = len(param_names)

        # Pre-calculate indices for speed
        # We need to know, for each required param, whether it comes from shared or specific
        shared_keys = set(self.shared[0].index) if self.shared else set()
        specific_keys = (
            set(self.unit_specific[0].index) if self.unit_specific else set()
        )

        # Build the array
        # This implementation uses numpy for construction flexibility, then converts to JAX
        # (constructing complex structures directly in JAX can be tricky without vmap)
        import numpy as np

        full_array = np.zeros((J, n_units, n_params))

        for j in range(J):
            # Extract data for this replicate
            s_df = self.shared[j] if self.shared else None
            u_df = self.unit_specific[j] if self.unit_specific else None

            for p_idx, p_name in enumerate(param_names):
                if p_name in specific_keys:
                    # Look up in unit_specific (row=p_name)
                    # We ensure we pull values in the order of `unit_names`
                    # Reindexing handles the ordering safety
                    if u_df is None:
                        raise ValueError(
                            "unit_specific is None but parameter is unit-specific"
                        )
                    try:
                        vals = u_df.loc[p_name, unit_names].values
                    except KeyError as e:
                        # This happens if a unit name in unit_names is missing from the DF columns
                        raise KeyError(
                            f"Unit {e} not found in unit_specific parameters"
                        )
                    full_array[j, :, p_idx] = vals
                elif p_name in shared_keys:
                    # Look up in shared and broadcast across units
                    if s_df is None:
                        raise ValueError("shared is None but parameter is shared")
                    val = s_df.loc[p_name].values[
                        0
                    ]  # Assumes single column 'shared' or similar
                    full_array[j, :, p_idx] = val
                else:
                    raise KeyError(
                        f"Parameter '{p_name}' not found in shared or unit_specific data."
                    )

        return jnp.array(full_array)

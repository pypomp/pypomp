import pandas as pd
from typing import TYPE_CHECKING
from pypomp.pomp_class import Pomp

if TYPE_CHECKING:
    from .interfaces import PanelPompInterface as Base
else:
    Base = object  # At runtime, this is just a normal class


class PanelValidationMixin(Base):
    """
    Handles internal validation of DataFrames, Pomp objects, and parameter names.
    """

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

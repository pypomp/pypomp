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

    def _validate_unit_objects(self) -> None:
        if not isinstance(self.unit_objects, dict):
            raise TypeError("unit_objects must be a dictionary")
        unit_objs = list(self.unit_objects.values())
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

    def _validate_params_and_units(self) -> None:
        """
        Validates:
        - Everything from _validate_unit_objects()
        - The unit names in the unit_objects dictionary must match the unit names in the theta object.
        - The canonical parameter names must match the canonical parameter names in the theta object.
        """
        self._validate_unit_objects()
        if self.get_unit_names() != list(self.theta.get_unit_names()):
            raise ValueError(
                "The unit names in the unit_objects dictionary must match the unit names in the theta object"
            )
        if set(self.canonical_param_names) != set(self.theta.get_param_names()):
            raise ValueError(
                "The canonical parameter names must match the canonical parameter names in the theta object"
            )
        first_unit_canonical_param_names = self.unit_objects[
            self.get_unit_names()[0]
        ].canonical_param_names
        unit_canonical_param_names_match = [
            set(self.unit_objects[unit].canonical_param_names)
            == set(first_unit_canonical_param_names)
            for unit in self.get_unit_names()
        ]
        if not all(unit_canonical_param_names_match):
            raise ValueError(
                "The canonical parameter names in the unit objects must match the canonical parameter names in the first unit for all units."
            )
        if set(self.canonical_param_names) != set(first_unit_canonical_param_names):
            raise ValueError(
                "The canonical parameter names must match the canonical parameter names in the unit objects (up to reordering)."
            )
        self.canonical_param_names = self.canonical_shared_param_names + self.canonical_unit_param_names

from typing import TYPE_CHECKING

from pypomp.core.pomp import Pomp

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
            if unit_obj.statenames != unit_objs[0].statenames:
                raise ValueError("All units must have the same statenames")
            if unit_obj.accumvars != unit_objs[0].accumvars:
                raise ValueError("All units must have the same accumvars")
            if unit_obj.covar_names != unit_objs[0].covar_names:
                raise ValueError("All units must have the same covar_names")
            if unit_obj.par_trans != unit_objs[0].par_trans:
                raise ValueError("All units must have the same par_trans")
            if getattr(unit_obj.rinit, "original_func", unit_obj.rinit) != getattr(
                unit_objs[0].rinit, "original_func", unit_objs[0].rinit
            ):
                raise ValueError("All units must have the same rinit")
            if getattr(unit_obj.rproc, "original_func", unit_obj.rproc) != getattr(
                unit_objs[0].rproc, "original_func", unit_objs[0].rproc
            ):
                raise ValueError("All units must have the same rproc")
            if getattr(unit_obj.dmeas, "original_func", unit_obj.dmeas) != getattr(
                unit_objs[0].dmeas, "original_func", unit_objs[0].dmeas
            ):
                raise ValueError("All units must have the same dmeas")
            if getattr(unit_obj.rmeas, "original_func", unit_obj.rmeas) != getattr(
                unit_objs[0].rmeas, "original_func", unit_objs[0].rmeas
            ):
                raise ValueError("All units must have the same rmeas")
            if getattr(unit_obj.dprior, "original_func", unit_obj.dprior) != getattr(
                unit_objs[0].dprior, "original_func", unit_objs[0].dprior
            ):
                raise ValueError("All units must have the same dprior")

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
        self.canonical_param_names = (
            self.canonical_shared_param_names + self.canonical_unit_param_names
        )

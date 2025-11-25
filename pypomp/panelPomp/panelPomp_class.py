"""
This module implements the OOP structure for PanelPOMP models.
"""

import jax
import pandas as pd
from pypomp.pomp_class import Pomp
from pypomp.panelPomp.validation_mixin import PanelValidationMixin
from pypomp.panelPomp.estimation_mixin import PanelEstimationMixin
from pypomp.panelPomp.analysis_mixin import PanelAnalysisMixin


class PanelPomp(PanelValidationMixin, PanelEstimationMixin, PanelAnalysisMixin):
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

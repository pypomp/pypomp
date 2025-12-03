"""
This module implements the OOP structure for PanelPOMP models.
"""

import jax
import pandas as pd
from pypomp.pomp_class import Pomp
from pypomp.panelPomp.validation_mixin import PanelValidationMixin
from pypomp.panelPomp.estimation_mixin import PanelEstimationMixin
from pypomp.panelPomp.analysis_mixin import PanelAnalysisMixin
from pypomp.results import ResultsHistory
from pypomp.parameters import PanelParameters


class PanelPomp(PanelValidationMixin, PanelEstimationMixin, PanelAnalysisMixin):
    def __init__(
        self,
        Pomp_dict: dict[str, Pomp],
        theta: PanelParameters
        | dict[str, pd.DataFrame | None]
        | list[dict[str, pd.DataFrame | None]]
        | None = None,
    ):
        """
        Initializes a PanelPOMP model, which consists of multiple POMP models
        (units) that share the same structure but may have different parameters
        and observations.

        Args:
            Pomp_dict (dict[str, Pomp]): A dictionary mapping unit names to Pomp objects. Each Pomp object represents a single unit in the panel data.
            The keys are used as unit identifiers.
            theta: A PanelParameters object, a dictionary with "shared" and "unit_specific" keys, or a list of such dictionaries.
        """
        unit_objects = self._validate_unit_objects(Pomp_dict)

        # Convert inputs to PanelParameters
        if theta is not None:
            if isinstance(theta, PanelParameters):
                self.theta = theta
            else:
                self.theta = PanelParameters(theta=theta)
        else:
            self.theta = PanelParameters(theta=None)

        self.unit_objects: dict[str, Pomp] = unit_objects
        self.results_history = ResultsHistory()
        self.fresh_key: jax.Array | None = None
        self.canonical_param_names: list[str] = self.theta.get_param_names()
        self.canonical_shared_param_names: list[str] = (
            self.theta.get_shared_param_names()
        )
        self.canonical_unit_param_names: list[str] = self.theta.get_unit_param_names()

        for unit in self.unit_objects.keys():
            self.unit_objects[unit].theta = None  # type: ignore

    def print_summary(self):
        """
        Print a summary of the PanelPomp object.
        """
        first_unit = list(self.unit_objects.keys())[0]
        print("Basics:")
        print("-------")
        print(f"Number of units: {len(self.unit_objects)}")
        print(f"Number of parameters: {len(self.canonical_param_names)}")
        print(
            f"Number of observations (first unit): {len(self.unit_objects[first_unit].ys)}"
        )
        print(
            f"Number of time steps (first unit): {len(self.unit_objects[first_unit]._dt_array_extended)}"
        )
        print()
        self.results_history.print_summary()

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

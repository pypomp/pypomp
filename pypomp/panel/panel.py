"""
This module implements the OOP structure for PanelPOMP models.
"""

import jax
import jax.numpy as jnp
from pypomp.functional.structs import PanelPompStruct
from pypomp.core.pomp import Pomp
from .validation_mixin import PanelValidationMixin
from .estimation_mixin import PanelEstimationMixin
from .analysis_mixin import PanelAnalysisMixin
from pypomp.core.results import ResultsHistory
from pypomp.core.parameters import PanelParameters
from pypomp.core.metadata import ModelMetadata
from copy import deepcopy


class PanelPomp(PanelValidationMixin, PanelEstimationMixin, PanelAnalysisMixin):
    """Panel of partially observed Markov process models.

    Extends the single-unit POMP framework to handle multiple units that share
    structural characteristics but may have distinct parameter values and
    observations.

    Parameters
    ----------
    Pomp_dict : dict of str to Pomp
        Mapping from unit names to :class:`~pypomp.Pomp` objects.
    theta : PanelParameters
        A :class:`~pypomp.core.parameters.PanelParameters` object containing
        the model parameters.
    """

    unit_objects: dict[str, Pomp]
    """A dictionary mapping unit names to their corresponding :class:`~pypomp.core.pomp.Pomp` objects."""
    results_history: ResultsHistory
    """A :class:`~pypomp.core.results.ResultsHistory` object storing the history of results from method calls."""
    fresh_key: jax.Array | None
    """Running a method that accepts a JAX PRNG key will store a fresh, unused key here."""
    metadata: ModelMetadata
    """Environment and version metadata initialized when this instance was built."""
    canonical_param_names: list[str]
    """All unique parameter names present in either the shared or unit-specific parameters."""
    canonical_shared_param_names: list[str]
    """Parameter names of parameters with values shared across all units in the panel."""
    canonical_unit_param_names: list[str]
    """Parameter names of parameters with values specific to individual units in the panel."""

    _theta: PanelParameters
    """The internal parameter object storage."""

    def __init__(
        self,
        Pomp_dict: dict[str, Pomp],
        theta: PanelParameters,
    ):
        if not isinstance(theta, PanelParameters):
            raise TypeError("theta must be a PanelParameters instance")
        self._theta = theta

        self.unit_objects = Pomp_dict
        self.results_history = ResultsHistory()
        self.fresh_key = None
        self.metadata = ModelMetadata()
        self.canonical_param_names = self.theta.get_param_names()
        self.canonical_shared_param_names = self.theta.get_shared_param_names()
        self.canonical_unit_param_names = self.theta.get_unit_param_names()

        self._validate_params_and_units()

        for unit in self.unit_objects.keys():
            self.unit_objects[unit].theta = None  # type: ignore

    @property
    def theta(self) -> PanelParameters:
        """The parameter object for the panel model."""
        return self._theta

    @theta.setter
    def theta(self, value: PanelParameters):
        if not isinstance(value, PanelParameters):
            raise TypeError("theta must be a PanelParameters instance")
        self._theta = value

    def get_unit_names(self) -> list[str]:
        """
        Returns a list of the names of the units in the panel.

        Returns
        -------
        list[str]
            The names of the units in the panel.
        """
        return list(self.unit_objects.keys())

    def to_struct(self) -> PanelPompStruct:
        """Export static data and compiled simulators into a JAX PyTree.

        Converts the panel model into a :class:`PanelPompStruct` suitable for
        use with pure-functional algorithms in :mod:`pypomp.functional`.

        Returns
        -------
        PanelPompStruct
            The compiled structural representation of the panel model.
        """

        unit_names = self.get_unit_names()
        rep_unit = self.unit_objects[unit_names[0]]

        ys_per_unit = jnp.stack(
            [jnp.array(self.unit_objects[u].ys) for u in unit_names], axis=0
        )
        covars_per_unit = self._get_covars_per_unit(unit_names)

        unit_param_permutations = jnp.stack(
            [self._get_unit_param_permutation(u) for u in unit_names], axis=0
        )

        return PanelPompStruct(
            ys_per_unit=ys_per_unit,
            dt_array_extended=jnp.array(rep_unit._dt_array_extended),
            nstep_array=jnp.array(rep_unit._nstep_array),
            t0=rep_unit.t0,
            times=jnp.array(rep_unit.ys.index),
            covars_per_unit=covars_per_unit,
            accumvars=rep_unit.rproc.accumvars,
            rinit_pf=rep_unit.rinit.struct_pf,
            rproc_pf=rep_unit.rproc.struct_pf_interp,
            dmeas_pf=rep_unit.dmeas.struct_pf if rep_unit.dmeas is not None else None,
            rinit_per=rep_unit.rinit.struct_per,
            rproc_per=rep_unit.rproc.struct_per_interp,
            dmeas_per=rep_unit.dmeas.struct_per if rep_unit.dmeas is not None else None,
            rmeas_pf=rep_unit.rmeas.struct_pf if rep_unit.rmeas is not None else None,
            par_trans=rep_unit.par_trans,
            param_names=self.canonical_param_names,
            shared_param_names=self.canonical_shared_param_names,
            unit_param_names=self.canonical_unit_param_names,
            unit_param_permutations=unit_param_permutations,
            unit_names=unit_names,
        )

    def print_metadata(self) -> None:
        """
        Prints the creation and runtime environment metadata for this instance.
        """
        self.metadata.print_metadata()

    def print_summary(self, n: int = 5):
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
        print(f"Number of parameter sets: {self.theta.num_replicates()}")
        print()
        self.results_history.print_summary(n=n)

    def __eq__(self, other):
        """
        Check structural equality with another PanelPomp object.

        Two PanelPomp instances are considered equal if they:
        - Are of the same type
        - Have identical canonical parameter name lists
        - Have equal PanelParameters (self.theta)
        - Have the same unit names in the same order
        - Have unit Pomp objects with identical data and parameter structure
        - Have equal results_history
        - Have equal fresh_key values (or both None)
        """
        if not isinstance(other, type(self)):
            return False

        # Canonical parameter structure
        if self.canonical_param_names != other.canonical_param_names:
            return False
        if self.canonical_shared_param_names != other.canonical_shared_param_names:
            return False
        if self.canonical_unit_param_names != other.canonical_unit_param_names:
            return False

        # Panel parameters
        if self.theta != other.theta:
            return False

        # Unit objects: same unit names and comparable structure
        self_units = list(self.unit_objects.keys())
        other_units = list(other.unit_objects.keys())
        if self_units != other_units:
            return False

        for unit in self_units:
            if self.unit_objects[unit] != other.unit_objects[unit]:
                return False

        if self.results_history != other.results_history:
            return False

        if (self.fresh_key is None) != (other.fresh_key is None):
            return False
        if self.fresh_key is not None and other.fresh_key is not None:
            if not jax.numpy.array_equal(
                jax.random.key_data(self.fresh_key),
                jax.random.key_data(other.fresh_key),
            ):
                return False

        return True

    @staticmethod
    def merge(*panel_pomp_objs: "PanelPomp") -> "PanelPomp":
        """
        Merge replications from multiple PanelPomp objects into a single object.
        All panel objects must have the same units and canonical parameter names.
        """
        if len(panel_pomp_objs) == 0:
            raise ValueError("At least one PanelPomp object must be provided.")
        first = panel_pomp_objs[0]

        for obj in panel_pomp_objs:
            if not isinstance(obj, type(first)):
                raise TypeError("All merged objects must be of type PanelPomp.")
            if obj.canonical_param_names != first.canonical_param_names:
                raise ValueError(
                    "All PanelPomp objects must have the same canonical_param_names."
                )
            if obj.canonical_shared_param_names != first.canonical_shared_param_names:
                raise ValueError(
                    "All PanelPomp objects must have the same canonical_shared_param_names."
                )
            if obj.canonical_unit_param_names != first.canonical_unit_param_names:
                raise ValueError(
                    "All PanelPomp objects must have the same canonical_unit_param_names."
                )
            if list(obj.unit_objects.keys()) != list(first.unit_objects.keys()):
                raise ValueError("All PanelPomp objects must have the same unit names.")

        merged_theta = PanelParameters.merge(*[obj.theta for obj in panel_pomp_objs])
        merged_history = ResultsHistory.merge(
            *[obj.results_history for obj in panel_pomp_objs]
        )

        merged_panel_pomp = deepcopy(first)
        merged_panel_pomp.theta = merged_theta
        merged_panel_pomp.results_history = merged_history
        merged_panel_pomp.fresh_key = first.fresh_key

        return merged_panel_pomp

    def __getstate__(self):
        """
        Custom pickling method to handle wrapped function objects. This is
        necessary because the JAX-wrapped functions in the Pomp objects are not picklable.
        """
        state = self.__dict__.copy()

        if self.fresh_key is not None:
            state["_fresh_key_data"] = jax.random.key_data(self.fresh_key)
        state.pop("fresh_key", None)
        if hasattr(self, "unit_objects") and self.unit_objects is not None:
            unit_objects_state = {}
            for unit_name, pomp_obj in self.unit_objects.items():
                unit_objects_state[unit_name] = pomp_obj.__getstate__()
            state["_unit_objects_state"] = unit_objects_state
            state.pop("unit_objects", None)

        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to reconstruct wrapped function objects. This is
        necessary because the JAX-wrapped functions in the Pomp objects are not picklable.
        """
        self.__dict__.update(state)

        if "_fresh_key_data" in state:
            self.fresh_key = jax.random.wrap_key_data(state["_fresh_key_data"])
        elif "fresh_key" not in self.__dict__:
            self.fresh_key = None
        self.__dict__.pop("_fresh_key_data", None)

        if "_unit_objects_state" in state:
            unit_objects = {}
            for unit_name, pomp_state in state["_unit_objects_state"].items():
                pomp_obj = Pomp.__new__(Pomp)
                pomp_obj.__setstate__(pomp_state)
                unit_objects[unit_name] = pomp_obj
            self.unit_objects = unit_objects
            del self.__dict__["_unit_objects_state"]
        else:
            self.unit_objects = {}

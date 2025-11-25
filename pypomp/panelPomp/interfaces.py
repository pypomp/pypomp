from typing import Protocol
import jax.numpy as jnp
import pandas as pd
import jax
from ..pomp_class import Pomp


# This Protocol defines what attributes the Mixins can expect to exist
class PanelPompInterface(Protocol):
    unit_objects: dict[str, Pomp]
    shared: list[pd.DataFrame] | None
    unit_specific: list[pd.DataFrame] | None
    results_history: "ResultsHistory"
    fresh_key: jax.Array | None
    canonical_param_names: list[str]
    canonical_shared_param_names: list[str]
    canonical_unit_param_names: list[str]

    # You can also add method signatures if Mixins call each other
    def _get_param_names(
        self, shared=None, unit_specific=None
    ) -> tuple[list[str], list[str]]: ...
    def _validate_params_and_units(
        self,
        shared: pd.DataFrame | list[pd.DataFrame] | None,
        unit_specific: pd.DataFrame | list[pd.DataFrame] | None,
        unit_objects: dict[str, Pomp],
    ) -> tuple[
        list[pd.DataFrame] | None, list[pd.DataFrame] | None, dict[str, Pomp]
    ]: ...
    def _get_theta_list_len(
        self,
        shared: list[pd.DataFrame] | None,
        unit_specific: list[pd.DataFrame] | None,
    ) -> int: ...
    def _dataframe_to_array_canonical(
        self, df: pd.DataFrame, param_names: list[str], column_name: str
    ) -> jnp.ndarray: ...

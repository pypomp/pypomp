from typing import Protocol
import jax.numpy as jnp
import pandas as pd
import jax
from ..pomp_class import Pomp
from ..parameters import PanelParameters
from ..results import ResultsHistory


# This Protocol defines what attributes the Mixins can expect to exist
class PanelPompInterface(Protocol):
    unit_objects: dict[str, Pomp]
    theta: PanelParameters
    results_history: ResultsHistory
    fresh_key: jax.Array | None
    canonical_param_names: list[str]
    canonical_shared_param_names: list[str]
    canonical_unit_param_names: list[str]

    # You can also add method signatures if Mixins call each other
    def _validate_params_and_units(
        self,
    ) -> None: ...
    def _dataframe_to_array_canonical(
        self, df: pd.DataFrame, param_names: list[str], column_name: str
    ) -> jnp.ndarray: ...

    def get_unit_names(self) -> list[str]: ...

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

import jax
import pandas as pd

from ..core.parameters import PanelParameters
from ..core.pomp import Pomp
from ..core.results import ResultsHistory
from ..functional.structs import PanelPompStruct

if TYPE_CHECKING:
    from .panel import PanelPomp


# This Protocol defines what attributes the Mixins can expect to exist
class PanelPompInterface(Protocol):
    unit_objects: dict[str, Pomp]

    @property
    def theta(self) -> PanelParameters: ...
    @theta.setter
    def theta(self, value: PanelParameters) -> None: ...

    results_history: ResultsHistory
    fresh_key: jax.Array | None
    canonical_param_names: list[str]
    canonical_shared_param_names: list[str]
    canonical_unit_param_names: list[str]

    def _validate_params_and_units(
        self,
    ) -> None: ...
    def _dataframe_to_array_canonical(
        self, df: pd.DataFrame, param_names: list[str], column_name: str
    ) -> jax.Array: ...

    def get_unit_names(self) -> list[str]: ...
    def to_struct(self) -> PanelPompStruct: ...

    @overload
    def simulate(
        self,
        nsim: int = 1,
        theta: PanelParameters | None = None,
        times: jax.Array | None = None,
        key: jax.Array | None = None,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        nsim: int = 1,
        theta: PanelParameters | None = None,
        times: jax.Array | None = None,
        key: jax.Array | None = None,
        *,
        as_pomp: Literal[True],
    ) -> PanelPomp: ...

    def simulate(
        self,
        nsim: int = 1,
        theta: PanelParameters | None = None,
        times: jax.Array | None = None,
        key: jax.Array | None = None,
        as_pomp: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | PanelPomp: ...

    @staticmethod
    def merge(*panel_pomp_objs: Any) -> Any: ...

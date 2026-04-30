from __future__ import annotations
from typing import Protocol, Any, overload, Union, Literal
import jax.numpy as jnp
import pandas as pd
import jax
from ..core.pomp import Pomp
from ..core.parameters import PanelParameters
from ..core.results import ResultsHistory


# This Protocol defines what attributes the Mixins can expect to exist
class PanelPompInterface(Protocol):
    unit_objects: dict[str, Pomp]
    theta: PanelParameters
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
    ) -> jnp.ndarray: ...

    def get_unit_names(self) -> list[str]: ...

    @overload
    def simulate(
        self,
        key: jax.Array,
        theta: PanelParameters
        | dict[str, pd.DataFrame | None]
        | list[dict[str, pd.DataFrame | None]]
        | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        key: jax.Array,
        theta: PanelParameters
        | dict[str, pd.DataFrame | None]
        | list[dict[str, pd.DataFrame | None]]
        | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        *,
        as_pomp: Literal[True],
    ) -> Any: ...

    def simulate(
        self,
        key: jax.Array,
        theta: PanelParameters
        | dict[str, pd.DataFrame | None]
        | list[dict[str, pd.DataFrame | None]]
        | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: bool = False,
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], Any]: ...

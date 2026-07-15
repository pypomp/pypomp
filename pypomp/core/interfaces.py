from __future__ import annotations
from typing import Protocol, Any, overload, Union, Literal
import jax
import numpy as np
import pandas as pd
from pypomp.functional.structs import PompStruct
from .parameters import PompParameters
from .results import ResultsHistory
from .model_struct import _RInit, _RProc, _DMeas, _RMeas
from .par_trans import ParTrans
from .metadata import ModelMetadata


class PompInterface(Protocol):
    """
    Protocol defining the attributes and methods of the Pomp class.
    Used by mixins for static type checking.
    """

    ys: pd.DataFrame
    _theta: PompParameters | None
    canonical_param_names: list[str]
    statenames: list[str]
    t0: float
    rinit: _RInit
    rproc: _RProc
    dmeas: _DMeas | None
    rmeas: _RMeas | None
    par_trans: ParTrans
    covars: pd.DataFrame | None
    _covars_extended: np.ndarray | None
    _nstep_array: np.ndarray
    _dt_array_extended: np.ndarray
    _max_steps_per_interval: int
    accumvars: list[str] | None
    _accumvars_indices: tuple[int, ...] | None
    results_history: ResultsHistory
    fresh_key: jax.Array | None
    metadata: ModelMetadata

    @property
    def theta(self) -> PompParameters: ...

    @theta.setter
    def theta(self, value: PompParameters | None) -> None: ...

    def _prepare_theta_input(
        self,
        theta: PompParameters | None,
    ) -> PompParameters: ...

    def _update_fresh_key(
        self, key: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array]: ...

    def to_struct(self) -> PompStruct: ...

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: Literal[False] = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    @overload
    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        *,
        as_pomp: Literal[True],
    ) -> Any: ...

    def simulate(
        self,
        key: jax.Array | None = None,
        theta: PompParameters | None = None,
        times: jax.Array | None = None,
        nsim: int = 1,
        as_pomp: bool = False,
    ) -> Union[tuple[pd.DataFrame, pd.DataFrame], Any]: ...

    def traces(self) -> pd.DataFrame: ...

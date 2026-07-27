"""Loop-internal pytree containers for the inference algorithms.

These types are the carries and stacked outputs of ``jax.lax.scan`` /
``jax.lax.fori_loop``, plus the per-step ``xs`` sequences fed into those loops.
Unlike the ``*Config`` / ``*Inputs`` containers in :mod:`.params`, they hold no
factory or conversion logic -- they are plain data shuttled through a single
algorithm's loop -- so they live next to nothing but each other and are
imported by the owning algorithm module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax.tree_util import register_dataclass

# ----MOP-----------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True)
class MopState:
    t: jax.Array | float
    particlesF: jax.Array
    loglik: float | jax.Array
    weightsF: jax.Array
    counts: jax.Array
    key: jax.Array
    t_idx: int


# ----TRAIN---------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True)
class TrainState:
    theta_ests: jax.Array
    key: jax.Array
    opt_state: Any


@register_dataclass
@dataclass(frozen=True)
class TrainMetrics:
    neg_loglik: jax.Array
    theta_ests: jax.Array


@register_dataclass
@dataclass(frozen=True)
class PanelTrainState:
    shared_ests: jax.Array
    unit_ests_chunked: jax.Array
    opt_state_shared: Any
    opt_state_unit_chunked: Any
    global_step: int


@register_dataclass
@dataclass(frozen=True)
class ChunkState:
    shared_ests: jax.Array
    opt_state_shared: Any
    global_step: int


@register_dataclass
@dataclass(frozen=True)
class ChunkMetrics:
    neg_loglik: jax.Array
    unit_ests_chunk: jax.Array
    opt_state_unit_chunk: Any


@register_dataclass
@dataclass(frozen=True)
class IterationMetrics:
    neg_loglik: jax.Array
    shared_ests: jax.Array
    unit_ests: jax.Array


# ----PFILTER-------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True)
class PfilterState:
    t: jax.Array | float
    particlesF: jax.Array
    loglik: float | jax.Array
    norm_weights: jax.Array
    counts: jax.Array
    key: jax.Array
    t_idx: int
    CLL_arr: jax.Array
    ESS_arr: jax.Array
    filter_mean_arr: jax.Array
    prediction_mean_arr: jax.Array


# ----MIF-----------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True)
class PerfilterState:
    t: float | jax.Array
    particlesF: jax.Array
    thetas: jax.Array
    loglik: jax.Array
    norm_weights: jax.Array
    counts: jax.Array
    t_idx: int
    ancestry: jax.Array


@register_dataclass
@dataclass(frozen=True)
class PerfilterStepInputs:
    y: jax.Array
    time: jax.Array
    nstep: jax.Array
    cooling_factor: jax.Array
    step_key: jax.Array


# ----PANEL MIF-----------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True)
class PanelMifState:
    shared: jax.Array  # Shape: (J, n_shared)
    unit_specific: jax.Array  # Shape: (J, U, n_spec)


@register_dataclass
@dataclass(frozen=True)
class UnitStepInputs:
    permutation: jax.Array
    ys: jax.Array
    covariates_dummy: jax.Array
    unit_idx: int | jax.Array
    key: jax.Array
    inverse_permutation: jax.Array


@register_dataclass
@dataclass(frozen=True)
class PanelMifIterInputs:
    m: int | jax.Array
    key: jax.Array

"""Algorithm context containers.

Inference algorithms accept a ``*Context`` pytree mixing inline-declared static
fields (callables/scalars marked with :func:`static`, stored in the treedef)
and dynamic traced fields (JAX arrays).

Per-algorithm contexts compose shared structures (:class:`SimFns`, :class:`ModelFns`,
:class:`SeriesData`). For ``jax.vmap`` ``in_axes`` prototypes, static fields must
match live instances; build them using ``.axes()`` or ``dataclasses.replace``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import jax
from jax.tree_util import register_dataclass

from ...functional.structs import PanelPompStruct, PompStruct
from ..rw_sigma import RWSigma


def static(**kwargs: Any) -> Any:
    """Mark a dataclass field as static (part of the pytree treedef).

    Static fields must be hashable and immutable; changing one causes JAX to
    retrace/recompile.  Use for model callables and Python scalars.
    """
    return field(metadata={"static": True}, **kwargs)


# ----shared contexts-----------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SimFns:
    """Callables needed to push the latent process forward, for one flavor.

    Entirely static, so this is a plain hashable dataclass rather than a pytree;
    it is always held in a :func:`static` field of a context object.

    This is the context for algorithms that only *simulate* the model and never
    evaluate a measurement density -- notably ABC, which is likelihood-free and
    uses ``rmeasure`` instead.  Filtering algorithms want :class:`ModelFns`.
    """

    rinitializer: Callable
    rprocess_interp: Callable
    accumvars: tuple[int, ...] | None

    @classmethod
    def pf(cls, struct: PompStruct | PanelPompStruct) -> SimFns:
        return cls(
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            accumvars=struct.accumvars,
        )


@dataclass(frozen=True, kw_only=True)
class ModelFns(SimFns):
    """:class:`SimFns` plus the measurement density the filters require.

    Build with :meth:`pf` for the particle-filter flavor or :meth:`per` for the
    IF2 perturbed flavor; both reject a model that lacks the corresponding
    ``dmeasure``.
    """

    dmeasure: Callable

    @classmethod
    def pf(cls, struct: PompStruct | PanelPompStruct) -> ModelFns:
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure (dmeas_pf) is required")
        return cls(
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            accumvars=struct.accumvars,
            dmeasure=struct.dmeas_pf,
        )

    @classmethod
    def per(cls, struct: PompStruct | PanelPompStruct) -> ModelFns:
        if struct.dmeas_per is None:
            raise ValueError("dmeasure (dmeas_per) is required")
        return cls(
            rinitializer=struct.rinit_per,
            rprocess_interp=struct.rproc_per,
            accumvars=struct.accumvars,
            dmeasure=struct.dmeas_per,
        )


@register_dataclass
@dataclass(frozen=True)
class SeriesData:
    """Observation series and integration grid.  Entirely dynamic (traced)."""

    ys: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    covars_extended: jax.Array | None

    @classmethod
    def from_struct(cls, struct: PompStruct) -> SeriesData:
        return cls(
            ys=struct.ys,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            covars_extended=struct.covars_extended,
        )

    @classmethod
    def from_panel_struct(cls, struct: PanelPompStruct) -> SeriesData:
        return cls(
            ys=struct.ys_per_unit,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            covars_extended=struct.covars_per_unit,
        )

    @classmethod
    def axes(
        cls,
        *,
        ys: Any = None,
        dt_array_extended: Any = None,
        nstep_array: Any = None,
        t0: Any = None,
        times: Any = None,
        covars_extended: Any = None,
    ) -> SeriesData:
        """Build a ``jax.vmap`` ``in_axes`` prototype for this context.

        Unspecified fields default to ``None`` (broadcast).
        """
        return cls(
            ys=cast(jax.Array, ys),
            dt_array_extended=cast(jax.Array, dt_array_extended),
            nstep_array=cast(jax.Array, nstep_array),
            t0=cast(float, t0),
            times=cast(jax.Array, times),
            covars_extended=cast(jax.Array, covars_extended),
        )


# ----PFILTER-------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class PfilterContext:
    series: SeriesData
    fns: ModelFns = static()
    J: int = static()
    thresh: float = static(default=0.0)
    CLL: bool = static(default=False)
    ESS: bool = static(default=False)
    filter_mean: bool = static(default=False)
    prediction_mean: bool = static(default=False)
    should_trans: bool = static(default=False)

    @classmethod
    def from_struct(
        cls,
        struct: PompStruct,
        J: int,
        thresh: float = 0.0,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        should_trans: bool = False,
    ) -> PfilterContext:
        return cls(
            series=SeriesData.from_struct(struct),
            fns=ModelFns.pf(struct),
            J=J,
            thresh=thresh,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
            should_trans=should_trans,
        )

    @classmethod
    def from_panel_struct(
        cls,
        struct: PanelPompStruct,
        J: int,
        thresh: float = 0.0,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        should_trans: bool = False,
    ) -> PfilterContext:
        return cls(
            series=SeriesData.from_panel_struct(struct),
            fns=ModelFns.pf(struct),
            J=J,
            thresh=thresh,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
            should_trans=should_trans,
        )


# ----MOP-----------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class MopContext:
    series: SeriesData
    alpha: float | jax.Array
    fns: ModelFns = static()
    J: int = static()

    @classmethod
    def from_struct(
        cls, struct: PompStruct, J: int, alpha: float | jax.Array
    ) -> MopContext:
        return cls(
            series=SeriesData.from_struct(struct),
            alpha=alpha,
            fns=ModelFns.pf(struct),
            J=J,
        )


# ----TRAIN---------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class TrainContext(MopContext):
    """Extends :class:`MopContext`; it is handed to the MOP gradient objective."""

    eta: jax.Array
    M: int = static()
    alpha_cooling: float = static()
    thresh: float = static()
    n_monitors: int = static()

    @classmethod
    def from_train_struct(
        cls,
        struct: PompStruct,
        J: int,
        M: int,
        alpha_cooling: float,
        thresh: float,
        n_monitors: int,
        eta: jax.Array,
        alpha: float | jax.Array,
    ) -> TrainContext:
        return cls(
            series=SeriesData.from_struct(struct),
            alpha=alpha,
            fns=ModelFns.pf(struct),
            J=J,
            eta=eta,
            M=M,
            alpha_cooling=alpha_cooling,
            thresh=thresh,
            n_monitors=n_monitors,
        )

    def to_mop_context(self) -> MopContext:
        return MopContext(series=self.series, alpha=self.alpha, fns=self.fns, J=self.J)

    def to_pfilter_context(self, should_trans: bool = False) -> PfilterContext:
        return PfilterContext(
            series=self.series,
            fns=self.fns,
            J=self.J,
            thresh=self.thresh,
            should_trans=should_trans,
        )


# ----DPOP TRAIN----------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class DpopTrainContext(TrainContext):
    """Extends :class:`TrainContext` for DPOP optimization with process weights."""

    process_weight_index: int | None = static()

    @classmethod
    def from_dpop_train_struct(
        cls,
        struct: PompStruct,
        J: int,
        M: int,
        alpha_cooling: float,
        thresh: float,
        n_monitors: int,
        eta: jax.Array,
        alpha: float | jax.Array,
        process_weight_index: int | None,
    ) -> DpopTrainContext:
        return cls(
            series=SeriesData.from_struct(struct),
            alpha=alpha,
            fns=ModelFns.pf(struct),
            J=J,
            eta=eta,
            M=M,
            alpha_cooling=alpha_cooling,
            thresh=thresh,
            n_monitors=n_monitors,
            process_weight_index=process_weight_index,
        )


# ----PANEL TRAIN---------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class PanelTrainContext:
    series: SeriesData
    unit_param_permutations: jax.Array
    keys: jax.Array
    eta_shared: jax.Array
    eta_spec: jax.Array
    alpha: float
    fns: ModelFns = static()
    J: int = static()
    chunk_size: int = static()
    M: int = static()
    alpha_cooling: float = static()
    n_obs: int = static()
    U: int = static()

    @classmethod
    def from_panel_train_struct(
        cls,
        struct: PanelPompStruct,
        J: int,
        chunk_size: int,
        M: int,
        alpha_cooling: float,
        keys: jax.Array,
        eta_shared: jax.Array,
        eta_spec: jax.Array,
        alpha: float,
    ) -> PanelTrainContext:
        return cls(
            series=SeriesData.from_panel_struct(struct),
            unit_param_permutations=struct.unit_param_permutations,
            keys=keys,
            eta_shared=eta_shared,
            eta_spec=eta_spec,
            alpha=alpha,
            fns=ModelFns.pf(struct),
            J=J,
            chunk_size=chunk_size,
            M=M,
            alpha_cooling=alpha_cooling,
            n_obs=struct.ys_per_unit.shape[1],
            U=len(struct.unit_names),
        )

    def to_mop_context(self) -> MopContext:
        return MopContext(series=self.series, alpha=self.alpha, fns=self.fns, J=self.J)


# ----MIF-----------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class MifContext:
    series: SeriesData
    rw_sigma: RWSigma
    per: ModelFns = static()
    pf: ModelFns = static()
    J: int = static()
    M: int = static()
    thresh: float = static(default=0.0)
    n_monitors: int = static(default=0)
    return_ancestry: bool = static(default=False)

    @classmethod
    def from_struct(
        cls,
        struct: PompStruct,
        rw_sigma: RWSigma,
        J: int,
        M: int,
        thresh: float = 0.0,
        n_monitors: int = 0,
        return_ancestry: bool = False,
    ) -> MifContext:
        return cls(
            series=SeriesData.from_struct(struct),
            rw_sigma=rw_sigma,
            per=ModelFns.per(struct),
            pf=ModelFns.pf(struct),
            J=J,
            M=M,
            thresh=thresh,
            n_monitors=n_monitors,
            return_ancestry=return_ancestry,
        )

    def to_pfilter_context(self) -> PfilterContext:
        return PfilterContext(
            series=self.series,
            fns=self.pf,
            J=self.J,
            thresh=self.thresh,
            should_trans=True,
        )


# ----PANEL MIF-----------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class PanelMifContext:
    series: SeriesData
    rw_sigma: RWSigma
    unit_param_permutations: jax.Array
    per: ModelFns = static()
    pf: ModelFns = static()
    J: int = static()
    M: int = static()
    U: int = static()
    thresh: float = static(default=0.0)
    n_monitors: int = static(default=0)
    block: bool = static(default=True)

    @classmethod
    def from_struct(
        cls,
        struct: PanelPompStruct,
        rw_sigma: RWSigma,
        J: int,
        M: int,
        U: int,
        thresh: float = 0.0,
        n_monitors: int = 0,
        block: bool = True,
    ) -> PanelMifContext:
        return cls(
            series=SeriesData.from_panel_struct(struct),
            rw_sigma=rw_sigma,
            unit_param_permutations=struct.unit_param_permutations,
            per=ModelFns.per(struct),
            pf=ModelFns.pf(struct),
            J=J,
            M=M,
            U=U,
            thresh=thresh,
            n_monitors=n_monitors,
            block=block,
        )

    def to_pfilter_context(self) -> PfilterContext:
        return PfilterContext(
            series=self.series,
            fns=self.pf,
            J=self.J,
            thresh=self.thresh,
            should_trans=True,
        )

    def to_mif_context(
        self,
        ys_u: jax.Array,
        rw_sigma_u: RWSigma,
        covars_u: jax.Array | None,
    ) -> MifContext:
        """Narrow this panel context to the single-unit MIF context for one unit."""
        return MifContext(
            series=SeriesData(
                ys=ys_u,
                dt_array_extended=self.series.dt_array_extended,
                nstep_array=self.series.nstep_array,
                t0=self.series.t0,
                times=self.series.times,
                covars_extended=covars_u,
            ),
            rw_sigma=rw_sigma_u,
            per=self.per,
            pf=self.pf,
            J=self.J,
            M=self.M,
            thresh=self.thresh,
            n_monitors=self.n_monitors,
            return_ancestry=not self.block,
        )


# ----ABC-----------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class AbcContext:
    series: SeriesData
    obs_probes: jax.Array
    scale_arr: jax.Array
    epsilon: float
    fns: SimFns = static()
    M: int = static()
    rmeasure: Callable = static()
    dprior: Callable = static()
    probe_fn: Callable = static()
    ydim: int = static(default=1)

    @classmethod
    def from_struct(
        cls,
        struct: PompStruct,
        M: int,
        obs_probes: jax.Array,
        scale_arr: jax.Array,
        epsilon: float,
        dprior: Callable | None = None,
        probe_fn: Callable = lambda y: y,
        ydim: int = 1,
    ) -> AbcContext:
        if struct.rmeas_pf is None:
            raise ValueError("abc requires struct.rmeas_pf to be non-None.")
        dprior_fn = dprior if dprior is not None else struct.dprior_pf
        if dprior_fn is None:
            raise ValueError("dprior is required for ABC")
        return cls(
            series=SeriesData.from_struct(struct),
            obs_probes=obs_probes,
            scale_arr=scale_arr,
            epsilon=epsilon,
            fns=SimFns.pf(struct),
            M=M,
            rmeasure=struct.rmeas_pf,
            dprior=dprior_fn,
            probe_fn=probe_fn,
            ydim=ydim,
        )


# ----PMCMC---------------------------------------------------------------------


@register_dataclass
@dataclass(frozen=True, kw_only=True)
class PmcmcContext:
    series: SeriesData
    fns: ModelFns = static()
    M: int = static()
    J: int = static()
    dprior: Callable = static()
    thresh: float = static(default=0.0)

    @classmethod
    def from_struct(
        cls,
        struct: PompStruct,
        M: int,
        J: int,
        dprior: Callable | None = None,
        thresh: float = 0.0,
    ) -> PmcmcContext:
        dprior_fn = dprior if dprior is not None else struct.dprior_pf
        if dprior_fn is None:
            raise ValueError("dprior is required for PMCMC")
        return cls(
            series=SeriesData.from_struct(struct),
            fns=ModelFns.pf(struct),
            M=M,
            J=J,
            dprior=dprior_fn,
            thresh=thresh,
        )

    def to_pfilter_context(self) -> PfilterContext:
        return PfilterContext(
            series=self.series,
            fns=self.fns,
            J=self.J,
            thresh=self.thresh,
            should_trans=True,
        )

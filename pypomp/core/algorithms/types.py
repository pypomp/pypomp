from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import jax
from jax.tree_util import register_dataclass
from ...functional.structs import PompStruct, PanelPompStruct

# ----MOP---------------------------------------


@register_dataclass
@dataclass(frozen=True)
class MopConfig:
    J: int
    rinitializer: Callable
    rprocess_interp: Callable
    dmeasure: Callable
    accumvars: tuple[int, ...] | None

    def to_mop_config(self) -> MopConfig:
        return MopConfig(
            J=self.J,
            rinitializer=self.rinitializer,
            rprocess_interp=self.rprocess_interp,
            dmeasure=self.dmeasure,
            accumvars=self.accumvars,
        )

    @classmethod
    def from_mop_struct(cls, struct: PompStruct, J: int) -> MopConfig:
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure is required for MOP")
        return cls(
            J=J,
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            dmeasure=struct.dmeas_pf,
            accumvars=struct.accumvars,
        )


@register_dataclass
@dataclass(frozen=True)
class MopInputs:
    ys: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    covars_extended: jax.Array | None
    alpha: float | jax.Array

    def to_mop_inputs(self) -> MopInputs:
        return MopInputs(
            ys=self.ys,
            dt_array_extended=self.dt_array_extended,
            nstep_array=self.nstep_array,
            t0=self.t0,
            times=self.times,
            covars_extended=self.covars_extended,
            alpha=self.alpha,
        )

    def to_pfilter_inputs(self) -> PfilterInputs:
        return PfilterInputs(
            ys=self.ys,
            dt_array_extended=self.dt_array_extended,
            nstep_array=self.nstep_array,
            t0=self.t0,
            times=self.times,
            covars_extended=self.covars_extended,
        )

    @classmethod
    def from_mop_struct(cls, struct: PompStruct, alpha: float | jax.Array) -> MopInputs:
        return cls(
            ys=struct.ys,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            covars_extended=struct.covars_extended,
            alpha=alpha,
        )


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


@register_dataclass
@dataclass(frozen=True)
class TrainConfig(MopConfig):
    M: int
    alpha_cooling: float
    thresh: float
    n_monitors: int

    @classmethod
    def from_train_struct(
        cls,
        struct: PompStruct,
        J: int,
        M: int,
        alpha_cooling: float,
        thresh: float,
        n_monitors: int,
    ) -> TrainConfig:
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure is required for training")
        return cls(
            J=J,
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            dmeasure=struct.dmeas_pf,
            accumvars=struct.accumvars,
            M=M,
            alpha_cooling=alpha_cooling,
            thresh=thresh,
            n_monitors=n_monitors,
        )

    def to_pfilter_config(
        self,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        should_trans: bool = False,
    ) -> PfilterConfig:
        return PfilterConfig(
            J=self.J,
            rinitializer=self.rinitializer,
            rprocess_interp=self.rprocess_interp,
            dmeasure=self.dmeasure,
            accumvars=self.accumvars,
            thresh=self.thresh,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
            should_trans=should_trans,
        )


# ----TRAIN---------------------------------------


@register_dataclass
@dataclass(frozen=True)
class TrainInputs(MopInputs):
    eta: jax.Array

    @classmethod
    def from_train_struct(
        cls,
        struct: PompStruct,
        eta: jax.Array,
        alpha: float | jax.Array,
    ) -> TrainInputs:
        return cls(
            ys=struct.ys,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            covars_extended=struct.covars_extended,
            alpha=alpha,
            eta=eta,
        )


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
class PanelTrainConfig(MopConfig):
    chunk_size: int
    M: int
    alpha_cooling: float
    n_obs: int
    U: int

    @classmethod
    def from_panel_train_struct(
        cls,
        struct: PanelPompStruct,
        J: int,
        chunk_size: int,
        M: int,
        alpha_cooling: float,
    ) -> PanelTrainConfig:
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure is required for panel training")
        return cls(
            J=J,
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            dmeasure=struct.dmeas_pf,
            accumvars=struct.accumvars,
            chunk_size=chunk_size,
            M=M,
            alpha_cooling=alpha_cooling,
            n_obs=struct.ys_per_unit.shape[1],
            U=len(struct.unit_names),
        )


@register_dataclass
@dataclass(frozen=True)
class PanelTrainInputs:
    unit_param_permutations: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    ys: jax.Array
    covars_extended: jax.Array | None
    keys: jax.Array
    eta_shared: jax.Array
    eta_spec: jax.Array
    alpha: float

    def to_mop_inputs(self) -> MopInputs:
        return MopInputs(
            ys=self.ys,
            dt_array_extended=self.dt_array_extended,
            nstep_array=self.nstep_array,
            t0=self.t0,
            times=self.times,
            covars_extended=self.covars_extended,
            alpha=self.alpha,
        )

    @classmethod
    def from_panel_train_struct(
        cls,
        struct: PanelPompStruct,
        keys: jax.Array,
        eta_shared: jax.Array,
        eta_spec: jax.Array,
        alpha: float,
    ) -> PanelTrainInputs:
        return cls(
            unit_param_permutations=struct.unit_param_permutations,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            ys=struct.ys_per_unit,
            covars_extended=struct.covars_per_unit,
            keys=keys,
            eta_shared=eta_shared,
            eta_spec=eta_spec,
            alpha=alpha,
        )


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


# ----PFILTER---------------------------------------


@register_dataclass
@dataclass(frozen=True)
class PfilterConfig(MopConfig):
    thresh: float = 0.0
    CLL: bool = False
    ESS: bool = False
    filter_mean: bool = False
    prediction_mean: bool = False
    should_trans: bool = False

    @classmethod
    def from_pfilter_struct(
        cls,
        struct: PompStruct,
        J: int,
        thresh: float = 0.0,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        should_trans: bool = False,
    ) -> PfilterConfig:
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure is required for particle filtering")
        return cls(
            J=J,
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            dmeasure=struct.dmeas_pf,
            accumvars=struct.accumvars,
            thresh=thresh,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
            should_trans=should_trans,
        )

    @classmethod
    def from_panel_pfilter_struct(
        cls,
        struct: PanelPompStruct,
        J: int,
        thresh: float = 0.0,
        CLL: bool = False,
        ESS: bool = False,
        filter_mean: bool = False,
        prediction_mean: bool = False,
        should_trans: bool = False,
    ) -> PfilterConfig:
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure is required for particle filtering")
        return cls(
            J=J,
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            dmeasure=struct.dmeas_pf,
            accumvars=struct.accumvars,
            thresh=thresh,
            CLL=CLL,
            ESS=ESS,
            filter_mean=filter_mean,
            prediction_mean=prediction_mean,
            should_trans=should_trans,
        )


@register_dataclass
@dataclass(frozen=True)
class PfilterInputs:
    ys: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    covars_extended: jax.Array | None

    @classmethod
    def from_pfilter_struct(cls, struct: PompStruct) -> PfilterInputs:
        return cls(
            ys=struct.ys,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            covars_extended=struct.covars_extended,
        )

    @classmethod
    def from_panel_pfilter_struct(cls, struct: PanelPompStruct) -> PfilterInputs:
        return cls(
            ys=struct.ys_per_unit,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            covars_extended=struct.covars_per_unit,
        )

    def to_pfilter_inputs(self) -> PfilterInputs:
        return self


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


# ----MIF-----------------------------------------


@register_dataclass
@dataclass(frozen=True)
class MifConfig:
    J: int
    M: int
    rinitializer: Callable
    rprocess_interp: Callable
    dmeasure: Callable
    rinitializer_pf: Callable
    rprocess_pf: Callable
    dmeasure_pf: Callable
    cooling_fn: Callable
    accumvars: tuple[int, ...] | None
    thresh: float
    n_monitors: int
    return_ancestry: bool = False

    def to_pfilter_config(self) -> PfilterConfig:
        return PfilterConfig(
            J=self.J,
            rinitializer=self.rinitializer_pf,
            rprocess_interp=self.rprocess_pf,
            dmeasure=self.dmeasure_pf,
            accumvars=self.accumvars,
            thresh=self.thresh,
            CLL=False,
            ESS=False,
            filter_mean=False,
            prediction_mean=False,
            should_trans=True,
        )

    @classmethod
    def from_mif_struct(
        cls,
        struct: PompStruct,
        J: int,
        M: int,
        cooling_fn: Callable,
        thresh: float = 0.0,
        n_monitors: int = 0,
        return_ancestry: bool = False,
    ) -> MifConfig:
        if struct.dmeas_per is None:
            raise ValueError("dmeasure is required for MIF")
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure_pf is required for MIF")
        return cls(
            J=J,
            M=M,
            rinitializer=struct.rinit_per,
            rprocess_interp=struct.rproc_per,
            dmeasure=struct.dmeas_per,
            rinitializer_pf=struct.rinit_pf,
            rprocess_pf=struct.rproc_pf,
            dmeasure_pf=struct.dmeas_pf,
            cooling_fn=cooling_fn,
            accumvars=struct.accumvars,
            thresh=thresh,
            n_monitors=n_monitors,
            return_ancestry=return_ancestry,
        )


@register_dataclass
@dataclass(frozen=True)
class MifInputs:
    ys: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    sigmas: float | jax.Array
    sigmas_init: float | jax.Array
    covars_extended: jax.Array | None

    def to_pfilter_inputs(self) -> PfilterInputs:
        return PfilterInputs(
            ys=self.ys,
            dt_array_extended=self.dt_array_extended,
            nstep_array=self.nstep_array,
            t0=self.t0,
            times=self.times,
            covars_extended=self.covars_extended,
        )

    @classmethod
    def from_mif_struct(
        cls,
        struct: PompStruct,
        sigmas: float | jax.Array,
        sigmas_init: float | jax.Array,
    ) -> MifInputs:
        return cls(
            ys=struct.ys,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            covars_extended=struct.covars_extended,
        )


@register_dataclass
@dataclass(frozen=True)
class PanelMifConfig:
    J: int
    M: int
    U: int
    rinitializer: Callable
    rprocess_interp: Callable
    dmeasure: Callable
    rinitializer_pf: Callable
    rprocess_pf: Callable
    dmeasure_pf: Callable
    cooling_fn: Callable
    accumvars: tuple[int, ...] | None
    thresh: float
    n_monitors: int
    block: bool

    def to_pfilter_config(self) -> PfilterConfig:
        return PfilterConfig(
            J=self.J,
            rinitializer=self.rinitializer_pf,
            rprocess_interp=self.rprocess_pf,
            dmeasure=self.dmeasure_pf,
            accumvars=self.accumvars,
            thresh=self.thresh,
            CLL=False,
            ESS=False,
            filter_mean=False,
            prediction_mean=False,
            should_trans=True,
        )

    def to_mif_config(self) -> MifConfig:
        return MifConfig(
            J=self.J,
            M=self.M,
            rinitializer=self.rinitializer,
            rprocess_interp=self.rprocess_interp,
            dmeasure=self.dmeasure,
            rinitializer_pf=self.rinitializer_pf,
            rprocess_pf=self.rprocess_pf,
            dmeasure_pf=self.dmeasure_pf,
            cooling_fn=self.cooling_fn,
            accumvars=self.accumvars,
            thresh=self.thresh,
            n_monitors=self.n_monitors,
            return_ancestry=not self.block,
        )

    @classmethod
    def from_panel_mif_struct(
        cls,
        struct: PanelPompStruct,
        J: int,
        M: int,
        U: int,
        cooling_fn: Callable,
        thresh: float = 0.0,
        n_monitors: int = 0,
        block: bool = True,
    ) -> PanelMifConfig:
        if struct.dmeas_per is None:
            raise ValueError("dmeasure is required for Panel MIF")
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure_pf is required for Panel MIF")
        return cls(
            J=J,
            M=M,
            U=U,
            rinitializer=struct.rinit_per,
            rprocess_interp=struct.rproc_per,
            dmeasure=struct.dmeas_per,
            rinitializer_pf=struct.rinit_pf,
            rprocess_pf=struct.rproc_pf,
            dmeasure_pf=struct.dmeas_pf,
            cooling_fn=cooling_fn,
            accumvars=struct.accumvars,
            thresh=thresh,
            n_monitors=n_monitors,
            block=block,
        )


@register_dataclass
@dataclass(frozen=True)
class PanelMifInputs:
    ys: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    sigmas: jax.Array
    sigmas_init: jax.Array
    covars_extended: jax.Array | None
    unit_param_permutations: jax.Array

    def to_mif_inputs(
        self,
        ys_u: jax.Array,
        sigmas_u: jax.Array,
        sigmas_init_u: jax.Array,
        covars_u: jax.Array | None,
    ) -> MifInputs:
        return MifInputs(
            ys=ys_u,
            dt_array_extended=self.dt_array_extended,
            nstep_array=self.nstep_array,
            t0=self.t0,
            times=self.times,
            sigmas=sigmas_u,
            sigmas_init=sigmas_init_u,
            covars_extended=covars_u,
        )

    @classmethod
    def from_panel_mif_struct(
        cls,
        struct: PanelPompStruct,
        sigmas: jax.Array,
        sigmas_init: jax.Array,
    ) -> PanelMifInputs:
        return cls(
            ys=struct.ys_per_unit,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            covars_extended=struct.covars_per_unit,
            unit_param_permutations=struct.unit_param_permutations,
        )


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


# ----ABC---------------------------------------


@register_dataclass
@dataclass(frozen=True)
class AbcConfig:
    M: int
    rinitializer: Callable
    rprocess_interp: Callable
    rmeasure: Callable
    accumvars: tuple[int, ...] | None
    dprior: Callable
    probe_fn: Callable
    ydim: int

    @classmethod
    def from_abc_struct(
        cls,
        struct: PompStruct,
        M: int,
        dprior: Callable,
        probe_fn: Callable,
        ydim: int,
    ) -> AbcConfig:
        if struct.rmeas_pf is None:
            raise ValueError("abc requires struct.rmeas_pf to be non-None.")
        return cls(
            M=M,
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            rmeasure=struct.rmeas_pf,
            accumvars=struct.accumvars,
            dprior=dprior,
            probe_fn=probe_fn,
            ydim=ydim,
        )


@register_dataclass
@dataclass(frozen=True)
class AbcInputs:
    obs_probes: jax.Array
    scale_arr: jax.Array
    epsilon: float
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    covars_extended: jax.Array | None

    @classmethod
    def from_abc_struct(
        cls,
        struct: PompStruct,
        obs_probes: jax.Array,
        scale_arr: jax.Array,
        epsilon: float,
    ) -> AbcInputs:
        return cls(
            obs_probes=obs_probes,
            scale_arr=scale_arr,
            epsilon=epsilon,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times,
            covars_extended=struct.covars_extended,
        )


# ----PMCMC---------------------------------------


@register_dataclass
@dataclass(frozen=True)
class PmcmcConfig:
    M: int
    J: int
    rinitializer: Callable
    rprocess_interp: Callable
    dmeasure: Callable
    accumvars: tuple[int, ...] | None
    dprior: Callable
    thresh: float
    should_trans: bool = False

    @classmethod
    def from_pmcmc_struct(
        cls,
        struct: PompStruct,
        M: int,
        J: int,
        dprior: Callable,
        thresh: float = 0.0,
        should_trans: bool = False,
    ) -> PmcmcConfig:
        if struct.dmeas_pf is None:
            raise ValueError("dmeasure is required for PMCMC")
        return cls(
            M=M,
            J=J,
            rinitializer=struct.rinit_pf,
            rprocess_interp=struct.rproc_pf,
            dmeasure=struct.dmeas_pf,
            accumvars=struct.accumvars,
            dprior=dprior,
            thresh=thresh,
            should_trans=should_trans,
        )

    def to_pfilter_config(self) -> PfilterConfig:
        return PfilterConfig(
            J=self.J,
            rinitializer=self.rinitializer,
            rprocess_interp=self.rprocess_interp,
            dmeasure=self.dmeasure,
            accumvars=self.accumvars,
            thresh=self.thresh,
            should_trans=self.should_trans,
        )


@register_dataclass
@dataclass(frozen=True)
class PmcmcInputs:
    ys: jax.Array
    dt_array_extended: jax.Array
    nstep_array: jax.Array
    t0: float
    times: jax.Array
    covars_extended: jax.Array | None

    @classmethod
    def from_pmcmc_struct(cls, struct: PompStruct) -> PmcmcInputs:
        return cls(
            ys=struct.ys,
            dt_array_extended=struct.dt_array_extended,
            nstep_array=struct.nstep_array,
            t0=struct.t0,
            times=struct.times.astype(float),
            covars_extended=struct.covars_extended,
        )

    def to_pfilter_inputs(self) -> PfilterInputs:
        return PfilterInputs(
            ys=self.ys,
            dt_array_extended=self.dt_array_extended,
            nstep_array=self.nstep_array,
            t0=self.t0,
            times=self.times,
            covars_extended=self.covars_extended,
        )

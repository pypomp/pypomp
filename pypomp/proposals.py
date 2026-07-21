"""
JIT-compatible MCMC proposal distributions for use with PMCMC and ABC.

Each proposal is a frozen dataclass registered as a JAX PyTree so it can flow
through ``jax.lax.scan`` and ``jax.vmap``.  Proposals expose three methods:

* ``init_state(theta_arr)`` returns the initial scan-carried state (a PyTree).
  For stateless proposals this is an empty container.
* ``step(state, theta_arr, key, n, accepts)`` is a pure JAX function that
  returns ``(theta_proposed, new_state)``.  All inputs/outputs are
  ``jax.Array``; no Python branching on traced values.
* ``canonicalize(canonical_names)`` returns a new proposal instance expanded
  and reordered to match the model's full canonical parameter vector.  This is
  applied automatically by the functional ``pmcmc`` / ``abc`` entry points, so
  callers do not need to invoke it themselves.

Three proposals are provided, following pomp (R):

* :class:`MVNDiagRW`         -- diagonal random walk
* :class:`MVNRWFull`         -- full-covariance random walk
* :class:`MVNRWAdaptive`     -- Roberts & Rosenthal 2009 adaptive scheme

Each is constructed directly via its ``__init__`` (e.g. ``MVNDiagRW({...})``)
and validated host-side; the standard-deviation / covariance arrays are stored
as ``numpy`` arrays (PyTree leaves) so instances pickle cheaply, while parameter
names and scalar settings are static PyTree aux metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Protocol, Any

import jax
import jax.numpy as jnp
import numpy as np


class Proposal(Protocol):
    """Protocol defining the interface for MCMC proposal distributions."""

    def init_state(self, theta_arr: jax.Array) -> Any:
        """Initialize carried state for MCMC scan loop."""
        ...

    def step(
        self,
        state: Any,
        theta_arr: jax.Array,
        key: jax.Array,
        n: jax.Array | int,
        accepts: jax.Array | int,
    ) -> tuple[jax.Array, Any]:
        """Generate a proposed parameter vector and updated state."""
        ...

    def canonicalize(self, canonical_names: Sequence[str]) -> Proposal:
        """Canonicalize proposal to match full model parameter vector."""
        ...


# ---------------------------------------------------------------------------
# Stateless proposals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False, eq=False)
class MVNDiagRW:
    """Diagonal multivariate normal random-walk proposal.

    Construct from a dict of per-parameter random-walk standard deviations;
    parameters with ``sd <= 0`` are silently dropped.

    Attributes
    ----------
    sd_arr : np.ndarray
        ``(d,)`` array of per-parameter random-walk standard
        deviations, in the order given by ``param_names``.
    param_names : tuple of str
        Tuple of parameter names corresponding to ``sd_arr``.
    """

    sd_arr: np.ndarray  # stored as numpy array so class can be pickled efficiently
    param_names: tuple[str, ...]

    def __init__(self, rw_sd: dict[str, float]):
        """Construct a diagonal multivariate normal random-walk proposal.

        Args:
            rw_sd: Dictionary mapping parameter names to random-walk standard
                deviations.  Parameters with ``sd <= 0`` are silently dropped.
        """
        rw_sd = {k: float(v) for k, v in rw_sd.items() if v > 0}
        if not rw_sd:
            raise ValueError("rw_sd must contain at least one positive entry.")
        param_names = tuple(rw_sd.keys())
        sd_arr = np.asarray([rw_sd[p] for p in param_names])
        object.__setattr__(self, "sd_arr", sd_arr)
        object.__setattr__(self, "param_names", param_names)

    @classmethod
    def _from_leaves(cls, sd_arr: Any, param_names: tuple[str, ...]) -> MVNDiagRW:
        """Rebuild from raw leaves + aux (for PyTree unflatten; skips validation)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "sd_arr", sd_arr)
        object.__setattr__(obj, "param_names", tuple(param_names))
        return obj

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MVNDiagRW):
            return NotImplemented
        return self.param_names == other.param_names and np.array_equal(
            np.asarray(self.sd_arr), np.asarray(other.sd_arr)
        )

    def init_state(self, theta_arr: jax.Array) -> tuple:
        """Initialize carried state (empty tuple for stateless proposal)."""
        return ()

    def step(
        self,
        state: tuple,
        theta_arr: jax.Array,
        key: jax.Array,
        n: jax.Array | int,
        accepts: jax.Array | int,
    ) -> tuple[jax.Array, tuple]:
        """Propose parameters via diagonal multivariate normal random walk."""
        z = jax.random.normal(key, shape=theta_arr.shape)
        return theta_arr + z * self.sd_arr, state

    def canonicalize(self, canonical_names: Sequence[str]) -> MVNDiagRW:
        """Expand and reorder standard deviations to match canonical parameter names."""
        names = tuple(canonical_names)
        d = len(names)
        name_to_idx = {n: i for i, n in enumerate(names)}
        full_sd = np.zeros(d)
        for i_local, p in enumerate(self.param_names):
            if p not in name_to_idx:
                raise ValueError(f"Proposal parameter {p!r} not in model.")
            full_sd[name_to_idx[p]] = self.sd_arr[i_local]
        return MVNDiagRW._from_leaves(full_sd, names)


jax.tree_util.register_pytree_node(
    MVNDiagRW,
    lambda p: ((p.sd_arr,), p.param_names),
    lambda aux, children: MVNDiagRW._from_leaves(children[0], aux),
)


@dataclass(frozen=True, init=False, eq=False)
class MVNRWFull:
    """Full-covariance multivariate normal random-walk proposal.

    Construct from a symmetric positive-definite covariance matrix and the
    parameter names corresponding to its rows/columns.

    Attributes
    ----------
    chol : np.ndarray
        ``(d, d)`` lower-triangular Cholesky factor of the proposal
        covariance.
    param_names : tuple of str
        Tuple of parameter names corresponding to the rows/columns
        of the covariance.
    """

    chol: np.ndarray  # stored as numpy array so class can be pickled efficiently
    param_names: tuple[str, ...]

    def __init__(self, rw_var: np.ndarray, param_names: list[str]):
        """Construct a full-covariance multivariate normal random-walk proposal.

        Args:
            rw_var: Symmetric positive-definite covariance matrix of shape
                ``(d, d)`` where ``d = len(param_names)``.
            param_names: Parameter names corresponding to the rows/columns of
                ``rw_var``.
        """
        rw_var = np.asarray(rw_var, dtype=float)
        if rw_var.ndim != 2 or rw_var.shape[0] != rw_var.shape[1]:
            raise ValueError("rw_var must be a square matrix.")
        if rw_var.shape[0] != len(param_names):
            raise ValueError("rw_var dimensions must match len(param_names).")
        chol = np.linalg.cholesky(rw_var)
        object.__setattr__(self, "chol", chol)
        object.__setattr__(self, "param_names", tuple(param_names))

    @classmethod
    def _from_leaves(cls, chol: Any, param_names: tuple[str, ...]) -> MVNRWFull:
        """Rebuild from raw leaves + aux (for PyTree unflatten; skips validation)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "chol", chol)
        object.__setattr__(obj, "param_names", tuple(param_names))
        return obj

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MVNRWFull):
            return NotImplemented
        return self.param_names == other.param_names and np.array_equal(
            np.asarray(self.chol), np.asarray(other.chol)
        )

    def init_state(self, theta_arr: jax.Array) -> tuple:
        """Initialize carried state (empty tuple for stateless proposal)."""
        return ()

    def step(
        self,
        state: tuple,
        theta_arr: jax.Array,
        key: jax.Array,
        n: jax.Array | int,
        accepts: jax.Array | int,
    ) -> tuple[jax.Array, tuple]:
        """Propose parameters via full-covariance multivariate normal random walk."""
        z = jax.random.normal(key, shape=theta_arr.shape)
        return theta_arr + self.chol @ z, state

    def canonicalize(self, canonical_names: Sequence[str]) -> MVNRWFull:
        """Expand and reorder Cholesky factor to match canonical parameter names."""
        names = tuple(canonical_names)
        d = len(names)
        name_to_idx = {n: i for i, n in enumerate(names)}
        full_chol = np.zeros((d, d))
        for i_local, p_i in enumerate(self.param_names):
            if p_i not in name_to_idx:
                raise ValueError(f"Proposal parameter {p_i!r} not in model.")
            i_global = name_to_idx[p_i]
            for j_local, p_j in enumerate(self.param_names):
                if p_j not in name_to_idx:
                    raise ValueError(f"Proposal parameter {p_j!r} not in model.")
                j_global = name_to_idx[p_j]
                full_chol[i_global, j_global] = self.chol[i_local, j_local]
        return MVNRWFull._from_leaves(full_chol, names)


jax.tree_util.register_pytree_node(
    MVNRWFull,
    lambda p: ((p.chol,), p.param_names),
    lambda aux, children: MVNRWFull._from_leaves(children[0], aux),
)


# ---------------------------------------------------------------------------
# Adaptive proposal  (Roberts & Rosenthal, 2009)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveState:
    """Mutable per-iteration state carried by :class:`MVNRWAdaptive` through scan.

    Attributes:
        scaling: Scalar scaling factor (Phase 1 scale adaptation).
        theta_mean: ``(d,)`` running mean of theta values (Welford).
        covmat_emp: ``(d, d)`` running empirical covariance accumulator
            (Welford).
        initialized: ``()`` flag (0/1) indicating whether ``theta_mean`` was
            seeded by the first call.
    """

    scaling: jax.Array
    theta_mean: jax.Array
    covmat_emp: jax.Array
    initialized: jax.Array


jax.tree_util.register_pytree_node(
    AdaptiveState,
    lambda s: ((s.scaling, s.theta_mean, s.covmat_emp, s.initialized), None),
    lambda _, children: AdaptiveState(
        scaling=children[0],
        theta_mean=children[1],
        covmat_emp=children[2],
        initialized=children[3],
    ),
)


@dataclass(frozen=True, init=False, eq=False)
class MVNRWAdaptive:
    """Adaptive multivariate normal random-walk proposal (Roberts & Rosenthal 2009).

    Implements two-phase adaptation:

    * **Phase 1** (``n >= scale_start`` and ``accepts < shape_start``):
      a global scaling factor is adjusted to drive the acceptance ratio
      toward ``target``.
    * **Phase 2** (``accepts >= shape_start``): the proposal covariance is
      replaced by the scaled empirical covariance ``(2.38^2 / d) * cov_emp``.

    Cholesky is computed each step with a small jitter (``1e-10 * I``) for
    JIT-friendly numerical robustness.

    Attributes
    ----------
    init_rw_var : jax.Array
        ``(d, d)`` initial covariance matrix.
    param_names : tuple of str
        Tuple of parameter names corresponding to rows/columns
        of ``init_rw_var``.
    scale_start : int
        Iteration index at which Phase 1 begins.
    scale_cooling : float
        Cooling base for the scale update.
    shape_start : int
        Accepted-proposal count at which to switch to Phase 2.
    target : float
        Target Metropolis acceptance ratio.
    max_scaling : float
        Upper bound for the scaling factor.
    """

    init_rw_var: np.ndarray  # stored as numpy array so class can be pickled efficiently
    param_names: tuple[str, ...]
    scale_start: int
    scale_cooling: float
    shape_start: int
    target: float
    max_scaling: float

    def __init__(
        self,
        rw_sd: dict[str, float] | None = None,
        rw_var: np.ndarray | None = None,
        param_names: list[str] | None = None,
        scale_start: int = 200,
        scale_cooling: float = 0.999,
        shape_start: int = 200,
        target: float = 0.234,
        max_scaling: float = 50.0,
    ):
        """Construct an adaptive MVN random-walk proposal (Roberts & Rosenthal 2009).

        Provide exactly one of ``rw_sd`` (diagonal initialisation) or ``rw_var``
        (full initial covariance).

        Parameters
        ----------
        rw_sd : dict, optional
            Named dict of per-parameter random-walk SDs.
        rw_var : array_like, optional
            Full initial covariance matrix.
        param_names : list of str, optional
            Required when ``rw_var`` is supplied.
        scale_start : int, default 200
            Iteration at which to begin scale adaptation.
        scale_cooling : float, default 0.999
            Cooling base for the scale update (in (0, 1]).
        shape_start : int, default 200
            Number of accepted proposals before switching to empirical covariance.
        target : float, default 0.234
            Target Metropolis acceptance ratio.
        max_scaling : float, default 50.0
            Upper bound for the scaling factor.
        """
        if (rw_sd is None) == (rw_var is None):
            raise ValueError("Exactly one of rw_sd and rw_var must be given.")
        if rw_sd is not None:
            rw_sd = {k: float(v) for k, v in rw_sd.items() if v > 0}
            names = tuple(rw_sd.keys())
            init_var = np.diag([rw_sd[p] ** 2 for p in names])
        else:
            if param_names is None:
                raise ValueError("param_names required when rw_var is given.")
            init_var = np.asarray(rw_var, dtype=float)
            if init_var.shape != (len(param_names), len(param_names)):
                raise ValueError("rw_var shape must match param_names.")
            names = tuple(param_names)

        if scale_start < 1:
            raise ValueError("scale_start must be a positive integer.")
        if not (0 < scale_cooling <= 1):
            raise ValueError("scale_cooling must be in (0, 1].")
        if shape_start < 1:
            raise ValueError("shape_start must be a positive integer.")
        if not (0 < target < 1):
            raise ValueError("target must be in (0, 1).")

        object.__setattr__(self, "init_rw_var", np.asarray(init_var))
        object.__setattr__(self, "param_names", names)
        object.__setattr__(self, "scale_start", int(scale_start))
        object.__setattr__(self, "scale_cooling", float(scale_cooling))
        object.__setattr__(self, "shape_start", int(shape_start))
        object.__setattr__(self, "target", float(target))
        object.__setattr__(self, "max_scaling", float(max_scaling))

    @classmethod
    def _from_leaves(
        cls,
        init_rw_var: Any,
        param_names: tuple[str, ...],
        scale_start: int,
        scale_cooling: float,
        shape_start: int,
        target: float,
        max_scaling: float,
    ) -> MVNRWAdaptive:
        """Rebuild from raw leaves + aux (for PyTree unflatten; skips validation)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "init_rw_var", init_rw_var)
        object.__setattr__(obj, "param_names", tuple(param_names))
        object.__setattr__(obj, "scale_start", scale_start)
        object.__setattr__(obj, "scale_cooling", scale_cooling)
        object.__setattr__(obj, "shape_start", shape_start)
        object.__setattr__(obj, "target", target)
        object.__setattr__(obj, "max_scaling", max_scaling)
        return obj

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MVNRWAdaptive):
            return NotImplemented
        return (
            self.param_names == other.param_names
            and self.scale_start == other.scale_start
            and self.scale_cooling == other.scale_cooling
            and self.shape_start == other.shape_start
            and self.target == other.target
            and self.max_scaling == other.max_scaling
            and np.array_equal(
                np.asarray(self.init_rw_var), np.asarray(other.init_rw_var)
            )
        )

    def init_state(self, theta_arr: jax.Array) -> AdaptiveState:
        """Initialize adaptive state tracking running mean, empirical covariance, and scale."""
        d = theta_arr.shape[-1]
        dt = theta_arr.dtype
        return AdaptiveState(
            scaling=jnp.asarray(1.0, dtype=dt),
            theta_mean=jnp.zeros(d, dtype=dt),
            covmat_emp=jnp.zeros((d, d), dtype=dt),
            initialized=jnp.asarray(0.0, dtype=dt),
        )

    def step(
        self,
        state: AdaptiveState,
        theta_arr: jax.Array,
        key: jax.Array,
        n: jax.Array | int,
        accepts: jax.Array | int,
    ) -> tuple[jax.Array, AdaptiveState]:
        """Propose parameters and update adaptive state (Roberts & Rosenthal 2009)."""
        d = theta_arr.shape[-1]
        dt = theta_arr.dtype
        n_arr = jnp.asarray(n, dtype=dt)
        accepts_arr = jnp.asarray(accepts, dtype=dt)
        nf = jnp.maximum(n_arr, 1.0)
        af = accepts_arr

        # Lazy-seed theta_mean on the very first call.
        seeded_mean = jnp.where(state.initialized > 0, state.theta_mean, theta_arr)

        # ---- Phase 1: scale adaptation (if n >= scale_start and accepts < shape_start) ----
        accept_rate = af / nf
        cool = self.scale_cooling ** jnp.maximum(n_arr - self.scale_start, 0.0)
        scale_factor = jnp.exp(cool * (accept_rate - self.target))
        scaling_p1 = jnp.minimum(state.scaling * scale_factor, self.max_scaling)
        in_phase1 = (n >= self.scale_start) & (accepts < self.shape_start)
        scaling_new = jnp.where(in_phase1, scaling_p1, state.scaling)

        # ---- Choose covariance matrix ----
        cov_p1 = scaling_new**2 * self.init_rw_var
        cov_p2 = (2.38**2 / d) * state.covmat_emp
        in_phase2 = accepts >= self.shape_start
        covmat = jnp.where(
            in_phase2, cov_p2, jnp.where(in_phase1, cov_p1, self.init_rw_var)
        )

        # ---- Update running mean and empirical covariance (Welford) ----
        old_mean = seeded_mean
        new_mean = old_mean + (theta_arr - old_mean) / nf
        diff_old = theta_arr - old_mean
        diff_new = theta_arr - new_mean
        new_cov_emp = (
            (nf - 1.0) * state.covmat_emp + jnp.outer(diff_old, diff_new)
        ) / nf

        # ---- Draw proposal (Cholesky with jitter for stability) ----
        active_mask = jnp.any(self.init_rw_var != 0, axis=0) | jnp.any(
            self.init_rw_var != 0, axis=1
        )
        active_mask = active_mask.astype(dt)
        jitter = 1e-10 * jnp.eye(d, dtype=dt)
        chol = jnp.linalg.cholesky(covmat + jitter)
        z = jax.random.normal(key, shape=(d,))
        theta_proposed = theta_arr + (chol @ z) * active_mask

        new_state = AdaptiveState(
            scaling=scaling_new,
            theta_mean=new_mean,
            covmat_emp=new_cov_emp,
            initialized=jnp.asarray(1.0, dtype=theta_arr.dtype),
        )
        return theta_proposed, new_state

    def canonicalize(self, canonical_names: Sequence[str]) -> MVNRWAdaptive:
        """Expand and reorder initial covariance matrix to match canonical parameter names."""
        names = tuple(canonical_names)
        d = len(names)
        name_to_idx = {n: i for i, n in enumerate(names)}
        full_var = np.zeros((d, d))
        for i_local, p_i in enumerate(self.param_names):
            if p_i not in name_to_idx:
                raise ValueError(f"Proposal parameter {p_i!r} not in model.")
            i_global = name_to_idx[p_i]
            for j_local, p_j in enumerate(self.param_names):
                if p_j not in name_to_idx:
                    raise ValueError(f"Proposal parameter {p_j!r} not in model.")
                j_global = name_to_idx[p_j]
                full_var[i_global, j_global] = self.init_rw_var[i_local, j_local]
        return MVNRWAdaptive._from_leaves(
            init_rw_var=full_var,
            param_names=names,
            scale_start=self.scale_start,
            scale_cooling=self.scale_cooling,
            shape_start=self.shape_start,
            target=self.target,
            max_scaling=self.max_scaling,
        )


jax.tree_util.register_pytree_node(
    MVNRWAdaptive,
    lambda p: (
        (p.init_rw_var,),
        (
            p.param_names,
            p.scale_start,
            p.scale_cooling,
            p.shape_start,
            p.target,
            p.max_scaling,
        ),
    ),
    lambda aux, children: MVNRWAdaptive._from_leaves(
        init_rw_var=children[0],
        param_names=aux[0],
        scale_start=aux[1],
        scale_cooling=aux[2],
        shape_start=aux[3],
        target=aux[4],
        max_scaling=aux[5],
    ),
)

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Sequence, Union
import jax
import jax.numpy as jnp


# NOTE:
# Users can construct parameter transforms directly via `ParTrans(...)`
# or `ParTrans.from_names(...)`. The older two-step helper
@dataclass(frozen=True)
class ParTrans:
    # Indices (on the last axis) that are log-transformed when mapping
    # from natural scale -> estimation scale, and exp-transformed back.
    log_idx: Tuple[int, ...] = ()
    # Indices that are logit-transformed (natural in (0,1)) when mapping
    # to estimation scale, and sigmoid-inverted when mapping back.
    logit_idx: Tuple[int, ...] = ()
    # Indices that use user-provided custom transforms.
    # The corresponding `to_est_custom` / `from_est_custom` must be provided.
    custom_idx: Tuple[int, ...] = ()
    # Custom forward transform: natural scale -> estimation scale
    to_est_custom: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    # Custom inverse transform: estimation scale -> natural scale
    from_est_custom: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    @property
    def is_custom(self) -> bool:
        """Backward-compatible flag: True if any custom indices are present."""
        return len(self.custom_idx) > 0

    @classmethod
    def from_names(
        cls,
        paramnames: Sequence[str],
        *,
        log: Union[str, Sequence[str], None] = None,
        logit: Union[str, Sequence[str], None] = None,
        custom: Union[str, Sequence[str], None] = None,
        to_est: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        from_est: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ) -> "ParTrans":
        """
        Parameters
        ----------
        paramnames : list of str
            Canonical ordered parameter names. We will look up indices from here.
        log, logit, custom : str or list[str] or None
            Names to be placed in respective transform sets. Sets must be disjoint.
        to_est, from_est : Callable or None
            Custom transforms for `custom` indices. Must be provided together.
        """
        def _as_list(x):
            if x is None:
                return []
            return [x] if isinstance(x, str) else list(x)

        def _idx(xs):
            # Convert a list of names into a tuple of indices by lookup.
            return tuple(paramnames.index(n) for n in _as_list(xs))

        # Require paired custom transforms when any custom transform is requested.
        if (to_est is None) ^ (from_est is None):
            raise ValueError("Both to_est and from_est must be provided (or both None).")

        log_idx = _idx(log)
        logit_idx = _idx(logit)
        custom_idx = _idx(custom)
        _ensure_disjoint(log_idx, logit_idx, custom_idx)

        if custom_idx and (to_est is None or from_est is None):
            raise ValueError("Custom indices provided but to_est/from_est missing.")

        return cls(log_idx, logit_idx, custom_idx, to_est, from_est)


def parameter_trans(
    log: Union[str, int, Sequence[str], Sequence[int], None] = None,
    logit: Union[str, int, Sequence[str], Sequence[int], None] = None,
    *,
    custom: Union[str, int, Sequence[str], Sequence[int], None] = None,
    to_est: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    from_est: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    paramnames: Sequence[str] | None = None,
) -> ParTrans:
    """
    Usage patterns
    --------------
    - If `paramnames` is provided, you may pass *names* (str) in log/logit/custom
      and we will look up their indices.
    - Otherwise, log/logit/custom must be integer indices (int or Sequence[int]).

    The transform always acts on the *last* axis of input arrays; all leading
    batch axes are preserved verbatim (i.e., broadcasting in the last dimension).
    """
    # Custom transforms must come in pairs.
    if (to_est is None) ^ (from_est is None):
        raise ValueError("Both to_est and from_est must be provided (or both None).")

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    # Heuristically detect whether the user passed names (strings) or indices.
    has_str = any(
        isinstance(v, str) or (isinstance(v, (list, tuple)) and v and isinstance(v[0], str))
        for v in (log, logit, custom) if v is not None
    )

    if has_str:
        # Name-based usage requires `paramnames`
        if paramnames is None:
            raise TypeError(
                "When using parameter *names*, supply `paramnames` or call "
                "ParTrans.from_names(paramnames=..., ...)."
            )
        return ParTrans.from_names(
            paramnames,
            log=log if isinstance(log, (str, list, tuple)) else None,
            logit=logit if isinstance(logit, (str, list, tuple)) else None,
            custom=custom if isinstance(custom, (str, list, tuple)) else None,
            to_est=to_est,
            from_est=from_est,
        )

    # Otherwise treat inputs as index sets.
    def _to_idx_tuple(x) -> Tuple[int, ...]:
        xs = _as_list(x)
        return tuple(int(i) for i in xs)

    log_idx = _to_idx_tuple(log)
    logit_idx = _to_idx_tuple(logit)
    custom_idx = _to_idx_tuple(custom)
    _ensure_disjoint(log_idx, logit_idx, custom_idx)

    if custom_idx and (to_est is None or from_est is None):
        raise ValueError("Custom indices provided but to_est/from_est missing.")

    return ParTrans(log_idx, logit_idx, custom_idx, to_est, from_est)


def _ensure_disjoint(a: Tuple[int, ...], b: Tuple[int, ...], c: Tuple[int, ...]) -> None:
    """Ensure that the three index sets are pairwise disjoint."""
    if (set(a) & set(b)) or (set(a) & set(c)) or (set(b) & set(c)):
        raise ValueError("parameter_trans sets must be disjoint.")


def _pt_forward(theta_nat: jnp.ndarray, pt: ParTrans) -> jnp.ndarray:
    """
    Forward map: natural scale -> estimation scale,
    applied along the last dimension of `theta_nat`.
    """
    z = theta_nat
    if pt.log_idx:
        # Log-transform (positive reals). We clip to avoid log(0).
        li = jnp.array(pt.log_idx, dtype=jnp.int32)
        cols = jnp.take(z, li, axis=-1)
        z = z.at[..., li].set(jnp.log(jnp.clip(cols, min=1e-12)))
    if pt.logit_idx:
        # Logit-transform (values in (0, 1)). Clip to avoid +/-inf logits.
        qi = jnp.array(pt.logit_idx, dtype=jnp.int32)
        cols = jnp.take(z, qi, axis=-1)
        u = jnp.clip(cols, 1e-12, 1.0 - 1e-12)
        logits = jnp.log(u) - jnp.log1p(-u)
        z = z.at[..., qi].set(logits)
    if pt.custom_idx and pt.to_est_custom is not None:
        # User-supplied transform for `custom_idx`.
        ci = jnp.array(pt.custom_idx, dtype=jnp.int32)
        sub = jnp.take(z, ci, axis=-1)
        sub_t = pt.to_est_custom(sub)
        if sub_t.shape != sub.shape:
            raise ValueError("Custom to_est must return the same shape as its input subvector.")
        z = z.at[..., ci].set(sub_t)
    return z


def _pt_inverse(z_est: jnp.ndarray, pt: ParTrans) -> jnp.ndarray:
    """
    Inverse map: estimation scale -> natural scale,
    applied along the last dimension of `z_est`.
    """
    x = z_est
    if pt.custom_idx and pt.from_est_custom is not None:
        # Apply user-supplied inverse transform first (on the estimation-scale values).
        ci = jnp.array(pt.custom_idx, dtype=jnp.int32)
        sub = jnp.take(x, ci, axis=-1)
        sub_x = pt.from_est_custom(sub)
        if sub_x.shape != sub.shape:
            raise ValueError("Custom from_est must return the same shape as its input subvector.")
        x = x.at[..., ci].set(sub_x)
    if pt.log_idx:
        # Invert log-transform via exp.
        li = jnp.array(pt.log_idx, dtype=jnp.int32)
        cols = jnp.take(x, li, axis=-1)
        x = x.at[..., li].set(jnp.exp(cols))
    if pt.logit_idx:
        # Invert logit-transform via sigmoid.
        qi = jnp.array(pt.logit_idx, dtype=jnp.int32)
        cols = jnp.take(x, qi, axis=-1)
        vals = jax.nn.sigmoid(cols)
        x = x.at[..., qi].set(vals)
    return x


# A shared identity transform that can be imported across modules.
# It performs no transformation on any coordinates.
IDENTITY_PARTRANS = ParTrans()

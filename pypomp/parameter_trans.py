from typing import Callable, NamedTuple, Optional, Tuple, Sequence
import jax
import jax.numpy as jnp


class ParTrans(NamedTuple):
    is_custom: bool
    log_idx: Tuple[int, ...]
    logit_idx: Tuple[int, ...]
    custom_idx: Tuple[int, ...]
    to_est_custom: Optional[Callable] = None
    from_est_custom: Optional[Callable] = None


def parameter_trans(
    log: str | int | Sequence[str] | Sequence[int] | None = None,
    logit: str | int | Sequence[str] | Sequence[int] | None = None,
    *,
    custom: str | int | Sequence[str] | Sequence[int] | None = None,
    to_est: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    from_est: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> dict:
    def _tolist(x):
        if x is None:
            return []
        if isinstance(x, (str, int)):
            return [x]
        return list(x)
    if (to_est is None) ^ (from_est is None):
        raise ValueError("Both to_est and from_est must be provided (or both None).")
    return {
        "log_names": tuple(_tolist(log)),
        "logit_names": tuple(_tolist(logit)),
        "custom_names": tuple(_tolist(custom)),
        "to_est_custom": to_est,
        "from_est_custom": from_est,
    }


def materialize_partrans(spec: dict | ParTrans | None,
                         paramnames: Sequence[str] | None) -> ParTrans:
    if spec is None:
        return ParTrans(False, (), (), (), None, None)
    if isinstance(spec, ParTrans):
        return spec
    names = list(paramnames) if paramnames is not None else None

    def _to_index(x):
        if isinstance(x, int):
            return int(x)
        if names is None:
            raise TypeError("paramnames is None but the spec contains parameter names.")
        try:
            return int(names.index(x))
        except ValueError as e:
            raise ValueError(f"Unknown parameter name {x!r} in {names}") from e

    log_idx = tuple(_to_index(x) for x in spec.get("log_names", ()))
    logit_idx = tuple(_to_index(x) for x in spec.get("logit_names", ()))
    custom_idx = tuple(_to_index(x) for x in spec.get("custom_names", ()))

    def _overlap(a, b): return set(a).intersection(b)
    ov1 = _overlap(log_idx, logit_idx)
    ov2 = _overlap(log_idx, custom_idx)
    ov3 = _overlap(logit_idx, custom_idx)
    if ov1 or ov2 or ov3:
        raise ValueError(
            f"parameter_trans sets must be disjoint. Overlaps found: "
            f"log∩logit={sorted(ov1)}, log∩custom={sorted(ov2)}, logit∩custom={sorted(ov3)}"
        )

    to_custom = spec.get("to_est_custom", None)
    from_custom = spec.get("from_est_custom", None)
    has_custom = len(custom_idx) > 0
    if has_custom and ((to_custom is None) or (from_custom is None)):
        raise ValueError("You specified custom indices but did not provide to_est/from_est for them.")
    if (to_custom is not None or from_custom is not None) and not has_custom:
        raise ValueError("to_est/from_est were provided but 'custom' indices are empty.")

    return ParTrans(has_custom, log_idx, logit_idx, custom_idx, to_custom, from_custom)


def _pt_forward(theta_nat: jnp.ndarray, pt: ParTrans) -> jnp.ndarray:
    z = theta_nat
    if pt.log_idx:
        li = jnp.array(pt.log_idx, dtype=jnp.int32)
        cols = jnp.take(z, li, axis=-1)
        z = z.at[..., li].set(jnp.log(jnp.clip(cols, a_min=1e-12)))
    if pt.logit_idx:
        qi = jnp.array(pt.logit_idx, dtype=jnp.int32)
        cols = jnp.take(z, qi, axis=-1)
        u = jnp.clip(cols, 1e-12, 1.0 - 1e-12)
        logits = jnp.log(u) - jnp.log1p(-u)
        z = z.at[..., qi].set(logits)
    if pt.is_custom:
        ci = jnp.array(pt.custom_idx, dtype=jnp.int32)
        sub = jnp.take(z, ci, axis=-1)
        sub_t = pt.to_est_custom(sub)
        if sub_t.shape != sub.shape:
            raise ValueError("Custom to_est must return the same shape as its input subvector.")
        z = z.at[..., ci].set(sub_t)
    return z


def _pt_inverse(z_est: jnp.ndarray, pt: ParTrans) -> jnp.ndarray:
    x = z_est
    if pt.is_custom:
        ci = jnp.array(pt.custom_idx, dtype=jnp.int32)
        sub = jnp.take(x, ci, axis=-1)
        sub_x = pt.from_est_custom(sub)
        if sub_x.shape != sub.shape:
            raise ValueError("Custom from_est must return the same shape as its input subvector.")
        x = x.at[..., ci].set(sub_x)
    if pt.log_idx:
        li = jnp.array(pt.log_idx, dtype=jnp.int32)
        cols = jnp.take(x, li, axis=-1)
        x = x.at[..., li].set(jnp.exp(cols))
    if pt.logit_idx:
        qi = jnp.array(pt.logit_idx, dtype=jnp.int32)
        cols = jnp.take(x, qi, axis=-1)
        vals = jax.nn.sigmoid(cols)
        x = x.at[..., qi].set(vals)
    return x
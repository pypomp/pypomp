"""Systematic resampling as one CuTe kernel, one thread block per filter.

XLA computes :func:`pypomp.core.algorithms.helpers._resample` as four separate
passes over the weights (a cumulative sum, two cumulative maxima and a scatter).
Here a block streams its filter's weights through shared memory in tiles, each
thread taking a run of consecutive particles, so the cumulative weights come from
one block-wide scan per tile. Two passes read the weights: the first finds the
largest cumulative weight, the second writes the indices. The result has the same
guarantees: particles with zero weight are never selected, and slots past the last
count go to the last live particle. Cumulative weights are summed in another order
than XLA's, so an index can differ where a systematic point falls within rounding
error of a cumulative weight.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import cutlass  # pyright: ignore[reportMissingImports]
import cutlass.cute as cute  # pyright: ignore[reportMissingImports]
from cutlass.jax import cutlass_call  # pyright: ignore[reportMissingImports]

try:  # cutlass.utils.SmemAllocator warns that it is deprecated from CuTe DSL 4.8 on.
    from cutlass.memory import SmemAllocator  # pyright: ignore[reportMissingImports]
except ImportError:
    SmemAllocator = cutlass.utils.SmemAllocator

from .translate import _select

# Threads per block, particles per thread in a tile, and warps per block.
_T = 256
_V = 8
_NW = _T // 32
# The binary search over a tile needs its size to be a power of two.
_LOG_TV = (_T * _V).bit_length() - 1
assert 1 << _LOG_TV == _T * _V
# At or below this many particles per filter the kernel is faster at any batch size.
_KERNEL_MAX_J = 10_000

_Scalar = Any


def _f32(x: float) -> _Scalar:
    return cutlass.Float32(x)


def _imax(a: _Scalar, b: _Scalar) -> _Scalar:
    """Return the larger of two Int32 scalars."""
    return _select(a > b, a, b, np.int32)


def _warp_scan_sum(x: _Scalar, lane: _Scalar) -> tuple[_Scalar, _Scalar]:
    """Return the inclusive and exclusive sums of `x` over the warp's lanes."""
    for d in (1, 2, 4, 8, 16):
        y = cute.arch.shuffle_sync_up(x, d)
        x = x + _select(lane >= d, y, _f32(0.0), np.float32)
    excl = _select(lane == 0, _f32(0.0), cute.arch.shuffle_sync_up(x, 1), np.float32)
    return x, excl


def _warp_scan_max(x: _Scalar, lane: _Scalar) -> tuple[_Scalar, _Scalar]:
    """Return the inclusive and exclusive maxima of Int32 `x` over the warp, from 0."""
    for d in (1, 2, 4, 8, 16):
        y = cute.arch.shuffle_sync_up(x, d)
        x = _select(lane >= d, _imax(x, y), x, np.int32)
    excl = _select(
        lane == 0, cutlass.Int32(0), cute.arch.shuffle_sync_up(x, 1), np.int32
    )
    return x, excl


def _load_tile(NW: Any, buf: Any, b: _Scalar, base: _Scalar, t: _Scalar, J: _Scalar):
    """Copy exp(weights) of particles base.. into shared memory, coalesced; zero past J."""
    for v in range(_V):
        i = base + v * _T + t
        ok = i < J
        e = cute.math.exp(NW[b, _select(ok, i, J - 1, np.int32)])
        buf[v * _T + t] = _select(ok, e, _f32(0.0), np.float32)


def _thread_items(buf: Any, t: _Scalar) -> tuple[list, _Scalar]:
    """Return this thread's run of exp(weights) and their sum."""
    es = [buf[t * _V + v] for v in range(_V)]
    s = es[0]
    for e in es[1:]:
        s = s + e
    return es, s


@cute.kernel
def _kernel(NW: cute.Tensor, U: cute.Tensor, IDX: cute.Tensor, J: cutlass.Int32):
    t, _, _ = cute.arch.thread_idx()
    b, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    smem = SmemAllocator()
    buf = smem.allocate_tensor(cutlass.Float32, cute.make_layout(_T * _V), 16)
    w_f = smem.allocate_tensor(cutlass.Float32, cute.make_layout(_NW), 16)
    w_i = smem.allocate_tensor(cutlass.Int32, cute.make_layout(_NW), 16)
    w_j = smem.allocate_tensor(cutlass.Int32, cute.make_layout(_NW), 16)
    n_buf = smem.allocate_tensor(cutlass.Int32, cute.make_layout(_T * _V), 16)
    TV = _T * _V
    ntiles = (J + TV - 1) // TV

    # 1. The largest cumulative weight and the last live particle, computed as in 2.
    carry = cutlass.Float32(0.0)
    top = cutlass.Float32(0.0)
    last = cutlass.Int32(-1)
    for tile in range(ntiles):
        base = tile * TV
        _load_tile(NW, buf, b, base, t, J)
        cute.arch.sync_threads()
        es, s = _thread_items(buf, t)
        incl, excl = _warp_scan_sum(s, lane)
        if lane == 31:
            w_f[warp] = incl
        cute.arch.sync_threads()
        P = carry + excl
        tile_sum = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(_NW):
            P = P + _select(w < warp, w_f[w], _f32(0.0), np.float32)
            tile_sum = tile_sum + w_f[w]
        r = cutlass.Float32(0.0)
        for v in cutlass.range_constexpr(_V):
            r = r + es[v]
            live = es[v] > cutlass.Float32(0.0)
            top = _select(live, cute.math.max(top, P + r), top, np.float32)
            last = _select(live, base + t * _V + v, last, np.int32)
        carry = carry + tile_sum
        cute.arch.sync_threads()
    top = cute.arch.warp_reduction(top, cute.math.max)
    lmax = cute.arch.warp_reduction(last.to(cutlass.Float32), cute.math.max)
    if lane == 0:
        w_f[warp] = top
        w_i[warp] = lmax.to(cutlass.Int32)
    cute.arch.sync_threads()
    total = w_f[0]
    last_live = w_i[0]
    for w in cutlass.range_constexpr(1, _NW):
        total = cute.math.max(total, w_f[w])
        last_live = _imax(last_live, w_i[w])
    if last_live < 0:  # no live particle: XLA's argmax over all-False gives J - 1
        last_live = J - 1
    scale = J.to(cutlass.Float32) / total
    u = U[b]
    Jf = J.to(cutlass.Float32)
    cute.arch.sync_threads()

    # 2. Points at or below each cumulative weight, n_k, made nondecreasing across
    # threads and tiles; particle k fills the slots from the previous n up to n_k.
    carry = cutlass.Float32(0.0)
    n_carry = cutlass.Int32(0)
    for tile in range(ntiles):
        base = tile * TV
        _load_tile(NW, buf, b, base, t, J)
        cute.arch.sync_threads()
        es, s = _thread_items(buf, t)
        incl, excl = _warp_scan_sum(s, lane)
        if lane == 31:
            w_f[warp] = incl
        cute.arch.sync_threads()
        P = carry + excl
        tile_sum = cutlass.Float32(0.0)
        for w in cutlass.range_constexpr(_NW):
            P = P + _select(w < warp, w_f[w], _f32(0.0), np.float32)
            tile_sum = tile_sum + w_f[w]
        r = cutlass.Float32(0.0)
        ns = []
        m = cutlass.Int32(0)
        for v in cutlass.range_constexpr(_V):
            r = r + es[v]
            nf = cute.math.min(
                cute.math.max(cute.math.ceil((P + r) * scale - u), _f32(0.0)), Jf
            )
            n = _select(
                es[v] > cutlass.Float32(0.0),
                nf.to(cutlass.Int32),
                cutlass.Int32(0),
                np.int32,
            )
            ns.append(n)
            m = _imax(m, n)
        m_incl, m_excl = _warp_scan_max(m, lane)
        if lane == 31:
            w_j[warp] = m_incl
        cute.arch.sync_threads()
        prev = _imax(n_carry, m_excl)
        tile_max = n_carry
        for w in cutlass.range_constexpr(_NW):
            prev = _select(w < warp, _imax(prev, w_j[w]), prev, np.int32)
            tile_max = _imax(tile_max, w_j[w])
        for v in cutlass.range_constexpr(_V):
            prev = _imax(prev, ns[v])
            n_buf[t * _V + v] = prev
        cute.arch.sync_threads()
        # Slot m goes to the tile's first particle whose n exceeds m: a binary search
        # with the same steps in every thread, and coalesced writes.
        for slot in range(n_carry + t, tile_max, _T):
            lo = cutlass.Int32(0)
            for d in cutlass.range_constexpr(_LOG_TV):
                step = (_T * _V) >> (d + 1)
                lo = _select(n_buf[lo + step - 1] <= slot, lo + step, lo, np.int32)
            IDX[b, slot] = base + lo
        n_carry = tile_max
        carry = carry + tile_sum
        cute.arch.sync_threads()

    # 3. Slots past the last count, if any, go to the last live particle.
    for slot in range(n_carry + t, J, _T):
        IDX[b, slot] = last_live


@cute.jit
def _launch(stream, NW: cute.Tensor, U: cute.Tensor, IDX: cute.Tensor):
    B = cute.size(NW, mode=[0])
    J = cute.size(NW, mode=[1])
    # Without cutlass, pyright sees _kernel as the undecorated function.
    _kernel(NW, U, IDX, J).launch(  # pyright: ignore[reportAttributeAccessIssue]
        grid=[B, 1, 1], block=[_T, 1, 1], stream=stream
    )


def _use_kernel(B: int, J: int) -> bool:
    """Return whether the kernel beats XLA at resampling B filters of J particles.

    A block takes time proportional to J to resample its filter, while XLA spreads
    every pass over all B * J particles. With J above about 10,000 and fewer
    filters than half the GPU's SMs, XLA finishes first.
    """
    sms = getattr(jax.devices()[0], "core_count", 0) or 0
    return J <= _KERNEL_MAX_J or 2 * B >= sms


def _core_impl(nw: jax.Array, u: jax.Array) -> jax.Array:
    lead, J = nw.shape[:-1], nw.shape[-1]
    B = math.prod(lead)
    if not _use_kernel(B, J):
        from ..core.algorithms.helpers import _systematic_indices

        idx = jax.vmap(_systematic_indices)(nw.reshape(B, J), u.reshape(B))
        return idx.reshape(lead + (J,))
    idx = cutlass_call(
        _launch,
        output_shape_dtype=jax.ShapeDtypeStruct((B, J), jnp.int32),
    )(nw.reshape(B, J).astype(jnp.float32), u.reshape(B).astype(jnp.float32))
    return idx.reshape(lead + (J,))


@jax.custom_batching.custom_vmap
def _core(nw: jax.Array, u: jax.Array) -> jax.Array:
    return _core_impl(nw, u)


@_core.def_vmap
def _core_vmap(axis_size, in_batched, nw, u):
    nw, u = (
        a if bat else jnp.broadcast_to(a, (axis_size,) + a.shape)
        for a, bat in zip((nw, u), in_batched, strict=True)
    )
    return _core(nw, u), True


# Indices carry no derivative; this keeps JAX from differentiating the custom call.
@jax.custom_jvp
def _indices(nw: jax.Array, u: jax.Array) -> jax.Array:
    return _core(nw, u)


@_indices.defjvp
def _indices_jvp(primals, tangents) -> tuple[jax.Array, Any]:
    out = _indices(*primals)
    return out, np.zeros(out.shape, jax.dtypes.float0)


def resample(norm_weights: jax.Array, subkey: jax.Array) -> jax.Array:
    """Return systematic resampling indices, as ``_resample`` does, from the kernel."""
    if norm_weights.dtype != jnp.float32:  # the kernel is float32; x64 stays on XLA
        from ..core.algorithms.helpers import _resample

        return _resample(norm_weights, subkey)
    return _indices(norm_weights, jax.random.uniform(key=subkey))

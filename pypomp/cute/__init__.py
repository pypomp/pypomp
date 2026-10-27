"""CuTe backend for filtering on NVIDIA GPUs.

``Pomp.set_backend`` chooses it. The model's ``rproc`` is translated from its
jaxpr into a CUDA kernel that runs every Euler step of an observation interval, one
particle per thread, and evaluates ``dmeas`` too when that translates. The kernels
are written with CuTe DSL, the Python interface to
NVIDIA CUTLASS's CuTe library, which needs the optional ``nvidia-cutlass-dsl``
package (``pip install pypomp[cute]``).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from ..core.pomp import Pomp
    from ..functional.structs import PanelPompStruct, PompStruct

    _S = TypeVar("_S", PompStruct, PanelPompStruct)


@dataclass(frozen=True)
class KernelRProc:
    """An ``rprocess_interp`` run by the CuTe kernel, with the XLA one it replaces.

    The kernel's derivative has no derivative of its own, so code that needs
    forward-mode or second-order derivatives swaps `xla` back in (see
    :func:`_with_xla_rproc`).
    """

    kernel: Callable
    xla: Callable
    resample: Callable | None = None
    """A replacement for ``_resample`` that runs on the same device, if any."""
    with_dmeasure: Callable | None = None
    """``kernel`` that also returns each particle's measurement log-density, if the
    model's ``dmeasure`` translates; see :func:`_with_dmeasure`."""
    dmeasure: Callable | None = None
    """The XLA measurement density that ``with_dmeasure`` evaluates."""
    perturb_step: Callable | None = None
    """For per-particle parameters with a fused density: an interval of iterated
    filtering, from the ancestors through the parameter perturbation to the density;
    see :func:`_perturb_step`."""

    def __call__(self, *args: Any) -> Any:
        return self.kernel(*args)


def _with_xla_rproc(struct: _S) -> _S:
    """Return `struct` with any CuTe process model replaced by its XLA original."""
    pf, per = struct.rproc_pf, struct.rproc_per
    return struct._replace(
        rproc_pf=pf.xla if isinstance(pf, KernelRProc) else pf,
        rproc_per=per.xla if isinstance(per, KernelRProc) else per,
    )


def _with_xla_grads(kr: KernelRProc) -> KernelRProc:
    """Return `kr` with its derivatives taken through XLA instead of the kernel.

    Undifferentiated calls still run the kernel. Under reverse-mode differentiation
    the interval, and the density of ``with_dmeasure``, run on XLA, which then
    supplies the derivatives.
    """
    xla, dmeasure = kr.xla, kr.dmeasure

    def xla_dm(X_, theta_, keys, covars_extended, *rest):
        X, t_end = xla(X_, theta_, keys, covars_extended, *rest[:6])
        trans, y, t_obs = rest[5:]
        cov = None if covars_extended is None else covars_extended[t_end]
        assert dmeasure is not None
        return X, t_end, dmeasure(y, X, theta_, cov, t_obs, trans)

    dm = kr.with_dmeasure
    return replace(
        kr,
        kernel=_differentiated_by(kr.kernel, xla),
        with_dmeasure=None if dm is None else _differentiated_by(dm, xla_dm),
    )


def _differentiated_by(fast: Callable, slow: Callable) -> Callable:
    """Return `fast`, an ``rprocess_interp`` or ``with_dmeasure``, with `slow`'s
    reverse-mode derivatives in the particles and parameters and none in the rest.

    Arguments 8 and 9 (``accumvars_`` and ``should_trans``) are static.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    def zero(x: Any) -> Any:
        if jnp.issubdtype(jnp.result_type(x), jnp.inexact):
            return jnp.zeros_like(x)
        return np.zeros(jnp.shape(x), jax.dtypes.float0)

    def call(X_: Any, theta_: Any, keys: Any, *rest: Any) -> Any:
        # Key arrays have no cotangent type, so their data goes through instead.
        typed = jnp.issubdtype(keys.dtype, jax.dtypes.prng_key)
        impl = jax.random.key_impl(keys) if typed else None
        static, dyn = rest[5:7], rest[:5] + rest[7:]

        def args(X, th, kd, dyn):
            k = jax.random.wrap_key_data(kd, impl=impl) if typed else kd
            return (X, th, k, *dyn[:5], *static, *dyn[5:])

        @jax.custom_vjp
        def f(X, th, kd, dyn):
            return fast(*args(X, th, kd, dyn))

        def f_fwd(X, th, kd, dyn):
            out, pull = jax.vjp(lambda X, th: slow(*args(X, th, kd, dyn)), X, th)
            return out, (pull, kd, dyn)

        def f_bwd(res, g):
            pull, kd, dyn = res
            return (*pull(g), zero(kd), jax.tree.map(zero, dyn))

        f.defvjp(f_fwd, f_bwd)
        return f(X_, theta_, jax.random.key_data(keys) if typed else keys, dyn)

    return call


def _with_dmeasure(rproc: Callable, dmeasure: Callable) -> Callable | None:
    """Return `rproc`'s fused process-and-density kernel when it evaluates
    `dmeasure`, otherwise None."""
    if isinstance(rproc, KernelRProc) and rproc.dmeasure is dmeasure:
        return rproc.with_dmeasure
    return None


def _perturb_step(rproc: Callable, dmeasure: Callable) -> Callable | None:
    """Return `rproc`'s iterated filtering kernel when it evaluates `dmeasure`,
    otherwise None."""
    if isinstance(rproc, KernelRProc) and rproc.dmeasure is dmeasure:
        return rproc.perturb_step
    return None


def available() -> tuple[bool, str]:
    """Check whether the CuTe backend can run here.

    Returns
    -------
    tuple of (bool, str)
        Whether it can run, and if not, the reason.
    """
    try:
        import cutlass  # pyright: ignore[reportMissingImports] # noqa: F401
        import cutlass.cute  # pyright: ignore[reportMissingImports] # noqa: F401
        import cutlass.jax as cj  # pyright: ignore[reportMissingImports]
    except Exception as e:  # noqa: BLE001
        return False, f"nvidia-cutlass-dsl is not importable ({e})"
    if not cj.is_available():
        return False, "cutlass.jax reports the JAX integration unavailable"
    import jax

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        return False, "no GPU device"
    # CuTe DSL compiles for NVIDIA GPUs from Ampere (compute capability 8.0) on.
    cc = getattr(gpus[0], "compute_capability", None)
    if cc is None or int(str(cc).split(".")[0]) < 8:
        return False, f"GPU compute capability {cc}, below CuTe DSL's 8.0"
    return True, ""


def _build(
    model: Pomp, quiet: bool, rng: str, check: bool = True
) -> tuple[KernelRProc, KernelRProc] | None:
    """Return CuTe replacements for (rproc_pf, rproc_per), drawing with `rng`
    ("threefry" or "philox"), or None to keep XLA.

    Warns before returning None unless `quiet`, as the "auto" backend is. A kernel
    that doesn't
    match XLA, or fails to build, is a translator bug, so that always warns. Without
    `check`, the kernels are not run against XLA before they are returned.
    """

    def skip(msg: str, always: bool = False) -> None:
        if always or not quiet:
            warnings.warn(msg + " Using XLA.", stacklevel=4)

    ok, why = available()
    if not ok:
        return skip(f"CuTe backend unavailable: {why}.")
    if model.rproc.nstep is None:
        return skip(
            "CuTe backend needs the same number of Euler steps in every "
            "observation interval (nstep, or a dt that divides the intervals evenly)."
        )
    if check and model._theta is None:
        return skip("CuTe backend needs model.theta to check the kernel.")
    import jax

    if rng == "threefry":
        from .kernel import _is_threefry

        if not _is_threefry(jax.random.key(0)):
            return skip(
                "CuTe backend with rng='threefry' needs JAX's default PRNG "
                "implementation to be threefry2x32, with jax_threefry_partitionable "
                "on; rng='philox' takes any keys of two 32-bit words."
            )
    elif jax.random.key_data(jax.random.key(0)).shape != (2,):
        return skip(
            "CuTe backend needs a default PRNG implementation whose keys are "
            "two 32-bit words, such as threefry."
        )
    try:
        return _build_kernels(model, rng, check)
    except _Skip as e:
        return skip(e.msg, e.always)
    except Exception as e:  # noqa: BLE001
        return skip(
            f"Building the CuTe rproc failed ({type(e).__name__}: {e}).", always=True
        )


class _Skip(Exception):
    """Raised by :func:`_build_kernels` to keep XLA, with the warning to give."""

    def __init__(self, msg: str, always: bool = False) -> None:
        super().__init__(msg)
        self.msg, self.always = msg, always


def _build_kernels(
    model: Pomp, rng: str, check: bool = True
) -> tuple[KernelRProc, KernelRProc]:
    """Translate, build and check the kernels, as :func:`_build` describes.

    Operations the kernel can't run, and particle coupling visible in the jaxprs,
    are rejected even without `check`.
    """
    from . import kernel
    from .translate import _untranslatable
    from .verify import _check, _check_dmeas, _check_grads, _check_perturb_step

    traced = kernel._trace_step(model)
    step_j, prep_j, vec, step_vjp_j, prep_vjp_j = traced
    bad = set().union(
        *(_untranslatable(j) for j in (step_j, prep_j, step_vjp_j, prep_vjp_j))
    )
    if vec:
        bad |= kernel._particle_mixing(model)
    if bad:
        raise _Skip(
            f"rproc uses {sorted(bad)}, which the CuTe backend cannot translate."
        )
    pf = kernel._make_rprocess_interp(model, traced, False, rng, None)
    per = kernel._make_rprocess_interp(model, traced, True, rng, None)
    why = _check(model, pf, per, vec, rng) if check else None
    if why is not None:
        raise _Skip(f"CuTe rproc check failed: {why}.", always="differ" in why)
    why = _check_grads(model, per, rng) if check else None
    if why is not None:
        warnings.warn(
            f"CuTe rproc's gradients failed their check ({why}); differentiating "
            "through XLA instead.",
            stacklevel=4,
        )
    grads = why is None
    pf, per = _finish(pf, grads), _finish(per, grads)
    # The density joins the kernel only if it translates and, when checked, matches
    # XLA; otherwise it stays a separate XLA computation.
    try:
        dm_j = kernel._trace_dmeas(model)
    except Exception:  # noqa: BLE001
        dm_j = None
    if dm_j is None or _untranslatable(dm_j):
        return pf, per
    make = kernel._make_rprocess_interp
    pf_dm = _finish(make(model, traced, False, rng, dm_j), grads)
    per_dm = _finish(make(model, traced, True, rng, dm_j), grads)
    if not check:
        return pf_dm, per_dm
    try:
        why = _check_dmeas(model, pf_dm, per_dm, rng)
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
    if why is not None:
        warnings.warn(
            f"CuTe kernel with the measurement density failed its check ({why}); "
            "evaluating the density separately.",
            stacklevel=4,
        )
        return pf, per
    try:
        why = _check_perturb_step(model, per_dm, rng)
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
    if why is not None:
        warnings.warn(
            f"CuTe iterated filtering kernel failed its check ({why}); resampling "
            "and perturbing the parameters separately.",
            stacklevel=4,
        )
        per_dm = replace(per_dm, perturb_step=None)
    return pf_dm, per_dm


def _finish(kr: KernelRProc, grads: bool) -> KernelRProc:
    """Return `kr`, differentiated through XLA unless its `grads` passed."""
    return kr if grads else _with_xla_grads(kr)

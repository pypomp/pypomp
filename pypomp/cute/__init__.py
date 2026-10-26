"""CuTe backend for the process model on NVIDIA GPUs.

``Pomp.set_rproc_backend`` chooses it. The model's ``rproc`` is translated from its
jaxpr into a CUDA kernel that runs every Euler step of an observation interval, one
particle per thread. The kernel is written with CuTe DSL, the Python interface to
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

    if not any(d.platform == "gpu" for d in jax.devices()):
        return False, "no GPU device"
    return True, ""


def _build(
    model: Pomp, quiet: bool, rng: str
) -> tuple[KernelRProc, KernelRProc] | None:
    """Return CuTe replacements for (rproc_pf, rproc_per), drawing with `rng`
    ("threefry" or "philox"), or None to keep XLA.

    Warns before returning None unless `quiet`. The "auto" backend is quiet and also
    keeps XLA for a model with one Euler step per observation. A kernel that doesn't
    match XLA, or fails to build, is a translator bug, so that always warns.
    """

    def skip(msg: str, always: bool = False) -> None:
        if always or not quiet:
            warnings.warn(msg + " Using XLA.", stacklevel=4)

    ok, why = available()
    if not ok:
        return skip(f"CuTe rproc backend unavailable: {why}.")
    if model.rproc.nstep is None:
        return skip(
            "CuTe rproc backend needs the same number of Euler steps in every "
            "observation interval (nstep, or a dt that divides the intervals evenly)."
        )
    if quiet and int(model.rproc.nstep) == 1:
        return None
    if model._theta is None:
        return skip("CuTe rproc backend needs model.theta to check the kernel.")
    import jax

    if rng == "threefry":
        from .kernel import _is_threefry

        if not _is_threefry(jax.random.key(0)):
            return skip(
                "CuTe rproc backend with rng='threefry' needs JAX's default PRNG "
                "implementation to be threefry2x32, with jax_threefry_partitionable "
                "on; rng='philox' takes any keys of two 32-bit words."
            )
    elif jax.random.key_data(jax.random.key(0)).shape != (2,):
        return skip(
            "CuTe rproc backend needs a default PRNG implementation whose keys are "
            "two 32-bit words, such as threefry."
        )
    try:
        return _build_kernels(model, rng)
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


def _build_kernels(model: Pomp, rng: str) -> tuple[KernelRProc, KernelRProc]:
    """Translate, build and check the kernels, as :func:`_build` describes."""
    from . import kernel
    from .translate import _untranslatable
    from .verify import _check, _check_dmeas, _check_perturb_step

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
    why = _check(model, pf, per, vec, rng)
    if why is not None:
        raise _Skip(f"CuTe rproc check failed: {why}.", always="differ" in why)
    # The density joins the kernel only if it translates and matches XLA; otherwise
    # it stays a separate XLA computation.
    try:
        dm_j = kernel._trace_dmeas(model)
    except Exception:  # noqa: BLE001
        dm_j = None
    if dm_j is None or _untranslatable(dm_j):
        return pf, per
    pf_dm = kernel._make_rprocess_interp(model, traced, False, rng, dm_j)
    per_dm = kernel._make_rprocess_interp(model, traced, True, rng, dm_j)
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

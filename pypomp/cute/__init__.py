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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.pomp import Pomp


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
    model: Pomp, accumvars: tuple[int, ...] | None, quiet: bool
) -> tuple[Callable, Callable] | None:
    """Return CuTe replacements for (rproc_pf, rproc_per), or None to keep XLA.

    Warns before returning None unless `quiet`. The "auto" backend is quiet and also
    keeps XLA for a model with one Euler step per observation.
    """

    def skip(msg: str) -> None:
        if not quiet:
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
    from . import kernel
    from .translate import _untranslatable

    traced = kernel._trace_step(model)
    step_j, prep_j, vec, step_vjp_j, prep_vjp_j = traced
    bad = set().union(
        *(_untranslatable(j) for j in (step_j, prep_j, step_vjp_j, prep_vjp_j))
    )
    if vec:
        bad |= kernel._particle_mixing(model)
    if bad:
        return skip(
            f"rproc uses {sorted(bad)}, which the CuTe backend cannot translate."
        )
    return (
        kernel._make_rprocess_interp(model, traced, accumvars, False),
        kernel._make_rprocess_interp(model, traced, accumvars, True),
    )

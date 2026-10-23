"""CuTe backend for the process model, on NVIDIA GPUs (see ``Pomp.set_rproc_backend``).

The model's ``rproc`` is translated from its jaxpr into a CuTe DSL kernel that runs
every Euler step of an observation interval for one particle per thread. It needs the
optional ``nvidia-cutlass-dsl`` package and a GPU; without them, or if the ``rproc``
uses an operation the translator does not handle, pypomp warns and keeps the XLA path.
"""

from __future__ import annotations

import warnings


def available() -> tuple[bool, str]:
    """Whether the CuTe backend can run here, and if not, why."""
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        import cutlass.jax as cj
    except Exception as e:  # noqa: BLE001
        return False, f"nvidia-cutlass-dsl is not importable ({e})"
    if not cj.is_available():
        return False, "cutlass.jax reports the JAX integration unavailable"
    import jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        return False, "no GPU device"
    return True, ""


def build(model, accumvars, rng: str = "philox", quiet: bool = False):
    """(rproc_pf, rproc_per) CuTe replacements for `model`, or None (with a warning
    unless `quiet`, which is the "auto" backend; that also skips one-step models)."""

    def skip(msg):
        if not quiet:
            warnings.warn(msg + " Using XLA.", stacklevel=4)
        return None

    ok, why = available()
    if not ok:
        return skip(f"CuTe rproc backend unavailable: {why}.")
    if model.rproc.nstep is None:
        return skip("CuTe rproc backend needs the same number of Euler steps in every observation "
                    "interval (nstep, or a dt that divides the intervals evenly).")
    if quiet and int(model.rproc.nstep) == 1:
        return None
    from . import kernel

    step_j, prep_j, _, step_vjp_j, prep_vjp_j = kernel.trace_step(model)
    bad = (kernel.unsupported(step_j) | kernel.unsupported(prep_j) | kernel.unsupported(step_vjp_j)
           | kernel.unsupported(prep_vjp_j))
    if bad:
        return skip(f"rproc uses {sorted(bad)}, which the CuTe backend does not translate.")
    return (kernel.make_rprocess_interp(model, accumvars, per_particle=False, rng=rng),
            kernel.make_rprocess_interp(model, accumvars, per_particle=True, rng=rng))

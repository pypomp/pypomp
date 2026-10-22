"""CuTe backend for the process model, on NVIDIA GPUs (opt in with ``Pomp.set_rproc_backend``).

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


def build(model, accumvars, rng: str = "philox"):
    """(rproc_pf, rproc_per) CuTe replacements for `model`, or None after a warning."""
    ok, why = available()
    if not ok:
        warnings.warn(f"CuTe rproc backend unavailable: {why}. Using XLA.", stacklevel=3)
        return None
    if model.rproc.nstep is None:
        warnings.warn("CuTe rproc backend needs the same number of Euler steps in every observation "
                      "interval (nstep, or a dt that divides the intervals evenly). Using XLA.", stacklevel=3)
        return None
    from . import kernel

    step_j, prep_j, _, step_vjp_j, prep_vjp_j = kernel.trace_step(model)
    bad = (kernel.unsupported(step_j) | kernel.unsupported(prep_j) | kernel.unsupported(step_vjp_j)
           | kernel.unsupported(prep_vjp_j))
    if bad:
        warnings.warn(f"rproc uses {sorted(bad)}, which the CuTe backend does not translate. Using XLA.",
                      stacklevel=3)
        return None
    return (kernel.make_rprocess_interp(model, accumvars, per_particle=False, rng=rng),
            kernel.make_rprocess_interp(model, accumvars, per_particle=True, rng=rng))

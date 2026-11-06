"""What the CuTe backend runs for a model, and why the rest stays on XLA."""

from __future__ import annotations

from dataclasses import dataclass

# The parts of filtering the backend can move to the GPU, in the report's order.
PARTS = ("rproc", "gradients", "dmeasure", "perturbation", "resampling")


@dataclass(frozen=True)
class Problem:
    """A use in the model's code that the CuTe translator can't run."""

    what: str
    """The primitive or use, such as ``"while loop with a runtime trip count"``."""
    where: tuple[str, ...] = ()
    """The lines of the model's code that make it, as ``"file:line (function)"``,
    each followed by the lines that called it when it is in a pypomp helper."""

    def __str__(self) -> str:
        return "\n".join([self.what, *(f"  at {w}" for w in self.where)])


@dataclass(frozen=True)
class Part:
    """How one part of filtering runs."""

    name: str
    """One of ``"rproc"``, ``"gradients"``, ``"dmeasure"``, ``"perturbation"``
    (iterated filtering's parameter perturbation) and ``"resampling"``."""
    cute: bool
    """Whether it runs on the CuTe backend."""
    translates: bool | None = None
    """Whether the translator accepts the model's code for it, or None where
    that doesn't apply."""
    note: str = ""
    """Why it stays on XLA, or when it still does."""
    problems: tuple[Problem, ...] = ()
    """What in the model's code keeps it from translating."""


@dataclass(frozen=True)
class BackendReport:
    """What runs on the CuTe backend for a model, from
    :meth:`pypomp.Pomp.backend_report`.

    Index it by part name, e.g. ``report["dmeasure"].cute``; ``print`` it for a
    summary. Where CuTe can't run, such as on a machine without a GPU, it still says
    which parts the translator accepts.
    """

    backend: str
    """The backend asked for: ``"auto"``, ``"cute"`` or ``"xla"``."""
    rng: str
    """The random numbers the CuTe kernel draws."""
    available: bool
    """Whether CuTe can run here (see :func:`pypomp.cute.available`)."""
    reason: str
    """Why CuTe wasn't tried, such as ``backend='xla'`` or no GPU, or ``""``."""
    checked: bool
    """Whether the kernels were checked numerically against XLA."""
    parts: tuple[Part, ...]

    def __getitem__(self, name: str) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def translates(self) -> bool:
        """Whether the process model translates, which CuTe needs to run at all."""
        return bool(self["rproc"].translates)

    def __str__(self) -> str:
        lines = [f"Backend {self.backend!r}, CuTe rng {self.rng!r}."]
        built = not self.reason
        if not built:
            lines.append(
                f"Everything runs on XLA ({self.reason}). With CuTe, the translator "
                "would see:"
            )
        width = max(len(p.name) for p in self.parts)
        for p in self.parts:
            if built:
                status = "CuTe" if p.cute else "XLA "
            elif p.translates is None:
                continue
            else:
                status = "translates" if p.translates else "doesn't translate"
            note = p.note if built else ""
            lines.append(f"  {p.name:<{width}}  {status}  {note}".rstrip())
            for prob in p.problems:
                lines.extend("      " + s for s in str(prob).split("\n"))
        if built and self["rproc"].cute:
            other = " or keys of another generator" if self.rng == "threefry" else ""
            lines.append(
                f"Calls with float64 particles{other} still run on XLA, as does "
                "train with an optimizer that needs Hessians."
            )
            if not self.checked:
                lines.append("The kernels were not checked against XLA (check=False).")
        elif not built and self.translates:
            lines.append("Whether the kernels match XLA is checked on a GPU.")
        return "\n".join(lines)

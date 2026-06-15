from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Optimizer:
    """Base class for all pypomp optimizers.

    Parameters
    ----------
    clip_norm : float, optional
        Maximum norm threshold for gradient clipping. Gradients are clipped to
        [-clip_norm, clip_norm] if provided. Defaults to None (no clipping).
    scale : bool, default False
        Whether to normalize the update search direction to unit length
        before applying the learning rate.
    ls : bool, default False
        Whether to enable the Armijo backtracking line search algorithm to
        determine optimal step size.
    c : float, default 0.1
        The Armijo condition constant for line search, controlling how much
        the objective must decrease to accept a step size. Only used when ls=True.
    max_ls_itn : int, default 10
        Maximum number of backtracking iterations per line search step.
        Only used when ls=True.
    """

    clip_norm: Optional[float] = None
    scale: bool = False
    ls: bool = False
    c: float = 0.1
    max_ls_itn: int = 10

    def __str__(self) -> str:
        from dataclasses import fields

        field_strs = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, float):
                field_strs.append(f"{f.name}={val:.4g}")
            else:
                field_strs.append(f"{f.name}={val}")
        return f"{self.__class__.__name__}({', '.join(field_strs)})"


@dataclass(frozen=True)
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    pass


@dataclass(frozen=True)
class Adam(Optimizer):
    """Adam optimizer.

    Parameters
    ----------
    beta1 : float, default 0.9
        The exponential decay rate for the first moment estimates (momentum).
    beta2 : float, default 0.999
        The exponential decay rate for the second moment estimates (variance).
    epsilon : float, default 1e-8
        A small constant for numerical stability.
    """

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


@dataclass(frozen=True)
class FullMatrixAdam(Optimizer):
    """Full-Matrix Adam optimizer.

    Parameters
    ----------
    beta1 : float, default 0.9
        The exponential decay rate for the first moment estimates.
    beta2 : float, default 0.999
        The exponential decay rate for the second moment estimates.
    epsilon : float, default 1e-4
        A small constant for numerical stability.
    """

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-4


@dataclass(frozen=True)
class BFGS(Optimizer):
    """Quasi-Newton BFGS optimizer."""

    pass


@dataclass(frozen=True)
class Newton(Optimizer):
    """Classic Second-Order Newton-Raphson optimizer."""

    pass


@dataclass(frozen=True)
class WeightedNewton(Optimizer):
    """Weighted Newton optimizer with decaying history."""

    pass

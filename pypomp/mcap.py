"""
This module implements Monte Carlo-adjusted profile (MCAP) for POMP models.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Sequence, Callable, Any, List
import numpy as np

# TODO list statsmodels as install dependency in package description
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
from scipy.stats import chi2

__all__ = ["MCAPResult", "mcap"]

# MCAP result container
@dataclass
class MCAPResult:
    level: float
    mle: float
    ci: Tuple[Optional[float], Optional[float]]
    delta: float
    se_stat: float
    se_mc: float
    se_total: float
    fit: Dict[str, np.ndarray]
    quadratic_max: float
    quadratic_coef: Dict[str, float]
    vcov: np.ndarray 

def _tricube_weights(dist: np.ndarray, cutoff: float) -> np.ndarray:
    """
    Acquires the tricube weights on (0, cutoff]
    """
    w = np.zeros_like(dist, dtype=float)
    if cutoff <= 0:
        return w
    u = np.clip(dist / cutoff, 0.0, 1.0)
    inside = u < 1.0
    w[inside] = (1.0 - u[inside] ** 3) ** 3
    return w

def mcap(
    logLik: np.ndarray,
    parameter: np.ndarray,
    level: float = 0.95,
    span: float = 0.75,
    Ngrid: int = 1000,
    lowess_it: int = 3,
) -> MCAPResult:
    
    phi = np.asarray(parameter, dtype = float).ravel()
    ll = np.asarray(logLik, dtype = float).ravel()
    if phi.size != ll.size:
        raise ValueError("`parameter` and `logLik` must have the same length.")
    if phi.size < 3:
        raise ValueError("Need at least 3 points to compute a quadratic fit.")

    # Sort by parameter for stable smoothing
    order = np.argsort(phi)
    phi = phi[order]
    ll = ll[order]

    # Build grid for smoothed curve & CI search
    p_grid = np.linspace(phi.min(), phi.max(), int(Ngrid))

    # LOWESS smoothing evaluated on grid
    smoothed = sm_lowess(
        endog = ll, exog = phi, frac = span, it = lowess_it, xvals = p_grid, is_sorted = True
    )

    # Smoothed argmax
    smooth_arg_max = p_grid[int(np.nanargmax(smoothed))]
"""
This module implements Monte Carlo-adjusted profile (MCAP) for POMP models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Callable, cast

import numpy as np
from scipy.stats import chi2

from .util import logmeanexp
from .pfilter import _vmapped_pfilter_internal2

# TODO list loess as install dependency in package description
from loess.loess_1d import loess_1d

__all__ = ["MCAPResult", "mcap", "mcap_profile"]


def _qchisq(level: float, df: int = 1) -> float:
    return float(chi2.ppf(level, df))


def _loess_smooth_1d(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    *,
    span: float = 0.75,
    degree: int = 2,
) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)

    # drop NaNs
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    if x.size == 0:
        return np.full_like(grid, np.nan, dtype=float)

    # loess_1d return: (xout, yout, wout)
    res = loess_1d(
        x,
        y,
        xnew=grid,
        degree=int(degree),
        frac=float(span),
    )

    if len(res) == 3:
        _, y_sm, _ = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], res)
        return y_sm
    else:
        # if frac == 0 in loess_1d
        y_raw, _ = cast(Tuple[np.ndarray, np.ndarray], res)
        return np.interp(grid, x, y_raw)


def _fit_local_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center: float,
    span: float,
) -> Tuple[float, float, float, np.ndarray]:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dist = np.abs(x - center)

    m = max(3, int(np.floor(span * len(x))))
    if m >= len(x):
        included = np.ones_like(x, dtype=bool)
    else:
        kth = np.partition(dist, m - 1)[m - 1]
        included = dist <= kth

    # tricube weights on chosen window
    w = np.zeros_like(x, dtype=float)
    if np.any(included):
        maxdist = dist[included].max()
        if maxdist > 0.0:
            w[included] = (1.0 - (dist[included] / maxdist) ** 3) ** 3
        else:
            w[included] = 1.0

    # uncentered
    X = np.column_stack([
        np.ones_like(x),
        -(x ** 2),
        x            
    ])


    # weighted least squares
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    c, a, b = map(float, coef)

    # residual based variance estimate
    yhat = X @ coef
    resid = (y - yhat) * sw
    df = int(np.sum(w > 0) - X.shape[1])
    if df > 0:
        s2 = float(np.sum(resid ** 2) / df)
    else:
        s2 = 0.0
    XtWX = Xw.T @ Xw
    cov_full = s2 * np.linalg.pinv(XtWX)

    vc_ab = cov_full[1:3, 1:3]

    return a, b, c, vc_ab


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


def mcap_profile(
    parameter: Sequence[float],
    loglik: Sequence[float],
    *,
    level: float = 0.95,
    span: float = 0.75,
    n_grid: int = 1000,
    loess_degree: int = 2,
) -> MCAPResult:

    x = np.asarray(parameter, dtype=float)
    y = np.asarray(loglik, dtype=float)

    # grid over observed parameter range
    grid = np.linspace(float(np.min(x)), float(np.max(x)), int(n_grid))

    # smooth noisy profile
    y_sm = _loess_smooth_1d(x, y, grid=grid, span=span, degree=loess_degree)

    return MCAPResult(
        level=level,

    )
"""
This module implements Monte Carlo-adjusted profile (MCAP) for POMP models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

# TODO list loess as install dependency in package description
from loess.loess_1d import loess_1d

FloatArray = npt.NDArray[np.floating[Any]]

__all__ = ["MCAPResult", "mcap", "mcap_profile"]


def _qchisq(level: float, df: int = 1) -> float:
    return float(chi2.ppf(level, df))

def _loess_smooth_1d(
    x: FloatArray,
    y: FloatArray,
    grid: FloatArray,
    *,
    span: float = 0.75,
    degree: int = 2,
) -> FloatArray:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)

    # drop NaNs
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    if x.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    
    # normalize predictor
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    scale = xmax - xmin

    if scale <= 0.0 or not np.isfinite(scale):
        # degenerate predictor: return flat line at mean(y)
        return np.full_like(grid, float(np.mean(y)), dtype=float)

    x_norm = (x - xmin) / scale
    grid_norm = (grid - xmin) / scale

    # neutralize robustness by making all biweights 1
    HUGE_SIGY = 1e9
    sigy = np.full_like(y, HUGE_SIGY, dtype=float)

    try:
        res = loess_1d(
            x_norm, y,
            xnew=grid_norm,
            degree=int(degree),
            frac=float(span),
            sigy=sigy,
        )
    except Exception:
        coeff = np.polyfit(x_norm, y, deg=min(degree, 2))
        y_sm = np.polyval(coeff, grid_norm)
        return y_sm.astype(float, copy=False)

    if len(res) == 3:
        _, y_sm, _ = res
        return y_sm.astype(float, copy=False)
    else:
        # if frac == 0 in loess_1d
        y_raw, _ = res
        return np.interp(grid_norm, x_norm, y_raw).astype(float, copy=False)


def _fit_local_quadratic(
    x: FloatArray,
    y: FloatArray,
    *,
    center: float,
    span: float,
) -> Tuple[float, float, float, FloatArray]:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dist = np.abs(x - center)

    m = max(3, int(np.floor(span * len(x))))
    if m >= len(x):
        included = np.ones_like(x, dtype=bool)
    else:
        kth = np.partition(dist, m - 1)[m - 1]
        included = dist < kth

    # tricube weights on chosen window
    w = np.zeros_like(x, dtype=float)
    if np.any(included):
        maxdist = dist[included].max()
        if maxdist > 0.0:
            w[included] = (1.0 - (dist[included] / maxdist) ** 3) ** 3
        else:
            w[included] = 1.0

    # uncentered
    X = np.column_stack([np.ones_like(x), -(x ** 2), x])

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
    fit: Dict[str, FloatArray]          
    quadratic_max: float
    quadratic_coef: Dict[str, float]
    vcov: FloatArray         


def mcap_profile(
    parameter: npt.ArrayLike,
    loglik: npt.ArrayLike,
    *,
    level: float = 0.95,
    span: float = 0.75,
    n_grid: int = 1000,
    loess_degree: int = 2,
) -> MCAPResult:
    x: FloatArray = np.asarray(parameter, dtype=float)
    y: FloatArray = np.asarray(loglik, dtype=float)

    # grid over observed parameter range
    grid = np.linspace(float(np.min(x)), float(np.max(x)), int(n_grid))

    # smooth noisy profile
    y_sm = _loess_smooth_1d(x, y, grid=grid, span=span, degree=loess_degree)

    # MLE = argmax of smoothed profile
    i_max = int(np.nanargmax(y_sm))
    mle = float(grid[i_max])

    # local quadratic at smoothed MLE with raw data
    a, b, c, vc_ab = _fit_local_quadratic(x, y, center=mle, span=span)

    # SE decomposition
    se_stat2 = 1.0 / (2.0 * a)

    # Monte Carlo variance from vcov(a, b)
    var_a = float(vc_ab[0, 0])
    var_b = float(vc_ab[1, 1])
    cov_ab = float(vc_ab[0, 1])

    se_mc2 = (
        1.0 / (4.0 * a * a)
         * (var_b - 2.0 * (b / a) * cov_ab + (b * b / (a * a)) * var_a)
    )

    se_tot2 = se_stat2 + se_mc2

    # MC-adjusted cutoff
    q = _qchisq(level, df=1)
    delta = float(q * (a * se_mc2 + 0.5))

    # CI from smoothed profile
    diff = float(np.nanmax(y_sm)) - y_sm
    inside = diff < delta
    if not np.any(inside):
        ci = (None, None)
    else:
        idx = np.where(inside)[0]
        ci = (float(grid[idx.min()]), float(grid[idx.max()]))

    # quadratic curve on grid
    quad = c - a * (grid ** 2) + b * grid

    if a > 0.0:
        quad_max = b / (2.0 * a)
    else:
    # fallback to smoothed MLE if curvature is non-positive
        quad_max = mle

    return MCAPResult(
        level = level,
        mle = mle,
        ci = ci,
        delta = delta,
        se_stat = float(np.sqrt(se_stat2)),
        se_mc = float(np.sqrt(se_mc2)),
        se_total = float(np.sqrt(se_stat2 + se_mc2)),
        fit = {
            "parameter": grid,
            "smoothed": y_sm,
            "quadratic": quad,
        },
        quadratic_max = float(quad_max),
        quadratic_coef = {"a": float(a), "b": float(b), "c": float(c)},
        vcov = vc_ab,
    )

def mcap(*args, **kwargs) -> MCAPResult:
    return mcap_profile(*args, **kwargs)
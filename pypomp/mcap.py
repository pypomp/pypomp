"""
This module implements Monte Carlo-adjusted profile (MCAP) for POMP models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

FloatArray = npt.NDArray[np.floating[Any]]

__all__ = ["MCAPResult", "mcap"]


def _qchisq(level: float, df: int = 1) -> float:
    return float(chi2.ppf(level, df))


def _loess_smooth_1d(
    x: FloatArray,
    y: FloatArray,
    grid: FloatArray,
    *,
    span: float = 0.75,
    degree: int = 2,
    max_iter: int = 10,
) -> FloatArray:
    """Perform 1D LOESS smoothing on a grid following Cleveland (1979).

    Parameters
    ----------
    x : FloatArray
        Predictor values.
    y : FloatArray
        Response values.
    grid : FloatArray
        Evaluation points at which to compute the smoothed values.
    span : float, optional
        Fraction of points to include in the local neighborhood. Defaults to ``0.75``.
    degree : int, optional
        Degree of the local polynomial (1 for linear, 2 for quadratic). Defaults to ``2``.
    max_iter : int, optional
        Maximum number of robust bisquare iterations. Defaults to ``10``.

    Returns
    -------
    FloatArray
        Smoothed response values evaluated at ``grid``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    scale = xmax - xmin

    if scale <= 0.0 or not np.isfinite(scale):
        # degenerate predictor: return flat line at mean(y)
        return np.full_like(grid, float(np.mean(y)), dtype=float)

    n = len(x)
    npoints = int(np.ceil(span * n))
    npoints = max(degree + 1, min(n, npoints))

    deg = int(degree)
    deg_powers = np.arange(deg + 1)
    y_sm = np.empty_like(grid, dtype=float)

    for j, xj in enumerate(grid):
        dist = np.abs(x - xj)
        w_idx = np.argsort(dist)[:npoints]
        xw = x[w_idx]
        yw = y[w_idx]
        dw = dist[w_idx]

        max_d = dw[-1]
        if max_d > 0.0:
            dist_weights = (1.0 - (dw / max_d) ** 3) ** 3
        else:
            dist_weights = np.ones_like(dw)

        A = xw[:, None] ** deg_powers
        sqw = np.sqrt(dist_weights)
        coef, _, _, _ = np.linalg.lstsq(A * sqw[:, None], yw * sqw, rcond=None)
        yfit = A @ coef

        bad = None
        for _ in range(max_iter):
            aerr = np.abs(yfit - yw)
            mad = float(np.median(aerr))
            if mad == 0.0:
                break
            uu = (aerr / (6.0 * mad)) ** 2
            uu = np.clip(uu, 0.0, 1.0)
            biweights = (1.0 - uu) ** 2
            tot_weights = dist_weights * biweights

            if np.all(tot_weights == 0.0):
                break

            sqw_tot = np.sqrt(tot_weights)
            try:
                coef, _, _, _ = np.linalg.lstsq(
                    A * sqw_tot[:, None], yw * sqw_tot, rcond=None
                )
                yfit = A @ coef
            except np.linalg.LinAlgError:
                break

            bad_old = bad
            bad = biweights < 0.34
            if bad_old is not None and np.array_equal(bad_old, bad):
                break

        a_xj = xj**deg_powers
        y_sm[j] = float(a_xj @ coef)

    return y_sm


def _fit_local_quadratic(
    x: FloatArray,
    y: FloatArray,
    *,
    center: float,
    span: float,
) -> tuple[float, float, float, FloatArray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dist = np.abs(x - center)

    m = int(np.trunc(span * len(x)))
    m = max(3, min(m, len(x)))

    # always compute kth distance
    kth = np.sort(dist)[m - 1]
    included = dist < kth

    if np.count_nonzero(included) < 3:
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
    X = np.column_stack([np.ones_like(x), -(x**2), x])

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
        s2 = float(np.sum(resid**2) / df)
    else:
        s2 = 0.0
    XtWX = Xw.T @ Xw

    try:
        cov_full = s2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        # if singular
        cov_full = s2 * np.linalg.pinv(XtWX)

    vc_ab = cov_full[1:3, 1:3]
    return a, b, c, vc_ab


# MCAP result container
@dataclass
class MCAPResult:
    """Results of a Monte Carlo adjusted profile (MCAP) analysis."""

    level: float
    """The confidence level of the profile likelihood confidence interval."""

    mle: float
    """The maximum likelihood estimate of the focal parameter, taken as the argmax of the smoothed profile."""

    ci: tuple[float | None, float | None]
    """The profile likelihood confidence interval (lower, upper)."""

    delta: float
    """The log-likelihood threshold used to define the confidence interval, relative to the maximum."""

    se_stat: float
    """The standard error due to statistical uncertainty (sampling variance)."""

    se_mc: float
    """The standard error due to Monte Carlo noise in the likelihood estimates."""

    se_total: float
    """The total standard error, calculated as the root sum of squares of se_stat and se_mc."""

    fit: dict[str, FloatArray]
    """A dictionary containing the grid of parameters ('parameter'), the smoothed log-likelihood values ('smoothed'), and the local quadratic fit values ('quadratic')."""

    quadratic_max: float
    """The parameter value that maximizes the local quadratic fit."""

    quadratic_coef: dict[str, float]
    """The coefficients of the local quadratic fit: c - ax^2 + bx."""

    vcov: FloatArray
    """The variance-covariance matrix of the quadratic coefficients a and b."""


def mcap(
    parameter: npt.ArrayLike,
    loglik: npt.ArrayLike,
    *,
    level: float = 0.95,
    span: float = 0.75,
    n_grid: int = 1000,
    loess_degree: int = 2,
) -> MCAPResult:
    """Compute Monte Carlo-adjusted profile (MCAP) confidence intervals.

    Constructs a profile likelihood confidence interval accommodating both
    Monte Carlo noise in the profile and statistical uncertainty in the
    likelihood function (Ionides et al. 2017 [1]_).

    Parameters
    ----------
    parameter : array-like
        Parameter values at which log-likelihoods were evaluated.
    loglik : array-like
        Log-likelihood values corresponding to ``parameter``.
    level : float, optional
        Confidence level for the interval.  Defaults to ``0.95``.
    span : float, optional
        Span parameter for the LOESS smoother.  Defaults to ``0.75``.
    n_grid : int, optional
        Number of grid points for evaluating the smoothed log-likelihood.
        Defaults to ``1000``.
    loess_degree : int, optional
        Polynomial degree for the LOESS smoother.  Defaults to ``2``.

    Returns
    -------
    MCAPResult
        Object containing the computed confidence interval and SE decomposition.

    References
    ----------
    .. [1] Ionides, Edward L., Carles Bretó, Joonha Park, R. A. Smith, and Aaron A. King.
       "Monte Carlo profile confidence intervals for dynamic systems."
       *Journal of The Royal Society Interface* 14, no. 132 (2017): 20170126.
       https://doi.org/10.1098/rsif.2017.0126.
    """
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
        1.0
        / (4.0 * a * a)
        * (var_b - 2.0 * (b / a) * cov_ab + (b * b / (a * a)) * var_a)
    )

    # se_tot2 = se_stat2 + se_mc2

    # MC-adjusted cutoff
    q = _qchisq(level, df=1)
    delta = float(q * (a * se_mc2 + 0.5))

    # CI from smoothed profile
    diff = float(np.nanmax(y_sm)) - y_sm
    inside = diff < delta
    ci: tuple[float | None, float | None]
    if not np.any(inside):
        ci = (None, None)
    else:
        idx = np.where(inside)[0]
        ci = (float(grid[idx.min()]), float(grid[idx.max()]))

    # quadratic curve on grid
    quad = c - a * (grid**2) + b * grid

    if a > 0.0:
        quad_max = b / (2.0 * a)
    else:
        # fallback to smoothed MLE if curvature is non-positive
        quad_max = mle

    return MCAPResult(
        level=level,
        mle=mle,
        ci=ci,
        delta=delta,
        se_stat=float(np.sqrt(se_stat2)),
        se_mc=float(np.sqrt(se_mc2)),
        se_total=float(np.sqrt(se_stat2 + se_mc2)),
        fit={
            "parameter": grid,
            "smoothed": y_sm,
            "quadratic": quad,
        },
        quadratic_max=float(quad_max),
        quadratic_coef={"a": float(a), "b": float(b), "c": float(c)},
        vcov=vc_ab,
    )

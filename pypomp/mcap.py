"""
This module implements Monte Carlo-adjusted profile (MCAP) for POMP models.
"""

from __future__ import annotations

import warnings
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
    npoints = int(np.floor(span * n + 1e-5))  # R's lowesd rule; eps matters at integer span*n
    npoints = max(degree + 1, min(n, npoints))

    deg = int(degree)
    deg_powers = np.arange(deg + 1)
    y_sm = np.empty_like(grid, dtype=float)

    rob = np.ones(n)
    max_iter = max(int(max_iter), 0)
    for it in range(max_iter + 1):
        pts = x if it < max_iter else grid
        y_sm = np.empty(len(pts), dtype=float)
        for j, xj in enumerate(pts):
            dist = np.abs(x - xj)
            w_idx = np.argsort(dist)[:npoints]
            xw, yw, dw = x[w_idx], y[w_idx], dist[w_idx]
            max_d = dw[-1]
            dist_weights = ((1.0 - (dw / max_d) ** 3) ** 3 if max_d > 0.0
                            else np.ones_like(dw))
            A = xw[:, None] ** deg_powers
            sqw = np.sqrt(dist_weights * rob[w_idx])
            if not np.any(sqw > 0.0):
                sqw = np.sqrt(dist_weights)
            coef, _, _, _ = np.linalg.lstsq(A * sqw[:, None], yw * sqw, rcond=None)
            y_sm[j] = float((xj**deg_powers) @ coef)
        if it == max_iter:
            break
        aerr = np.abs(y - y_sm)
        cmad = 6.0 * float(np.median(aerr))
        rob = np.where(aerr >= cmad, 0.0, (1.0 - (aerr / cmad) ** 2) ** 2) if cmad > 0.0 else rob

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
    *,
    parameter: npt.ArrayLike,
    loglik: npt.ArrayLike,
    level: float = 0.95,
    span: float = 0.75,
    n_grid: int = 1000,
    loess_degree: int = 2,
    loess_family: str = "gaussian",
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
    loess_family : str, optional
        LOESS smoothing mode, as in R's ``loess``: ``"gaussian"`` for plain
        least-squares smoothing (the ``pomp::mcap`` behaviour) or
        ``"symmetric"`` for robust bisquare reweighting.  Defaults to
        ``"gaussian"``.

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

    # both fits run on u = (x - x0) / s0, so no design matrix carries the offset
    x0 = 0.5 * (float(np.min(x)) + float(np.max(x)))
    s0 = float(np.max(x)) - float(np.min(x))
    if not np.isfinite(s0) or s0 <= 0.0:
        x0, s0 = 0.0, 1.0  # degenerate range: keep the identity transform
    u: FloatArray = (x - x0) / s0

    # grid over observed parameter range
    u_grid = np.linspace(float(np.min(u)), float(np.max(u)), int(n_grid))
    grid = x0 + s0 * u_grid

    # smooth noisy profile; robust bisquare reweighting only on explicit request
    if loess_family not in ("gaussian", "symmetric"):
        raise ValueError("loess_family must be 'gaussian' or 'symmetric'")
    y_sm = _loess_smooth_1d(
        u,
        y,
        grid=u_grid,
        span=span,
        degree=loess_degree,
        max_iter=10 if loess_family == "symmetric" else 0,
    )

    # MLE = argmax of smoothed profile
    i_max = int(np.nanargmax(y_sm))
    u_mle = float(u_grid[i_max])
    mle = float(x0 + s0 * u_mle)

    # local quadratic at smoothed MLE with raw data, in local coordinates
    a_u, b_u, c_u, vc_ab_u = _fit_local_quadratic(u, y, center=u_mle, span=span)

    # SE decomposition (local units; s0 converts back to parameter units)
    se_stat2 = s0 * s0 / (2.0 * a_u)

    # Monte Carlo variance from vcov(a_u, b_u): no cancellation in local units
    var_a = float(vc_ab_u[0, 0])
    var_b = float(vc_ab_u[1, 1])
    cov_ab = float(vc_ab_u[0, 1])

    se_mc2_u = (
        1.0
        / (4.0 * a_u * a_u)
        * (var_b - 2.0 * (b_u / a_u) * cov_ab + (b_u * b_u / (a_u * a_u)) * var_a)
    )
    se_mc2 = s0 * s0 * se_mc2_u

    # se_tot2 = se_stat2 + se_mc2

    # MC-adjusted cutoff (a * se_mc2 is scale free)
    q = _qchisq(level, df=1)
    delta = float(q * (a_u * se_mc2_u + 0.5))

    # CI from smoothed profile
    diff = float(np.nanmax(y_sm)) - y_sm
    inside = diff < delta
    ci: tuple[float | None, float | None]
    if not np.any(inside):
        ci = (None, None)
    else:
        idx = np.where(inside)[0]
        if idx.max() - idx.min() + 1 != idx.size:
            warnings.warn(
                "acceptance set is not an interval (multimodal profile); "
                "ci is reported as its convex hull"
            )
        ci = (float(grid[idx.min()]), float(grid[idx.max()]))
        if idx.min() == 0 or idx.max() == len(grid) - 1:
            warnings.warn(
                "confidence interval truncated at the profiled range boundary; "
                "widen the parameter range"
            )

    # quadratic curve on grid, evaluated in local units
    quad = c_u - a_u * (u_grid**2) + b_u * u_grid

    if a_u > 0.0:
        quad_max = x0 + s0 * (b_u / (2.0 * a_u))
    else:
        # fallback to smoothed MLE if curvature is non-positive
        quad_max = mle

    # coefficients and their covariance reported in the caller's units
    a = a_u / (s0 * s0)
    b = b_u / s0 + 2.0 * a * x0
    c = c_u - a * x0 * x0 - (b_u / s0) * x0
    J = np.array([[1.0 / (s0 * s0), 0.0], [2.0 * x0 / (s0 * s0), 1.0 / s0]])
    vc_ab = J @ vc_ab_u @ J.T

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

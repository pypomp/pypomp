"""Exact likelihood and MLE for the 1-D linear Gaussian model.

This is the ground truth the accuracy tests measure pypomp against, so it is
deliberately a plain NumPy implementation with no pypomp code in the path.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


def kalman_loglik(
    ys: np.ndarray, a: float, c: float, q: float, r: float, x0: float, p0: float
) -> float:
    """Exact log-likelihood of the 1-D linear Gaussian model.

    Uses the same convention as LG: the initial state is drawn from N(x0, p0)
    at t0, then propagated once before the first observation. ``q`` and ``r``
    are variances, not standard deviations.
    """
    x, p, loglik = x0, p0, 0.0
    for y in ys:
        x_pred = a * x
        p_pred = a * a * p + q
        v = y - c * x_pred
        s = c * c * p_pred + r
        loglik += -0.5 * (np.log(2.0 * np.pi * s) + v * v / s)
        k = c * p_pred / s
        x = x_pred + k * v
        p = (1.0 - k * c) * p_pred
    return float(loglik)


# Bounds on the estimated LG parameters, on the natural scale. A11 is left wide
# rather than pinned to (0, 1): LG's own ParTrans does not constrain it.
_BOUNDS = {"A11": (-0.99, 0.99), "Q11": (1e-4, 5.0), "R11": (1e-4, 5.0)}


def lg_mle(
    ys: np.ndarray,
    estimated: Sequence[str],
    fixed: Mapping[str, float],
    start: Mapping[str, float],
) -> dict[str, float]:
    """Maximum likelihood estimate for a 1-D LG unit.

    ``estimated`` names the parameters to optimize over and ``fixed`` supplies
    the rest (typically ``C11`` and ``X0_1``). Q11/R11 are Cholesky factors, so
    the Kalman filter squares them into variances. Returns the MLE plus the
    attained ``logLik``.
    """
    names = list(estimated)

    def unpack(vec: np.ndarray) -> dict[str, float]:
        theta = dict(fixed)
        theta.update({n: float(v) for n, v in zip(names, vec, strict=True)})
        return theta

    def neg_loglik(vec: np.ndarray) -> float:
        theta = unpack(vec)
        return -kalman_loglik(
            ys,
            a=theta["A11"],
            c=theta["C11"],
            q=theta["Q11"] ** 2,
            r=theta["R11"] ** 2,
            x0=theta["X0_1"],
            p0=theta["Q11"] ** 2,
        )

    res = minimize(
        neg_loglik,
        np.array([float(start[n]) for n in names]),
        method="L-BFGS-B",
        bounds=[_BOUNDS[n] for n in names],
    )
    mle = {n: float(v) for n, v in zip(names, res.x, strict=True)}
    mle["logLik"] = float(-res.fun)
    return mle


def lg_panel_mle(
    ys_by_unit: Mapping[str, np.ndarray],
    shared: Sequence[str],
    unit_specific: Sequence[str],
    fixed: Mapping[str, float],
    start: Mapping[str, float],
) -> dict[str, float]:
    """Maximum likelihood estimate for a panel of 1-D LG units.

    Maximizes the summed per-unit Kalman log-likelihood. Keys of the result are
    the shared names plus ``"{name}_{unit}"`` for the unit-specific ones, with
    the attained total under ``logLik``.
    """
    units = list(ys_by_unit)
    slots: list[tuple[str | None, str]] = [(None, n) for n in shared]
    slots += [(u, n) for u in units for n in unit_specific]

    def unpack(vec: np.ndarray) -> dict[str, dict[str, float]]:
        per_unit = {u: dict(fixed) for u in units}
        for (unit, name), val in zip(slots, vec, strict=True):
            if unit is None:
                for u in units:
                    per_unit[u][name] = float(val)
            else:
                per_unit[unit][name] = float(val)
        return per_unit

    def neg_loglik(vec: np.ndarray) -> float:
        per_unit = unpack(vec)
        total = 0.0
        for u in units:
            theta = per_unit[u]
            total += kalman_loglik(
                ys_by_unit[u],
                a=theta["A11"],
                c=theta["C11"],
                q=theta["Q11"] ** 2,
                r=theta["R11"] ** 2,
                x0=theta["X0_1"],
                p0=theta["Q11"] ** 2,
            )
        return -total

    res = minimize(
        neg_loglik,
        np.array([float(start[n]) for _, n in slots]),
        method="L-BFGS-B",
        bounds=[_BOUNDS[n] for _, n in slots],
    )

    mle: dict[str, float] = {}
    for (unit, name), val in zip(slots, res.x, strict=True):
        mle[name if unit is None else f"{name}_{unit}"] = float(val)
    mle["logLik"] = float(-res.fun)
    return mle

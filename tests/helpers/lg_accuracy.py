"""Helpers for the statistical accuracy tests.

These build 1-D instances of the bundled LG model and compare pypomp's
estimates against exact Kalman results.

Parameter naming follows LG: ``A11`` is the autoregressive coefficient, and
``Q11``/``R11`` are the process and measurement standard deviations. LG takes
``Q``/``R`` as covariances but stores their Cholesky factors in ``theta``, so a
standard deviation ``s`` is passed as ``[[s**2]]`` and read back as ``s``.

``C11`` and ``X0_1`` exist in LG's parameter vector but are held fixed here, so
the tests keep comparing against a three-parameter MLE.
"""

import jax
import numpy as np
import pandas as pd

import pypomp as pp
from tests.helpers.kalman import kalman_loglik

A, Q, R = "A11", "Q11", "R11"
ESTIMATED = [A, Q, R]
FIXED = {"C11": 1.0, "X0_1": 0.0}

# LG draws its initial state from N(X0, Q), so the exact filter starts with
# p0 = Q rather than a point mass.
X0 = 0.0


def lg_1d(a: float, q: float, r: float, T: int, key: jax.Array) -> pp.Pomp:
    """A 1-D LG model whose data is generated at the given parameters."""
    return pp.models.lg(
        T=T,
        A=np.array([[a]]),
        C=np.array([[1.0]]),
        Q=np.array([[q**2]]),
        R=np.array([[r**2]]),
        X0=np.array([X0]),
        key=key,
    )


def lg_1d_ys(model: pp.Pomp) -> np.ndarray:
    """The single observation column of a 1-D LG model."""
    return model.ys.iloc[:, 0].to_numpy().astype(float)


def lg_1d_loglik(ys: np.ndarray, a: float, q: float, r: float) -> float:
    """Exact log-likelihood of ``ys`` under the 1-D LG model."""
    return kalman_loglik(ys, a=a, c=1.0, q=q**2, r=r**2, x0=X0, p0=q**2)


def theta_bounds(bounds: dict[str, tuple[float, float]]) -> dict:
    """Sampling bounds covering LG's full parameter vector.

    The estimated names take ``bounds``; the fixed ones are pinned to a
    degenerate interval so sampling leaves them at their true values.
    """
    return {**bounds, **{n: (v, v) for n, v in FIXED.items()}}


def rw_sd(sigma: float = 0.02, cooling: float = 0.5) -> pp.RWSigma:
    """Perturb only the estimated parameters; hold C11 and X0_1 still."""
    sigmas = {n: sigma for n in ESTIMATED}
    sigmas.update({n: 0.0 for n in FIXED})
    return pp.RWSigma(sigmas=sigmas, init_names=[]).geometric_cooling(cooling)


def lg_1d_panel(
    unit_true: dict[str, dict[str, float]],
    unit_start: dict[str, dict[str, float]],
    shared_names: list[str],
    T: int,
    key: jax.Array,
) -> tuple[pp.PanelPomp, dict[str, np.ndarray]]:
    """A panel of 1-D LG units, each carrying data at its own true parameters.

    ``unit_true`` gives the data-generating parameters per unit and
    ``unit_start`` the (perturbed) values the panel starts from. Returns the
    panel and the per-unit observations for the exact likelihood.
    """
    units = list(unit_true)
    # One key for every unit: units differ through their parameters, not through
    # a separate noise draw. Per-unit keys give some realizations an MLE pinned
    # against a bound, which no estimator can converge to.
    pomps = {
        u: lg_1d(unit_true[u][A], unit_true[u][Q], unit_true[u][R], T=T, key=key)
        for u in units
    }
    ys_by_unit = {u: lg_1d_ys(p) for u, p in pomps.items()}

    # Callers give starting values only for the estimated names; the fixed ones
    # always start at their true values.
    unit_start = {u: {**FIXED, **vals} for u, vals in unit_start.items()}
    unit_names = [n for n in (*ESTIMATED, *FIXED) if n not in shared_names]
    shared_df = (
        pd.DataFrame(
            {
                "shared": [
                    float(np.mean([unit_start[u][n] for u in units]))
                    for n in shared_names
                ]
            },
            index=pd.Index(shared_names),
        )
        if shared_names
        else None
    )
    unit_df = (
        pd.DataFrame(
            {u: [unit_start[u][n] for n in unit_names] for u in units},
            index=pd.Index(unit_names),
        )
        if unit_names
        else pd.DataFrame(index=pd.Index([]), columns=units)
    )

    panel = pp.PanelPomp(
        pomp_dict=pomps,
        theta=pp.PanelParameters([{"shared": shared_df, "unit_specific": unit_df}]),
    )
    return panel, ys_by_unit

"""Model builders shared across the test suite.

Every builder returns a freshly constructed object, so callers may mutate the
result without disturbing other tests.
"""

from collections.abc import Sequence
from typing import Literal

import jax
import numpy as np
import pandas as pd

import pypomp as pp

Sharing = Literal["some", "none", "all"]

# Parameters held shared by the "some" sharing pattern, per model.
LG_SHARED_NAMES = ["A11", "C11"]
SIR_SHARED_NAMES = ["gamma", "mu"]


def _unit_names(n_units: int) -> list[str]:
    return [f"unit{i}" for i in range(1, n_units + 1)]


def _split_names(
    names: Sequence[str], sharing: Sharing, shared_names: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    """Partition parameter names into (shared, unit-specific)."""
    if sharing == "none":
        shared: list[str] = []
    elif sharing == "all":
        shared = list(names)
    else:
        if shared_names is None:
            raise ValueError("sharing='some' requires shared_names")
        shared = list(shared_names)
    return shared, [n for n in names if n not in shared]


def _panel_theta(
    pomps: dict[str, pp.Pomp],
    shared: Sequence[str],
    unit_specific: Sequence[str],
    n_reps: int,
    unit_scales: Sequence[float] | None = None,
) -> pp.PanelParameters:
    """Build PanelParameters, averaging per-unit values for shared entries.

    ``unit_scales`` multiplies each unit's unit-specific starting values, which
    is how tests spread the units apart from a common starting point.
    """
    units = list(pomps)
    params = {name: pomp.theta[0] for name, pomp in pomps.items()}
    scales = (
        {u: 1.0 for u in units}
        if unit_scales is None
        else dict(zip(units, unit_scales, strict=True))
    )

    shared_df = (
        pd.DataFrame(
            {"shared": [float(np.mean([params[u][n] for u in units])) for n in shared]},
            index=pd.Index(list(shared)),
        )
        if shared
        else None
    )

    if unit_specific:
        unit_df = pd.DataFrame(
            {u: [params[u][n] * scales[u] for n in unit_specific] for u in units},
            index=pd.Index(list(unit_specific)),
        )
    else:
        # With every parameter shared, PanelParameters still records unit
        # identity through the columns of an otherwise empty frame.
        unit_df = pd.DataFrame(index=pd.Index([]), columns=units)

    theta = pp.PanelParameters(theta=[{"shared": shared_df, "unit_specific": unit_df}])
    return theta * n_reps if n_reps > 1 else theta


def _panel(
    pomps: dict[str, pp.Pomp],
    sharing: Sharing,
    n_reps: int,
    default_shared: Sequence[str],
    shared_names: Sequence[str] | None,
    unit_scales: Sequence[float] | None = None,
    par_trans: pp.ParTrans | None = None,
) -> pp.PanelPomp:
    if par_trans is not None:
        for pomp in pomps.values():
            pomp.par_trans = par_trans
    names = list(next(iter(pomps.values())).canonical_param_names)
    shared, unit_specific = _split_names(
        names, sharing, default_shared if shared_names is None else shared_names
    )
    theta = _panel_theta(pomps, shared, unit_specific, n_reps, unit_scales)
    return pp.PanelPomp(pomp_dict=pomps, theta=theta)


def lg_panel(
    sharing: Sharing = "some",
    n_units: int = 2,
    n_reps: int = 1,
    shared_names: Sequence[str] | None = None,
    unit_scales: Sequence[float] | None = None,
    par_trans: pp.ParTrans | None = None,
) -> pp.PanelPomp:
    """Build a PanelPomp of LG units.

    ``sharing`` selects which parameters are shared across units: ``"some"``
    shares ``shared_names`` (defaulting to ``A11`` and ``C11``), ``"none"``
    makes every parameter unit-specific, and ``"all"`` shares every parameter.
    Shared starting values are the mean of the per-unit values, and
    ``unit_scales`` spreads the unit-specific ones apart.
    """
    pomps = {name: pp.models.lg() for name in _unit_names(n_units)}
    return _panel(
        pomps, sharing, n_reps, LG_SHARED_NAMES, shared_names, unit_scales, par_trans
    )


def sir_panel(
    sharing: Sharing = "none",
    n_units: int = 2,
    n_reps: int = 1,
    times: np.ndarray | None = None,
    shared_names: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> pp.PanelPomp:
    """Build a PanelPomp of SIR units, each seeded differently.

    ``times`` defaults to four weekly observations, the shortest series the dpop
    tests exercise. ``seeds`` defaults to 100, 200, ... one per unit.
    """
    if times is None:
        times = np.arange(1 / 52, 5 / 52, 1 / 52)
    if seeds is None:
        seeds = [100 * (i + 1) for i in range(n_units)]
    pomps = {
        name: pp.models.sir(times=times, key=jax.random.key(seeds[i]))
        for i, name in enumerate(_unit_names(n_units))
    }
    return _panel(pomps, sharing, n_reps, SIR_SHARED_NAMES, shared_names)

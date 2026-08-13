"""Random-walk sigma and learning-rate builders shared across the test suite."""

from collections.abc import Sequence

import pypomp as pp

# The measles random-walk sigmas used by both the panel and single-unit tests.
# The initial-condition entries are perturbed half as hard as the rest.
MEASLES_RW_SIGMAS = {
    "R0": 0.02,
    "sigma": 0.02,
    "gamma": 0.02,
    "iota": 0.02,
    "rho": 0.02,
    "sigmaSE": 0.02,
    "psi": 0.02,
    "cohort": 0.02,
    "amplitude": 0.02,
    "S_0": 0.01,
    "E_0": 0.01,
    "I_0": 0.01,
    "R_0": 0.01,
}
MEASLES_INIT_NAMES = ["S_0", "E_0", "I_0", "R_0"]


ParamSource = pp.Pomp | pp.PanelPomp | Sequence[str]


def _names(source: ParamSource) -> Sequence[str]:
    """Accept either a model or an explicit list of parameter names."""
    if isinstance(source, pp.Pomp | pp.PanelPomp):
        return source.canonical_param_names
    return source


def uniform_rw_sd(
    source: ParamSource,
    sigma: float = 0.02,
    init_names: Sequence[str] | None = None,
    cooling: float | None = None,
) -> pp.RWSigma:
    """An RWSigma giving every parameter the same sigma.

    ``cooling`` applies geometric cooling with that factor when given.
    """
    rw_sd = pp.RWSigma(
        sigmas={n: sigma for n in _names(source)},
        init_names=list(init_names) if init_names else [],
    )
    return rw_sd if cooling is None else rw_sd.geometric_cooling(cooling)


def uniform_eta(source: ParamSource, eta: float = 0.01) -> pp.LearningRate:
    """A LearningRate giving every parameter the same rate."""
    return pp.LearningRate({n: eta for n in _names(source)})


def measles_rw_sd() -> pp.RWSigma:
    """The standard measles RWSigma, perturbing initial conditions at step 0."""
    return pp.RWSigma(
        sigmas=dict(MEASLES_RW_SIGMAS), init_names=list(MEASLES_INIT_NAMES)
    )

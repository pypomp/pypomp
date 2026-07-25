"""Direct unit tests for the Config/Inputs classmethods in core/algorithms/types.py.

The public functional entry points (pypomp.functional.mif/abc/pmcmc/...) each
have their own upfront validation (e.g. "dmeas cannot be None") before ever
building a Config object, so these classmethods' own internal checks are
otherwise unreachable except by calling them directly, as done here.
"""

import pandas as pd
import pytest

import pypomp as pp
from pypomp.core.algorithms.types import AbcConfig, MifConfig, PmcmcConfig


def _rinit(theta_, key, covars, t0):
    return {"X": 0.0}


def _rproc(X_, theta_, key, covars, t, dt):
    return {"X": X_["X"] + theta_["sigma"] * 0.0}


def _rmeas(X_, theta_, key, covars, t):
    return {"Y": X_["X"]}


def _dmeas(Y_, X_, theta_, covars, t):
    import jax.scipy.stats

    return jax.scipy.stats.norm.logpdf(Y_["Y"], loc=X_["X"], scale=0.1)


def _build_pomp(with_dmeas: bool, with_rmeas: bool, with_dprior: bool = False):
    def dprior(theta_):
        return 0.0

    return pp.Pomp(
        ys=pd.DataFrame({"Y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"sigma": 0.1}),
        statenames=["X"],
        t0=0.0,
        rinit=_rinit,
        rproc=_rproc,
        dmeas=_dmeas if with_dmeas else None,
        rmeas=_rmeas if with_rmeas else None,
        dprior=dprior if with_dprior else None,
        nstep=1,
    )


def test_mif_config_requires_dmeas():
    struct = _build_pomp(with_dmeas=False, with_rmeas=True).to_struct()
    with pytest.raises(ValueError, match="dmeasure is required for MIF"):
        MifConfig.from_mif_struct(struct, J=2, M=2)


def test_abc_config_requires_rmeas():
    struct = _build_pomp(with_dmeas=True, with_rmeas=False).to_struct()
    with pytest.raises(ValueError, match="abc requires struct.rmeas_pf"):
        AbcConfig.from_abc_struct(struct, M=2)


def test_abc_config_requires_dprior():
    struct = _build_pomp(
        with_dmeas=False, with_rmeas=True, with_dprior=False
    ).to_struct()
    with pytest.raises(ValueError, match="dprior is required for ABC"):
        AbcConfig.from_abc_struct(struct, M=2, dprior=None)


def test_pmcmc_config_requires_dmeas():
    struct = _build_pomp(with_dmeas=False, with_rmeas=True).to_struct()
    with pytest.raises(ValueError, match="dmeasure is required for PMCMC"):
        PmcmcConfig.from_pmcmc_struct(struct, M=2, J=2)


def test_pmcmc_config_requires_dprior():
    struct = _build_pomp(
        with_dmeas=True, with_rmeas=False, with_dprior=False
    ).to_struct()
    with pytest.raises(ValueError, match="dprior is required for PMCMC"):
        PmcmcConfig.from_pmcmc_struct(struct, M=2, J=2, dprior=None)

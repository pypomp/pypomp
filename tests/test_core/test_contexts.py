"""Coverage for validation branches in pypomp/core/algorithms/contexts.py.

The context factory classmethods (``ModelFns.pf``/``.per``,
``AbcContext.from_struct``, ``PmcmcContext.from_struct``) each re-validate
that the struct carries the components they need. Every one of these turns
out to be a belt-and-suspenders duplicate of a check the public ``Pomp``
methods or ``functional`` wrappers already perform earlier (confirmed by
reading ``estimation_mixin.py`` and ``functional/abc.py``/``pmcmc.py``), so
they are unreachable through the public API. We exercise the classmethods
directly instead, since they are a real internal safety net worth having
correct even if nothing currently reaches them.
"""

import jax.numpy as jnp
import pytest

import pypomp as pp
from pypomp.core.algorithms.contexts import AbcContext, ModelFns, PmcmcContext


def test_modelfns_pf_requires_dmeas_pf():
    pomp = pp.models.sir(seed=42)
    struct = pomp.to_struct()._replace(dmeas_pf=None)
    with pytest.raises(ValueError, match="dmeasure \\(dmeas_pf\\) is required"):
        ModelFns.pf(struct)


def test_modelfns_per_requires_dmeas_per():
    pomp = pp.models.sir(seed=42)
    struct = pomp.to_struct()._replace(dmeas_per=None)
    with pytest.raises(ValueError, match="dmeasure \\(dmeas_per\\) is required"):
        ModelFns.per(struct)


def test_abccontext_requires_rmeas_pf():
    pomp = pp.models.sir(seed=42)
    struct = pomp.to_struct()._replace(rmeas_pf=None)
    with pytest.raises(ValueError, match="abc requires struct.rmeas_pf"):
        AbcContext.from_struct(
            struct,
            M=2,
            obs_probes=jnp.array([0.0]),
            scale_arr=jnp.array([1.0]),
            epsilon=1.0,
        )


def test_abccontext_requires_dprior():
    pomp = pp.models.sir(seed=42)
    struct = pomp.to_struct()
    assert struct.dprior_pf is None
    with pytest.raises(ValueError, match="dprior is required for ABC"):
        AbcContext.from_struct(
            struct,
            M=2,
            obs_probes=jnp.array([0.0]),
            scale_arr=jnp.array([1.0]),
            epsilon=1.0,
            dprior=None,
        )


def test_pmcmccontext_requires_dprior():
    pomp = pp.models.sir(seed=42)
    struct = pomp.to_struct()
    assert struct.dprior_pf is None
    with pytest.raises(ValueError, match="dprior is required for PMCMC"):
        PmcmcContext.from_struct(struct, M=2, J=5, dprior=None)


def test_pfilter_public_api_rejects_missing_dmeas_earlier():
    """Documents that the public API's own guard fires before ModelFns.pf's."""
    import jax
    import pandas as pd

    def rinit(theta_, key, covars, t0):
        return {"X": theta_["X0"]}

    def rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"]}

    def rmeas(X_, theta_, key, covars, t):
        return {"y": X_["X"] + 0.1 * jax.random.normal(key, ())}

    pomp = pp.Pomp(
        ys=pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"X0": 0.0}),
        rinit=rinit,
        rproc=rproc,
        rmeas=rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="self.dmeas cannot be None"):
        pomp.pfilter(J=5, reps=1, key=jax.random.key(0))

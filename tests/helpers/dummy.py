"""A minimal two-observation POMP used by the validation and pickling tests.

The component functions live here rather than in a conftest so that tests and
fixtures share one function object: pytest loads conftest.py under its own
module name, so importing from it would yield a second, unequal copy.
"""

import jax
import pandas as pd

import pypomp as pp


def dummy_rinit(theta_, key, covars, t0):
    val = next(iter(theta_.values()))
    return {"X": val}


def dummy_rproc(X_, theta_, key, covars, t, dt):
    return {"X": X_["X"] + theta_["sigma"] * jax.random.normal(key, ())}


def dummy_dmeas(Y_, X_, theta_, covars, t):
    return jax.scipy.stats.norm.logpdf(Y_["y"], loc=X_["X"], scale=0.1)


def dummy_rmeas(X_, theta_, key, covars, t):
    return {"y": X_["X"] + 0.1 * jax.random.normal(key, ())}


def dummy_dprior(theta_):
    return 0.0


def dummy_pomp(with_dprior: bool = False) -> pp.Pomp:
    """Build the minimal POMP, optionally carrying a dprior."""
    pomp = pp.Pomp(
        ys=pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"X0": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        dprior=dummy_dprior if with_dprior else None,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp.fresh_key = jax.random.key(1)
    return pomp

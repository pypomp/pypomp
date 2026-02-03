"""
Simple SIS (Susceptible–Infected–Susceptible) model for POMP.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd

from pypomp.pomp_class import Pomp
from pypomp.ParTrans_class import ParTrans

# ---------------------------------------------------------------------
# 1. Defaults and state names
# ---------------------------------------------------------------------

STATES = ["S", "I", "logw"]

# Default natural-scale parameters (can be overridden by user)
DEFAULT_THETA = {
    "beta": 0.6,     # infection rate           > 0
    "gamma": 0.4,    # recovery rate            > 0
    "rho": 0.3,      # reporting probability    in (0,1)
    "N": 1000.0,     # total population         > 0
    "I0": 10.0,      # initial infected         > 0
}


# ---------------------------------------------------------------------
# 2. Parameter transforms: natural <-> estimation space
# ---------------------------------------------------------------------

def sis_to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """
    Map natural parameters to unconstrained estimation space.

    Natural space:
        beta  > 0
        gamma > 0
        rho   in (0,1)
        N     > 0
        I0    > 0

    Estimation space (all real numbers):
        beta_est  = log(beta)
        gamma_est = log(gamma)
        rho_est   = logit(rho)
        N_est     = log(N)
        I0_est    = log(I0)
    """
    return {
        "beta": jnp.log(theta["beta"]),
        "gamma": jnp.log(theta["gamma"]),
        "rho": jnp.log(theta["rho"] / (1.0 - theta["rho"])),
        "N": jnp.log(theta["N"]),
        "I0": jnp.log(theta["I0"]),
    }


def sis_from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """
    Map estimation-space parameters back to natural space.

    Given:
        beta_est, gamma_est, rho_est, N_est, I0_est in R

    Return:
        beta  = exp(beta_est)           > 0
        gamma = exp(gamma_est)          > 0
        rho   = sigmoid(rho_est)        in (0,1)
        N     = exp(N_est)              > 0
        I0    = exp(I0_est)             > 0
    """
    return {
        "beta": jnp.exp(theta["beta"]),
        "gamma": jnp.exp(theta["gamma"]),
        "rho": 1.0 / (1.0 + jnp.exp(-theta["rho"])),
        "N": jnp.exp(theta["N"]),
        "I0": jnp.exp(theta["I0"]),
    }


# ---------------------------------------------------------------------
# 3. Model components: rinit, rproc, dmeas, rmeas
# ---------------------------------------------------------------------

def rinit(theta_, key, covars=None, t0=None):
    """
    Initial state simulator for the SIS model.
    """
    N = theta_["N"]
    I0 = theta_["I0"]
    S0 = N - I0

    return {"S": S0, "I": I0, "logw": 0.0}


def rproc(X_, theta_, key, covars, t, dt):
    """
    One Euler step of the SIS process with binomial transitions.
    """
    S = X_["S"]
    I = X_["I"]
    logw = X_["logw"]

    beta = theta_["beta"]
    gamma = theta_["gamma"]
    N = theta_["N"]

    # Avoid degenerate N
    N = jnp.maximum(N, 1.0)

    # Force of infection per susceptible
    lam = beta * I / N

    # Probabilities for binomial transitions over dt
    p_SI = 1.0 - jnp.exp(-lam * dt)     # S -> I
    p_IR = 1.0 - jnp.exp(-gamma * dt)   # I -> S

    # Clip probabilities to avoid exact 0 / 1 (for logpmf stability)
    p_SI = jnp.clip(p_SI, 1e-8, 1.0 - 1e-8)
    p_IR = jnp.clip(p_IR, 1e-8, 1.0 - 1e-8)

    # Split key for the two binomials
    key, k1, k2 = jax.random.split(key, 3)

    # Sample transitions (integer S, I)
    S_int = jnp.maximum(jnp.round(S), 0.0).astype(jnp.int32)
    I_int = jnp.maximum(jnp.round(I), 0.0).astype(jnp.int32)

    n_SI = jax.random.binomial(k1, n=S_int, p=p_SI)  # S -> I
    n_IR = jax.random.binomial(k2, n=I_int, p=p_IR)  # I -> S

    # Update compartments
    S_new = S - n_SI + n_IR
    I_new = I + n_SI - n_IR

    # Enforce non-negativity
    S_new = jnp.maximum(S_new, 0.0)
    I_new = jnp.maximum(I_new, 0.0)

    # Process log-density contributions
    lp_SI = jsp.stats.binom.logpmf(n_SI, S_int, p_SI)
    lp_IR = jsp.stats.binom.logpmf(n_IR, I_int, p_IR)

    logw_new = logw + lp_SI + lp_IR

    return {"S": S_new, "I": I_new, "logw": logw_new}


def dmeas(Y_, X_, theta_, covars=None, t=None):
    """
    """
    y = Y_["cases"]
    I = X_["I"]
    rho = theta_["rho"]

    lam = rho * jnp.maximum(I, 0.0)
    lam = jnp.clip(lam, 1e-8, 1e12)

    return jsp.stats.poisson.logpmf(y, lam)


def rmeas(X_, theta_, key, covars=None, t=None):
    """
    Measurement simulator: generate Y_t ~ Poisson(rho * I_t).
    """
    I = X_["I"]
    rho = theta_["rho"]

    lam = rho * jnp.maximum(I, 0.0)
    lam = jnp.clip(lam, 1e-8, 1e12)

    y = jax.random.poisson(key, lam=lam)
    return jnp.array([y], dtype=jnp.float64)


# ---------------------------------------------------------------------
# 4. Helper constructor: SIS(...) -> Pomp
# ---------------------------------------------------------------------

def SIS(
    T: int = 100,
    theta: dict | None = None,
    key: jax.Array | None = None,
):
    """
    Construct a Pomp object for the simple SIS model and simulate data.
    """
    if theta is None:
        theta = DEFAULT_THETA

    if key is None:
        key = jax.random.PRNGKey(123)

    # Parameter transform object: log / logit transforms
    par_trans = ParTrans(to_est=sis_to_est, from_est=sis_from_est)

    # 4.1 Dummy observations (all zeros) just to run simulate()
    times = np.arange(1, T + 1, dtype=float)
    ys_dummy = pd.DataFrame({"cases": np.zeros(T)}, index=times)

    sis_temp = Pomp(
        ys=ys_dummy,
        theta=theta,
        statenames=STATES,
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        par_trans=par_trans,
        nstep=None,
        dt=1.0 / 7.0,    # 7 Euler substeps per observation
        accumvars=("logw",),  # name of accumulated state variable
        covars=None,
        ydim=1,
    )

    # 4.2 Simulate one trajectory (latent + observed)
    X_long, Y_long = sis_temp.simulate(key=key, nsim=1)

    # Extract the simulated observations for replicate=0, sim=0
    y_sims = (
        Y_long.loc[(Y_long["replicate"] == 0) & (Y_long["sim"] == 0)]
        .set_index("time")[["obs_0"]]
        .rename(columns={"obs_0": "cases"})
    )

    assert isinstance(y_sims, pd.DataFrame)

    # 4.3 Build final Pomp object for inference / experiments
    sis_obj = Pomp(
        ys=y_sims,
        theta=theta,
        statenames=STATES,
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        par_trans=par_trans,
        nstep=None,
        dt=1.0 / 7.0,
        accumvars=("logw",),  # name of accumulated state variable
        covars=None,
        ydim=1,
    )

    return sis_obj

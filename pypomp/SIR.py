"""
SIR model with seasonal forcing, translated from pomp::sir in R.
This version supports DPOP (Differentiable Particle Filter) by accumulating
process log-density in the state variable 'logw'.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
import numpy as np
import pandas as pd

from pypomp.pomp_class import Pomp
from pypomp.ParTrans_class import ParTrans
from pypomp.ctmc_multinom import sample_and_log_prob

# ---------------------------------------------------------------------
# State names and default parameters
# ---------------------------------------------------------------------

STATENAMES = ["S", "I", "R", "cases", "W", "logw"]

DEFAULT_THETA = {
    "gamma": 26.0,
    "mu": 0.02,
    "iota": 0.01,
    "beta1": 400.0,
    "beta2": 480.0,
    "beta3": 320.0,
    "beta_sd": 0.001,
    "rho": 0.6,
    "k": 0.1,
    "pop": 2100000.0,
    "S_0": 26.0 / 400.0,
    "I_0": 0.001,
    "R_0": 1.0 - 26.0 / 400.0 - 0.001,
}


# ---------------------------------------------------------------------
# Periodic B-spline basis (from pomp's C implementation)
# ---------------------------------------------------------------------

def _bspline_eval(x, knots, i, degree, deriv=0):
    if deriv > degree:
        return 0.0
    elif deriv > 0:
        i2 = i + 1
        p2 = degree - 1
        d2 = deriv - 1
        y1 = _bspline_eval(x, knots, i, p2, d2)
        y2 = _bspline_eval(x, knots, i2, p2, d2)
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i2 + degree] - knots[i2]
        a = jnp.where(jnp.abs(denom1) > 1e-10, degree / denom1, 0.0)
        b = jnp.where(jnp.abs(denom2) > 1e-10, degree / denom2, 0.0)
        return a * y1 - b * y2
    else:
        if degree > 0:
            i2 = i + 1
            p2 = degree - 1
            y1 = _bspline_eval(x, knots, i, p2, 0)
            y2 = _bspline_eval(x, knots, i2, p2, 0)
            denom1 = knots[i + degree] - knots[i]
            denom2 = knots[i2 + degree] - knots[i2]
            a = jnp.where(
                jnp.abs(denom1) > 1e-10, (x - knots[i]) / denom1, 0.0
            )
            b = jnp.where(
                jnp.abs(denom2) > 1e-10,
                (knots[i2 + degree] - x) / denom2,
                0.0,
            )
            return a * y1 + b * y2
        else:
            return jnp.where(
                (knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0
            )


def periodic_bspline_basis_eval(x, period, degree, nbasis, deriv=0):
    nknots = nbasis + 2 * degree + 1
    shift = (degree - 1) // 2
    dx = period / nbasis
    knots = jnp.array(
        [(k) * dx for k in range(-degree, nbasis + degree + 2)]
    )
    x_wrapped = x % period
    x_wrapped = jnp.where(x_wrapped < 0, x_wrapped + period, x_wrapped)
    yy = jnp.array(
        [
            _bspline_eval(x_wrapped, knots, k, degree, deriv)
            for k in range(nknots)
        ]
    )
    yy_adjusted = yy.at[:degree].add(yy[nbasis : nbasis + degree])
    y = jnp.array(
        [yy_adjusted[(shift + k) % nbasis] for k in range(nbasis)]
    )
    return y


def precompute_bspline_covars(times, t0, period=1.0, nbasis=3, degree=3):
    t_min = t0
    t_max = max(times) + 0.2
    t_grid = np.arange(t_min, t_max + 0.01, 0.01)
    basis_data = {f"seas_{i+1}": [] for i in range(nbasis)}
    for t in t_grid:
        basis = periodic_bspline_basis_eval(
            t, period, degree, nbasis, deriv=0
        )
        for i in range(nbasis):
            basis_data[f"seas_{i+1}"].append(float(basis[i]))
    return pd.DataFrame(basis_data, index=pd.Index(t_grid, name="time"))


# ---------------------------------------------------------------------
# Parameter transformations
# ---------------------------------------------------------------------

def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    SIR_0 = jnp.array([theta["S_0"], theta["I_0"], theta["R_0"]])
    SIR_0 = SIR_0 / jnp.sum(SIR_0)
    S_0_est, I_0_est, R_0_est = jnp.log(SIR_0)
    return {
        "gamma": jnp.log(theta["gamma"]),
        "mu": jnp.log(theta["mu"]),
        "iota": jnp.log(theta["iota"]),
        "beta1": jnp.log(theta["beta1"]),
        "beta2": jnp.log(theta["beta2"]),
        "beta3": jnp.log(theta["beta3"]),
        "beta_sd": jnp.log(theta["beta_sd"]),
        "rho": jspecial.logit(theta["rho"]),
        "k": jnp.log(theta["k"]),
        "pop": jnp.log(theta["pop"]),
        "S_0": S_0_est,
        "I_0": I_0_est,
        "R_0": R_0_est,
    }


def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    SIR_0 = jnp.exp(
        jnp.array([theta["S_0"], theta["I_0"], theta["R_0"]])
    )
    SIR_0 = SIR_0 / jnp.sum(SIR_0)
    return {
        "gamma": jnp.exp(theta["gamma"]),
        "mu": jnp.exp(theta["mu"]),
        "iota": jnp.exp(theta["iota"]),
        "beta1": jnp.exp(theta["beta1"]),
        "beta2": jnp.exp(theta["beta2"]),
        "beta3": jnp.exp(theta["beta3"]),
        "beta_sd": jnp.exp(theta["beta_sd"]),
        "rho": jspecial.expit(theta["rho"]),
        "k": jnp.exp(theta["k"]),
        "pop": jnp.exp(theta["pop"]),
        "S_0": SIR_0[0],
        "I_0": SIR_0[1],
        "R_0": SIR_0[2],
    }


# ---------------------------------------------------------------------
# Model components: rinit, rproc, dmeas, rmeas
# ---------------------------------------------------------------------

def rinit(theta_, key, covars=None, t0=None):
    pop = theta_["pop"]
    S_0 = theta_["S_0"]
    I_0 = theta_["I_0"]
    R_0 = theta_["R_0"]
    m = pop / (S_0 + I_0 + R_0)
    S = jnp.round(m * S_0)
    I = jnp.round(m * I_0)
    R = jnp.round(m * R_0)
    return {"S": S, "I": I, "R": R, "cases": 0.0, "W": 0.0, "logw": 0.0}


def rproc(X_, theta_, key, covars, t, dt):
    # Unpack state
    S, I, R = X_["S"], X_["I"], X_["R"]
    cases, W, logw = X_["cases"], X_["W"], X_["logw"]

    # Unpack parameters
    gamma = theta_["gamma"]
    mu = theta_["mu"]
    iota = theta_["iota"]
    beta1, beta2, beta3 = theta_["beta1"], theta_["beta2"], theta_["beta3"]
    beta_sd = theta_["beta_sd"]
    pop = theta_["pop"]

    # Seasonal transmission
    seas_1, seas_2, seas_3 = (
        covars["seas_1"],
        covars["seas_2"],
        covars["seas_3"],
    )
    beta = beta1 * seas_1 + beta2 * seas_2 + beta3 * seas_3

    # Gamma white noise
    key, subkey = jax.random.split(key)
    shape = dt / (beta_sd**2 + 1e-10)
    dW = jax.random.gamma(subkey, shape) * (beta_sd**2)

    # Transition rates
    rate_foi = (iota + beta * I * dW / dt) / pop

    # Split keys
    key, k1, k2, k3, k4 = jax.random.split(key, 5)

    # Births: Poisson
    births = jax.random.poisson(k1, mu * pop * dt)

    # S transitions
    S_int = jnp.maximum(jnp.round(S), 0.0)
    rates_S = jnp.array([rate_foi, mu])
    trans_S, lp_S, _ = sample_and_log_prob(S_int, rates_S, dt, k2)
    infections, deaths_S = trans_S[0], trans_S[1]

    # I transitions
    I_int = jnp.maximum(jnp.round(I), 0.0)
    rates_I = jnp.array([gamma, mu])
    trans_I, lp_I, _ = sample_and_log_prob(I_int, rates_I, dt, k3)
    recoveries, deaths_I = trans_I[0], trans_I[1]

    # R transitions
    R_int = jnp.maximum(jnp.round(R), 0.0)
    rates_R = jnp.array([mu])
    trans_R, lp_R, _ = sample_and_log_prob(R_int, rates_R, dt, k4)
    deaths_R = trans_R[0]

    # Accumulate process log-density
    logw_step = lp_S + lp_I + lp_R
    logw_step = jnp.where(jnp.isfinite(logw_step), logw_step, 0.0)

    # Update state
    S_new = S + births - infections - deaths_S
    I_new = I + infections - recoveries - deaths_I
    R_new = R + recoveries - deaths_R
    cases_new = cases + recoveries
    W_new = jnp.where(beta_sd > 0, W + (dW - dt) / beta_sd, W)
    logw_new = logw + logw_step

    return {
        "S": S_new,
        "I": I_new,
        "R": R_new,
        "cases": cases_new,
        "W": W_new,
        "logw": logw_new,
    }


def dmeas(Y_, X_, theta_, covars=None, t=None):
    reports = Y_["reports"]
    cases = X_["cases"]
    rho, k = theta_["rho"], theta_["k"]
    mu = jnp.maximum(cases * rho, 1e-10)
    size = 1.0 / k
    reports_int = jnp.round(reports)
    return (
        jax.scipy.special.gammaln(reports_int + size)
        - jax.scipy.special.gammaln(size)
        - jax.scipy.special.gammaln(reports_int + 1)
        + size * jnp.log(size / (size + mu))
        + reports_int * jnp.log(mu / (size + mu))
    )


def rmeas(X_, theta_, key, covars=None, t=None):
    cases = X_["cases"]
    rho, k = theta_["rho"], theta_["k"]
    mu = jnp.maximum(cases * rho, 1e-10)
    size = 1.0 / k

    # Negative binomial as Gamma-Poisson mixture
    key1, key2 = jax.random.split(key)
    scale = mu / size
    gamma_sample = jax.random.gamma(key1, size) * scale
    reports = jax.random.poisson(key2, gamma_sample)

    return jnp.array([reports], dtype=jnp.float64)


# ---------------------------------------------------------------------
# Constructor function
# ---------------------------------------------------------------------

def sir(
    gamma: float = 26.0,
    mu: float = 0.02,
    iota: float = 0.01,
    beta1: float = 400.0,
    beta2: float = 480.0,
    beta3: float = 320.0,
    beta_sd: float = 0.001,
    rho: float = 0.6,
    k: float = 0.1,
    pop: float = 2100000.0,
    S_0: float = 26.0 / 400.0,
    I_0: float = 0.001,
    R_0: float | None = None,
    t0: float = 0.0,
    times: np.ndarray | None = None,
    seed: int = 329343545,
    delta_t: float = 1 / 52 / 20,
) -> Pomp:
    """
    Create a Pomp object for the SIR model with seasonal forcing.
    Supports DPOP through the logw state variable.
    """
    if R_0 is None:
        R_0 = 1.0 - S_0 - I_0
    if times is None:
        times = np.arange(t0 + 1 / 52, t0 + 4 + 1 / 52, 1 / 52)

    theta = {
        "gamma": gamma,
        "mu": mu,
        "iota": iota,
        "beta1": beta1,
        "beta2": beta2,
        "beta3": beta3,
        "beta_sd": beta_sd,
        "rho": rho,
        "k": k,
        "pop": pop,
        "S_0": S_0,
        "I_0": I_0,
        "R_0": R_0,
    }

    covars = precompute_bspline_covars(times, t0)
    par_trans = ParTrans(to_est=to_est, from_est=from_est)
    ys_dummy = pd.DataFrame(
        {"reports": np.zeros(len(times))}, index=pd.Index(times)
    )

    accumvars = ("cases", "logw")

    sir_temp = Pomp(
        ys=ys_dummy,
        theta=theta,
        statenames=STATENAMES,
        t0=t0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        par_trans=par_trans,
        nstep=20,
        accumvars=accumvars,
        covars=covars,
        ydim=1,
    )

    # Simulate data
    key = jax.random.PRNGKey(seed)
    X_long, Y_long = sir_temp.simulate(key=key, nsim=1)
    y_sims = (
        Y_long.loc[(Y_long["replicate"] == 0) & (Y_long["sim"] == 0)]
        .set_index("time")[["obs_0"]]
        .rename(columns={"obs_0": "reports"})
    )

    # Final Pomp object with simulated data
    return Pomp(
        ys=y_sims,
        theta=theta,
        statenames=STATENAMES,
        t0=t0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        par_trans=par_trans,
        nstep=20,
        accumvars=accumvars,
        covars=covars,
        ydim=1,
    )


def get_process_weight_index():
    """Return the index of logw in STATENAMES for DPOP."""
    return STATENAMES.index("logw")

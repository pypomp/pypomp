import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax
from pypomp.pomp_class import Pomp
from pypomp.ParTrans_class import ParTrans

module_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_dir, "data")
data_file = os.path.join(data_dir, "SPX.csv")

sp500_raw = pd.read_csv(data_file)
sp500 = sp500_raw.copy()
sp500["date"] = pd.to_datetime(sp500["Date"])
sp500["diff_days"] = (sp500["date"] - sp500["date"].min()).dt.days
sp500["time"] = sp500["diff_days"].astype(float)
sp500["y"] = np.log(sp500["Close"] / sp500["Close"].shift(1))
sp500 = sp500.dropna(subset=["y"])[["time", "y"]]
sp500.set_index("time", inplace=True)

first_time = float(np.asarray(sp500.index)[0]) - 1.0
covars = pd.DataFrame(sp500["y"], index=sp500.index)
covars.loc[first_time] = 0
covars = covars.sort_index()
covars = covars.rename(columns={"y": "y_prev"})


theta = {
    "mu": 3.68e-4,
    "kappa": 3.14e-2,
    "theta": 1.12e-4,
    "xi": 2.27e-3,
    "rho": -7.38e-1,
    "V_0": 7.66e-3**2,
}

statenames = ["V", "S"]


def rinit(theta_, key, covars=None, t0=None):
    V_0 = theta_["V_0"]
    S_0 = 1105  # Initial price
    return {"V": V_0, "S": S_0}


def rproc(X_, theta_, key, covars, t=None, dt=None):
    V, S = X_["V"], X_["S"]
    mu, kappa, theta_val, xi, rho = (
        theta_["mu"],
        theta_["kappa"],
        theta_["theta"],
        theta_["xi"],
        theta_["rho"],
    )
    y_prev = covars["y_prev"]
    # Wiener process generation (Gaussian noise)
    dZ = random.normal(key)
    dWs = (y_prev - mu + 0.5 * V) / jnp.sqrt(V)
    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho**2) * dZ
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 1e-32)) * dWs)
    V = V + kappa * (theta_val - V) + xi * jnp.sqrt(V) * dWv
    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    # Results must be returned as a dict
    return {"V": V, "S": S}


def dmeas(Y_, X_, theta_, covars=None, t=None):
    V = X_["V"]
    mu = theta_["mu"]
    return jax.scipy.stats.norm.logpdf(Y_, mu - 0.5 * V, jnp.sqrt(V))


def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "mu": jnp.log(theta["mu"]),
        "kappa": jnp.log(theta["kappa"]),
        "theta": jnp.log(theta["theta"]),
        "xi": jnp.log(theta["xi"]),
        "rho": jnp.log((1 + theta["rho"]) / (1 - theta["rho"])),
        "V_0": jnp.log(theta["V_0"]),
    }


def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        "mu": jnp.exp(theta["mu"]),
        "kappa": jnp.exp(theta["kappa"]),
        "theta": jnp.exp(theta["theta"]),
        "xi": jnp.exp(theta["xi"]),
        "rho": -1 + 2 / (1 + jnp.exp(-theta["rho"])),
        "V_0": jnp.exp(theta["V_0"]),
    }


def spx():
    """
    Creates a POMP model for the S&P 500 stock index data.

    This function constructs a Partially Observed Markov Process (POMP) model
    for analyzing the S&P 500 stock index data. The model uses a stochastic
    volatility framework where the volatility follows a mean-reverting process
    and the log returns follow a normal random walk with time-varying variance.

    Returns
    -------
    Pomp
        A POMP model object containing:
        - ys: S&P 500 log returns data.
        - theta: Model parameters including mu, kappa, theta, xi, rho, and V_0.
        - rinit: Initial state distribution function.
        - rproc: Process model function implementing the stochastic volatility dynamics.
        - dmeas: Measurement model function for the log returns.
        - covars: Covariates used in the model. In this case, the log returns of the
            S&P 500 stock index at the previous time step.
    """
    assert isinstance(sp500, pd.DataFrame)
    return Pomp(
        ys=sp500,
        theta=theta,
        t0=0.0,
        nstep=1,
        dt=None,
        ydim=1,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        covars=covars,
        statenames=statenames,
        par_trans=ParTrans(to_est, from_est),
    )

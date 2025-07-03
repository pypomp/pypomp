import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as random
import jax
from functools import partial
from pypomp.model_struct import RInit, RProc, DMeas
from pypomp.pomp_class import Pomp

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


first_time = sp500.index[0] - 1  # noqa
covars = pd.DataFrame(sp500["y"].values, index=sp500.index)  # noqa
covars.loc[first_time] = 0
covars = covars.sort_index()
covars = covars.rename(columns={0: "y_prev"})


def _rho_transform(x):
    """Transform rho to perturbation scale"""
    return float(jnp.log((1 + x) / (1 - x)))


theta = {
    "mu": float(jnp.log(3.68e-4)),
    "kappa": float(jnp.log(3.14e-2)),
    "theta": float(jnp.log(1.12e-4)),
    "xi": float(jnp.log(2.27e-3)),
    "rho": _rho_transform(-7.38e-1),
    "V_0": float(jnp.log(7.66e-3**2)),
}


@partial(RInit, t0=0.0)
def rinit(theta_, key, covars=None, t0=None):
    V_0 = jnp.exp(theta_[5])
    S_0 = 1105  # Initial price
    return jnp.array([V_0, S_0])


@partial(RProc, step_type="fixedstep", nstep=1)
def rproc(X_, theta_, key, covars, t=None, dt=None):
    V, S = X_
    mu, kappa, theta, xi, rho, V_0 = theta_
    y_prev = covars
    # Transform parameters onto natural scale
    mu = jnp.exp(mu)
    kappa = jnp.exp(kappa)
    theta = jnp.exp(theta)
    xi = jnp.exp(xi)
    rho = -1 + 2 / (1 + jnp.exp(-rho))
    # Wiener process generation (Gaussian noise)
    dZ = random.normal(key)
    dWs = (y_prev - mu + 0.5 * V) / jnp.sqrt(V)
    # dWv with correlation
    dWv = rho * dWs + jnp.sqrt(1 - rho**2) * dZ
    S = S + S * (mu + jnp.sqrt(jnp.maximum(V, 1e-32)) * dWs)
    V = V + kappa * (theta - V) + xi * jnp.sqrt(V) * dWv
    # Feller condition to keep V positive
    V = jnp.maximum(V, 1e-32)
    # Results must be returned as a JAX array
    return jnp.array([V, S]).squeeze()


@DMeas
def dmeas(Y_, X_, theta_, covars=None, t=None):
    V, S = X_
    # Transform mu onto the natural scale
    mu = jnp.exp(theta_[0])
    return jax.scipy.stats.norm.logpdf(Y_, mu - 0.5 * V, jnp.sqrt(V))


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
        ys=sp500, theta=theta, rinit=rinit, rproc=rproc, dmeas=dmeas, covars=covars
    )

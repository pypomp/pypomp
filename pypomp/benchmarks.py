import pandas as pd


import importlib.util


def _check_statsmodels():
    """Check if statsmodels is installed, raising an ImportError if not."""
    if importlib.util.find_spec("statsmodels") is None:
        raise ImportError(
            "The 'statsmodels' package is required for benchmark functions. "
            "You can install it with: pip install pypomp[benchmarks] "
            "or pip install statsmodels directly."
        )


def arma_benchmark(ys: pd.DataFrame, order: tuple[int, int, int] = (1, 0, 1)) -> float:
    """
    Fits an ARIMA model to the data and returns the estimated log-likelihood.

    If 'ys' contains multiple columns, it fits independent ARMA models to each
    column and returns the sum of the log likelihoods.

    Args:
        ys (pd.DataFrame): The observed data.
        order (tuple, optional): The (p, d, q) order of the ARIMA model. Defaults to (1, 0, 1).

    Returns:
        float: The sum of the log-likelihoods from the fitted models.
    """
    _check_statsmodels()
    from statsmodels.tsa.arima.model import ARIMA

    total_llf = 0.0
    for col in ys.columns:
        data = ys[col].dropna()
        if len(data) > 0:
            model = ARIMA(data, order=order)
            # method="innovations_mle" can be faster or we can use default
            res = model.fit()
            total_llf += res.llf

    return float(total_llf)


def negbin_benchmark(ys: pd.DataFrame) -> float:
    """
    Fits an independent Negative Binomial model to the data and returns the log-likelihood.

    If 'ys' contains multiple columns, it fits independent models to each
    column and returns the sum of the log likelihoods.

    Args:
        ys (pd.DataFrame): The observed data.

    Returns:
        float: The sum of the log-likelihoods from the fitted models.
    """
    _check_statsmodels()
    import statsmodels.api as sm
    import numpy as np

    total_llf = 0.0
    for col in ys.columns:
        data = ys[col].dropna()
        if len(data) > 0:
            # Add a constant (intercept) for the mean
            exog = np.ones_like(data)
            model = sm.NegativeBinomial(data, exog)
            res = model.fit(disp=0)
            total_llf += res.llf

    return float(total_llf)

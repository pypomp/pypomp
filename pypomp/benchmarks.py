import pandas as pd
import numpy as np


import importlib.util


def _check_statsmodels():
    """Check if statsmodels is installed, raising an ImportError if not."""
    if importlib.util.find_spec("statsmodels") is None:
        raise ImportError(
            "The 'statsmodels' package is required for benchmark functions. "
            "You can install it with: pip install pypomp[benchmarks] "
            "or pip install statsmodels directly."
        )


def arma_benchmark(
    ys: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 1),
    log_ys: bool = False,
    suppress_warnings: bool = True,
) -> float:
    """
    Fits an ARIMA model to the data and returns the estimated log-likelihood.

    If 'ys' contains multiple columns, it fits independent ARMA models to each
    column and returns the sum of the log likelihoods.

    Args:
        ys (pd.DataFrame): The observed data.
        order (tuple, optional): The (p, d, q) order of the ARIMA model. Defaults to (1, 0, 1).
        log_ys (bool, optional): If True, fits the model to log(y+1). Defaults to False.
        suppress_warnings (bool, optional): If True, suppresses individual warnings from statsmodels
            and issues a summary warning instead. Defaults to True.

    Returns:
        float: The sum of the log-likelihoods from the fitted models.
    """
    _check_statsmodels()
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    total_llf = 0.0

    with warnings.catch_warnings(record=True) as w:
        if suppress_warnings:
            warnings.simplefilter("always")

        for col in ys.columns:
            data = ys[col].dropna()
            if len(data) > 0:
                if log_ys:
                    data = np.log(data + 1)
                model = ARIMA(data, order=order)
                # method="innovations_mle" can be faster or we can use default
                res = model.fit()
                total_llf += res.llf

    if suppress_warnings and len(w) > 0:
        warnings.warn(
            f"arma_benchmark: {len(w)} warnings were produced by statsmodels. "
            "Set suppress_warnings=False to see the raw output.",
            UserWarning,
            stacklevel=2,
        )
    elif not suppress_warnings:
        # Re-issue caught warnings
        for warning in w:
            warnings.warn_explicit(
                message=warning.message,
                category=warning.category,
                filename=warning.filename,
                lineno=warning.lineno,
                source=warning.source,
            )

    return float(total_llf)


def negbin_benchmark(ys: pd.DataFrame, suppress_warnings: bool = True) -> float:
    """
    Fits an independent Negative Binomial model to the data and returns the log-likelihood.

    If 'ys' contains multiple columns, it fits independent models to each
    column and returns the sum of the log likelihoods.

    Args:
        ys (pd.DataFrame): The observed data.
        suppress_warnings (bool, optional): If True, suppresses individual warnings from statsmodels
            and issues a summary warning instead. Defaults to True.

    Returns:
        float: The sum of the log-likelihoods from the fitted models.
    """
    _check_statsmodels()
    import statsmodels.api as sm
    import warnings

    total_llf = 0.0

    with warnings.catch_warnings(record=True) as w:
        if suppress_warnings:
            warnings.simplefilter("always")

        for col in ys.columns:
            data = ys[col].dropna()
            if len(data) > 0:
                # Add a constant (intercept) for the mean
                exog = np.ones_like(data)
                model = sm.NegativeBinomial(data, exog)
                res = model.fit(disp=0)
                total_llf += res.llf

    if suppress_warnings and len(w) > 0:
        warnings.warn(
            f"negbin_benchmark: {len(w)} warnings were produced by statsmodels. "
            "Set suppress_warnings=False to see the raw output.",
            UserWarning,
            stacklevel=2,
        )
    elif not suppress_warnings:
        for warning in w:
            warnings.warn_explicit(
                message=warning.message,
                category=warning.category,
                filename=warning.filename,
                lineno=warning.lineno,
                source=warning.source,
            )

    return float(total_llf)

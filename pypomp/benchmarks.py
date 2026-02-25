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

    def fit_all():
        nonlocal total_llf
        for col in ys.columns:
            data = ys[col].dropna()
            if len(data) > 0:
                if log_ys:
                    data = np.log(data + 1)
                model = ARIMA(data, order=order)
                res = model.fit()
                total_llf += res.llf

    if not suppress_warnings:
        fit_all()
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit_all()

        if len(w) > 0:
            warnings.warn(
                f"arma_benchmark: {len(w)} warnings were produced by statsmodels. "
                "Set suppress_warnings=False to see the raw output.",
                UserWarning,
                stacklevel=2,
            )

    return float(total_llf)


def negbin_benchmark(
    ys: pd.DataFrame, autoregressive: bool = False, suppress_warnings: bool = True
) -> float:
    """
    Fits a Negative Binomial model to the data and returns the log-likelihood.

    If 'ys' contains multiple columns, it fits independent models to each
    column and returns the sum of the log likelihoods.

    Args:
        ys (pd.DataFrame): The observed data.
        autoregressive (bool, optional): If True, fits an AR(1) model where
            Y_n | Y_{n-1} ~ NB(a + b*Y_{n-1}, size). If False (default),
            fits an iid Negative Binomial model.
        suppress_warnings (bool, optional): If True, suppresses individual warnings from statsmodels/optimization
            and issues a summary warning instead. Defaults to True.

    Returns:
        float: The sum of the log-likelihoods from the fitted models.
    """
    _check_statsmodels()
    import warnings

    total_llf = 0.0

    def fit_all():
        nonlocal total_llf
        for col in ys.columns:
            data = ys[col].dropna()
            if len(data) == 0:
                continue

            if not autoregressive:
                import statsmodels.api as sm

                exog = np.ones_like(data)
                model = sm.NegativeBinomial(data, exog)
                res = model.fit(disp=0)
                total_llf += res.llf
            else:
                # AR(1) Case: Y_n | Y_{n-1} ~ NB(a + b*Y_{n-1}, size)
                from scipy.optimize import minimize
                from scipy.stats import nbinom

                y = data.values
                if len(y) < 2:
                    continue
                y_past = y[:-1]
                y_curr = y[1:]

                def neg_log_lik(params):
                    a, b, size = params
                    mu = a + b * y_past
                    if np.any(mu <= 0) or size <= 0:
                        return 1e10
                    # scipy nbinom p = size / (size + mu)
                    p = size / (size + mu)
                    ll_obs = nbinom.logpmf(y_curr, size, p)
                    return -float(np.sum(ll_obs))

                # Use iid NB estimates for starting values
                y_float = y.astype(float)
                mean_y = float(np.mean(y_float))
                var_y = float(max(float(np.var(y_float)), mean_y + 1e-6))
                start_size = mean_y**2 / (var_y - mean_y)
                start = [mean_y * 0.5, 0.5, start_size]
                bounds = [(1e-6, None), (0, None), (1e-6, None)]

                res = minimize(neg_log_lik, start, bounds=bounds, method="L-BFGS-B")
                total_llf += -float(res.fun)

    if not suppress_warnings:
        fit_all()
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit_all()

        if len(w) > 0:
            warnings.warn(
                f"negbin_benchmark: {len(w)} warnings were produced by statsmodels. "
                "Set suppress_warnings=False to see the raw output.",
                UserWarning,
                stacklevel=2,
            )

    return float(total_llf)

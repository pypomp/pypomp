import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)
ys = np.random.poisson(5, 100)

model = ARIMA(ys, order=(1, 0, 1))
res = model.fit()
print("ARMA llf:", res.llf)

model_nb = sm.NegativeBinomial(ys, np.ones_like(ys))
res_nb = model_nb.fit(disp=0)
print("NB llf:", res_nb.llf)

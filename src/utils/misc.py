import pandas as pd
import numpy as np

def C_emp(ts: pd.Series, tau: pd.Timedelta):
    """Computes the realized volatility of time series ts over time scale tau.
    See `Modeling microstructure noise with mutually exciting point processes` paper (https://arxiv.org/pdf/1101.3422.pdf) for more details."""
    T = ts.index[-1] - ts.index[0]
    ts = ts.resample(rule=tau).first().ffill()
    rv = ((ts.diff())**2).sum() / T.total_seconds() # kinda of like quadratic variation basically
    return rv

def range_timedelta(start, end, step, unit):
    """Generates a grid of pd.Timedelta objects between start and end with step size step."""
    timedeltas = range(start, end, step)
    return pd.Series([pd.Timedelta(t, unit=unit) for t in timedeltas])

def C_th(theta):
    mu, alpha, beta = theta
    kappa = alpha/beta
    Lambda = (2*mu)/(1 - kappa)
    k = 1/(1 + kappa)
    gamma = alpha + beta
    C = lambda tau: Lambda*(k**2 + (1 - k**2) * (1 - np.exp(-gamma * tau)) / (gamma * tau))
    return C

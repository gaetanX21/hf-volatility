import pandas as pd
import numpy as np
from scipy.optimize import minimize, curve_fit

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
    phi_l1_norm = alpha/beta
    Lambda = (2*mu)/(1 - phi_l1_norm)
    kappa = 1/(1 + phi_l1_norm)
    gamma = alpha + beta
    C = lambda tau: Lambda*(kappa**2 + (1 - kappa**2) * (1 - np.exp(-gamma * tau)) / (gamma * tau))
    return C

def curve_calibrate(taus, C_emp_values, p0=[0.1,0.1,1]): # TODO: find better initial values
    """Calibrates the curve C to the realized volatility of ts."""
    xdata = taus.dt.total_seconds().values
    ydata = C_emp_values
    
    # difference between curve_fit and minimize? (curve_fit uses non-linear least squares?)
    def func(x, mu, alpha, beta):
        phi_l1_norm = alpha/beta
        Lambda = (2*mu)/(1 - phi_l1_norm)
        kappa = 1/(1 + phi_l1_norm)
        gamma = alpha + beta
        return Lambda*(kappa**2 + (1 - kappa**2) * (1 - np.exp(-gamma * x)) / (gamma * x))
    
    popt, pcov = curve_fit(func, xdata, ydata, p0=[0.1,0.1,1]) # /!\ adding bounds triggers VERY bad fit (why?)
    return popt

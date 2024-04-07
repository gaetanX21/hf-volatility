import pandas as pd
import numpy as np
# Implements various estimators of realized variance as proposed in "A Tale of Two Time Scales: Determining Integrated Volatility With Noisy High-Frequency Data"


def rv_naive(ts: pd.Series, tau: pd.Timedelta, verbose: bool=True):
        n_orig = len(ts)
        ts = ts.resample(tau).first()
        n_new = len(ts)
        if verbose:
            print('\n')
            print(f'Using naive estimator with tau={tau.total_seconds()}s.')
            print(f'{n_orig} --> {n_new} data points.')
            print(f'Used ffill {ts.isna().sum()} times.')
        ts = ts.ffill()
        rv = (ts.diff()**2).sum()
        return rv

def rv_multigrid(ts: pd.Series, tau: pd.Timedelta, K: int, verbose: bool=True):
    # we resample such that t_{i+1}-t_i = tau and then we have K subgrids, such that each subgrid essentially
    # uses tau_prime=tau*K (e.g. G_0 = {t_0, t_{K-1}, t_{2K-1}, ...})
    n_orig = len(ts)
    ts = ts.resample(tau).first()
    n_new = len(ts)
    if verbose:
        print('\n')
        print(f'Using multigrid estimator with tau={tau.total_seconds()}s and K={K}.')
        print(f'{n_orig} --> {n_new} data points.')
        print(f'Used ffill {ts.isna().sum()} times.')
    ts = ts.ffill()
    subgrids = [ts[start::K] for start in range(K)]
    tau_prime = tau * K
    rv_naive_estimators = []
    for subgrid in subgrids:
        rv_naive_est = rv_naive(subgrid, tau_prime, verbose=False)
        rv_naive_estimators.append(rv_naive_est)
    rv_naive_avg = np.mean(rv_naive_estimators)
    return rv_naive_avg


def rv_best(ts: pd.Series, tau: pd.Timedelta, K: int, verbose: bool=True):
    n_orig = len(ts)
    n_new = ts.resample(tau).first().count()
    if verbose:
        print('\n')
        print(f'Using best estimator with tau={tau.total_seconds()}s and K={K}.')
        print(f'{n_orig} --> {n_new} data points.')

    rv_naive_est = rv_naive(ts, tau, verbose=False)
    rv_multi_est = rv_multigrid(ts, tau, K, verbose=False)

    n_bar = (n_new-K+1)/K # why not n_bar=n/K?
    rv_best_est = rv_multi_est - (n_bar/n_new)*rv_naive_est
    return rv_best_est
     
import numpy as np
from scipy.optimize import minimize
from time import time


class HawkesCalibrator:

    """Calibrates the parameters theta=(mu,alpha,beta) given event realisations of a univariate Hawkes process."""

    def __init__(self, events):
        self.events = events
    
    def nll(self, theta):
        # LL formula in red box page 32 at http://lamp.ecp.fr/MAS/fiQuant/ioane_files/HawkesCourseSlides.pdf (P=1 in our case)
        mu, alpha, beta = theta
        kappa = alpha/beta
        t_n = self.events[-1]
        n = len(self.events)
        ll = t_n - mu * t_n

        # NUMPYIFY
        # for i in range(n):
        #     ll = ll - kappa * (1 - np.exp(-beta*(t_n - self.events[i])))
        ll -= kappa * (1 - np.exp(-beta*(t_n - self.events))).sum()

        # recursion here so we can numpyify the loop
        r_array = np.zeros(n)
        for i in range(1, n):
            r_array[i] = np.exp(-beta * (self.events[i] - self.events[i - 1])) * (1 + r_array[i - 1])

        # NUMPYIFY
        # for i in range(n):
        #     ll += np.log(mu + alpha*r_array[i])
        ll += np.log(mu + alpha*r_array).sum()
        
        return -ll
    
    def MLE(self, timeit=True, x0=[1,1,1], method='L-BFGS-B'):
        # TODO: find better x0 (e.g. using the method of moments)
        start = time()
        res = minimize(fun=self.nll, x0=x0, bounds=[(1e-3, None), (1e-3, None), (1e-3, None)], method=method)
        end = time()
        if timeit:
            print("\033[92mCalibration time: ", end - start, "s\033[0m")
        return res.x
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
        """Maximum likelihood estimation of the parameters of a univariate Hawkes process."""
        # TODO: find better x0 (e.g. using the method of moments)
        start = time()
        res = minimize(fun=self.nll, x0=x0, bounds=[(1e-3, None), (1e-3, None), (1e-3, None)], method=method)
        end = time()
        if timeit:
            print("\033[92mCalibration time: ", end - start, "s\033[0m")
        return tuple(res.x)
    

    def M_estimation(self):
        # TODO: implement M-estimation (ie. method of moments)
        pass



class MultiHawkesCalibrator:
    def __init__(self, events, d):
        self.events = events
        self.d = d

    def calibrate(self, initial_guess_flat, T):
        bounds = [(1e-10, None)] * self.d + [(1e-10, None)] * self.d**2 + [(1e-10, None)] * self.d**2
        res = minimize(fun=self.nll_d, args=(T, ), x0=initial_guess_flat, bounds=bounds, method='L-BFGS-B')
        return res.x
    
    def nll_2d(self, theta, T):
        mus = theta[:self.d]
        alphas = theta[self.d:self.d + self.d**2].reshape((self.d, self.d))
        betas = theta[self.d + self.d**2:].reshape((self.d, self.d))
        ll_1 = (1 - mus[0]) * T
        ll_2 = (1 - mus[1]) * T
        for k in range(len(self.events[0])):
            ll_1 = ll_1 - alphas[0, 0] / betas[0, 0] * (1 - np.exp(-betas[0, 0]*(T - self.events[0][k])))

    def nll_d(self, theta, T):
        # source: https://www.ism.ac.jp/editsec/aism/pdf/031_1_0145.pdf
        mus = theta[:self.d]
        alphas = theta[self.d:self.d + self.d**2].reshape((self.d, self.d))
        betas = theta[self.d + self.d**2:].reshape((self.d, self.d))
        n = np.max([len(self.events[i]) for i in range(self.d)])
        ll = np.zeros(self.d)
        for i in range(self.d):
            ll[i] = T - mus[i] * T
            for j in range(self.d):
                for k in range(len(self.events[i])):
                    ll[i] = ll[i] - alphas[i, j] / betas[i, j] * (1 - np.exp(-betas[i, j]*(T - self.events[i][k])))
        
        r_array = np.zeros((self.d, self.d, n))
        for i in range(self.d):
            for j in range(self.d):
                for k in range(1, len(self.events[i])):
                    if i==j:
                        r_array[i, j, k] = np.exp(-betas[i, j] * (self.events[i][k] - self.events[i][k - 1])) * (1 + r_array[i, j, k - 1])
                    else:
                        sum = 0
                        l = 0
                        while l < len(self.events[j]) and self.events[j][l] >= self.events[i][k - 1] and self.events[j][l] < self.events[i][k]:
                            sum += np.exp(-betas[i, j] * (self.events[i][k] - self.events[j][l]))
                            l += 1
                        r_array[i, j, k] = np.exp(-betas[i, j] * (self.events[i][k] - self.events[i][k - 1])) * (1 + r_array[i, j, k - 1]) + sum

        for i in range(self.d):
            for j in range(self.d):
                for k in range(len(self.events[i])):
                    ll[i] += np.log(mus[i] + alphas[i, j]*r_array[i, j, k])  
                    # r_ijk = 0
                    # l = 0
                    # while l < len(events[j]) and events[j][l] < events[i][k]:
                    #     r_ijk += np.exp(-betas[i, j] * (events[i][k] - events[j][l]))
                    #     l += 1
                    # ll[i] = ll[i] + np.log(mus[i] + alphas[i, j]*r_ijk) 

        return -np.sum(ll)
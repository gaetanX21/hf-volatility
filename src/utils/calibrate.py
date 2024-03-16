# whoever you are add comments please
import numpy as np
from scipy.optimize import minimize

class HawkesProcessCalibrator:
    def __init__(self, events):
        self.events = events

    def calibrate(self):
        # source: https://www.ism.ac.jp/editsec/aism/pdf/031_1_0145.pdf
        def nll(theta):
            mu, alpha, beta = theta
            t_n = self.events[-1]
            n = len(self.events)
            ll = - mu * t_n
            for i in range(n):
                ll += alpha/beta * (np.exp(-beta*(t_n - self.events[i])) - 1)
            for i in range(n):
                A_i = sum(np.exp(-beta*(self.events[j] - self.events[i])) for j in range(i+1,n))
                ll += np.log(mu + alpha*A_i)
            return -ll
        
        res = minimize(nll, [1,1,1], bounds=[(0, None), (0, None), (0, None)])
        return res.x
        



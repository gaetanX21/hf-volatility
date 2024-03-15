import numpy as np

class HomogeneousPoissonProcess:
    def __init__(self, rate):
        self.rate = rate

    def simulate(self, T):
        t = 0
        events = np.array([])
        while t < T:
            t += np.random.exponential(1/self.rate)
            if t < T:
                events = np.append(events, t)
        return events

 
class InhomogeneousPoissonProcess:
    def __init__(self, rate_function, rate_majorant):
        self.rate_function = rate_function
        self.rate_majorant = rate_majorant

    def simulate(self, T):
        t_homo = 0
        events = np.array([])
        while t_homo < T:
            e = np.random.exponential(1/self.rate_majorant)
            t_homo += e
            u = np.random.rand()
            ratio = self.rate_function(t_homo)/self.rate_majorant
            if u < ratio and t_homo < T:
                events = np.append(events, t_homo)
        return events
    

class HawkesProcess:
    def __init__(self, mu, alpha, beta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def get_rate(self, events, t):
        return self.mu + self.alpha * (np.exp(-self.beta*(t-events))*(t>events)).sum()

    def simulate(self, T):
        t = 0
        events = np.array([])
        while t < T:
            lambda_bar = self.get_rate(events, t)
            e = np.random.exponential(1/lambda_bar)
            t += e
            u = np.random.rand()
            ratio = self.get_rate(events, t) / lambda_bar
            if u < ratio and t < T:
                events = np.append(events, t)
        return events
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

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
    
    def get_compensator(self, events, t):
        # return self.mu * t + (self.alpha / self.beta) * ((1-np.exp(-self.beta*(t-events))*(t>events))).sum()
        total = self.mu * t
        for t_i in events:
            if t_i<t:
                total += (self.alpha/self.beta) * (1 - np.exp(-self.beta*(t-t_i)))
        return total

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
    
    def plot_QQ(self, events):
        # cf. https://stats.stackexchange.com/questions/492978/ks-test-for-hawkes-process
        compensator_func = lambda t: self.get_compensator(events, t)
        s = np.array([compensator_func(t_i) for t_i in events])
        exp = s[1:] - s[:-1]

        sts.probplot(exp, dist="expon", plot=plt)
        plt.show()


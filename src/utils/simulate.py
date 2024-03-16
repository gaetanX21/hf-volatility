import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import pandas as pd

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


class MultiHawkesProcess:

    # Implements Algo 1 from https://www.math.fsu.edu/~ychen/research/multiHawkes.pdf
    # possible improvement: numpy arrays instead of lists

    def __init__(self, mus, alphas, betas) -> None:
        self.M = len(mus) # number of dimensions
        self.mus = mus
        self.alphas = alphas
        self.betas = betas

    def get_rate(self, m, events, t):
        res = self.mus[m]
        for n in range(self.M):
            for t_i in events[n]:
                if t_i < t:
                    res += self.alphas[m][n] * np.exp(-self.betas[m][n] * (t - t_i))
        return res
    
    def get_rate_sum(self, events, t):
        return sum([self.get_rate(m, events, t) for m in range(self.M)])

    def simulate(self, T):
        s = 0
        events = [[] for i in range(self.M)]
        while s < T:
            lambda_bar = self.get_rate_sum(events, s)
            e = np.random.exponential(1/lambda_bar)
            s += e
            D = np.random.rand()
            ratio = self.get_rate_sum(events, s) / lambda_bar
            if D < ratio:
                k = 0
                new_sum = self.get_rate(k, events, s)
                while D*lambda_bar > new_sum:
                    k += 1
                    new_sum += self.get_rate(k, events, s)
                if s < T:
                    events[k].append(s)

        return events
    
    def plot(self, events, T, n_points=1000):
        T_range = np.linspace(0, T, n_points)
        rates = [[self.get_rate(m, events, t) for t in T_range] for m in range(self.M)]
        fig, axs = plt.subplots(self.M, 1, figsize=(10, 6))
        max_rate = max([max(r) for r in rates])

        for i in range(self.M):
            ax = axs[i]
            ax.plot(T_range, rates[i], label=f'$\\lambda_{i}(t)$', c='blue')
            ax.plot(events[i], [0]*len(events[i]), marker=2, ls='', label=f'events {i}', c='red')
            ax.legend()
            ax.set_ylim(0, max_rate)

        plt.tight_layout()
        plt.show()



class PriceProcess:

    """We simulate the price as X(t)=N_+(t)-N_-(t) where N_+ (resp. N_-) is a Hawkes process with parameters (mu, alpha, beta)
    representing positive (resp. negative) price jumps."""

    def __init__(self, mu, alpha, beta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def simulate_wrong(self, T):
        """Here we simulate the price process by simulating two Hawkes processes and then taking the difference.
        This doesn't work because in practice N_+ excites N_- and vice versa., so they're not self-exciting but they excite each other."""
        hawkes = HawkesProcess(self.mu, self.alpha, self.beta)
        pos_events = hawkes.simulate(T)
        neg_events = hawkes.simulate(T)
        # create time series with index = time and value = price
        time_series = pd.Series(index=np.concatenate(pos_events, neg_events), data=0)
        time_series.loc[pos_events] = 1
        time_series.loc[neg_events] = -1
        time_series = time_series.sort_index()
        time_series = time_series.cumsum()
        return time_series



    
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import pandas as pd


class HomogeneousPoissonProcess:

    """Simulate a homogeneous Poisson process with rate `rate`."""

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
    
    """Simulate an inhomogeneous Poisson process with rate function `rate_function` and majorant `rate_majorant`
    using Lewis and Shedler's algorithm. (https://doi.org/10.1002/nav.3800260304)"""

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

    """Simulates a univariate Hawkes Process with baseline intensity mu and exponential kernel with magnitude
    alpha and decay time beta."""

    def __init__(self, mu, alpha, beta):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def get_rate(self, events, t):
        return self.mu + self.alpha * (np.exp(-self.beta*(np.maximum(t-events,0)))*(t>=events)).sum() # for the same reason as multihawkes, here it's >= and not >
        # we add the max(t-events,0) to avoid overflow in the exp...

    def get_compensator(self, events, t):
        total = self.mu * t
        # for t_i in events:
        #     if t_i<t:
        #         total += (self.alpha/self.beta) * (1 - np.exp(-self.beta*(t-t_i)))

        # numpy is faster
        total += (self.alpha/self.beta) * ((1 - np.exp(-self.beta*(np.maximum(t-events,0))))*(t>events)).sum() # note that here using >= or > doesn't make any difference, and it's smarter to use > since using >= amounts to adding a 0 to the sum
        # we add the max(t-events,0) to avoid overflow in the exp...
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
    
    def plot(self, events, T, n_points=1000):
        T_range = np.linspace(0, T, n_points)
        rates = [self.get_rate(events, t) for t in T_range]
        fig, ax = plt.subplots(figsize=(10, 6))
        max_rate = max(rates)
        ax.plot(T_range, rates, label='$\\lambda(t)$', c='blue')
        ax.plot(events, [0]*len(events), marker=2, ls='', label=f'events', c='red')
        ax.legend()
        ax.set_ylim(0, max_rate)

        plt.tight_layout()
        plt.show()
    
    def plot_QQ(self, events):
        # cf. https://stats.stackexchange.com/questions/492978/ks-test-for-hawkes-process
        compensator_func = lambda t: self.get_compensator(events, t)
        s = np.array([compensator_func(t_i) for t_i in events])
        exp = s[1:] - s[:-1]

        sts.probplot(exp, dist="expon", plot=plt)
        plt.show()


class MultiHawkesProcess:

    """Simulates a multivariate Hawkes Process with baseline intensities mu, matrix of magnitudes alpha and matrix of decay times beta.
    The algorithm used is Algorithm 1 from https://www.math.fsu.edu/~ychen/research/multiHawkes.pdf"""

    def __init__(self, mus, alphas, betas) -> None:
        self.M = len(mus) # number of dimensions
        self.mus = mus
        self.alphas = alphas
        self.betas = betas

    def get_rate(self, m, events, t):
        res = self.mus[m]
        for n in range(self.M):
            # /!\ HERE WE NEED TO USE >= INSTEAD OF >, OTHERWISE WE UNDERESTIMATE THE RATE, WHICH CREATES PROBLEMS
            # (the last dimension of events will be underserved)
            res += self.alphas[m][n] * (np.exp(-self.betas[m][n] * (np.maximum(t - events[n],0))) * (t >= events[n])).sum() # again, it's >= and not >
            # we add the max(t-events,0) to avoid overflow in the exp...
        return res
    
    def get_rate_sum(self, events, t):
        return sum([self.get_rate(m, events, t) for m in range(self.M)])
    
    def get_compensator(self, m, events, t):
        total = self.mus[m] * t
        for n in range(self.M):
            # for t_i in events[n]:
            #     if t_i < t:
            #         total += (self.alphas[m][n] / self.betas[m][n]) * (1 - np.exp(-self.betas[m][n] * (t - t_i)))
            # numpy is faster
            total += (self.alphas[m][n] / self.betas[m][n]) * ((1 - np.exp(-self.betas[m][n] * np.maximum(t - events[n],0)))*(t>events[n])).sum() # note that here using >= or > doesn't make any difference, and it's smarter to use > since using >= amounts to adding a 0 to the sum
            # we add the max(t-events,0) to avoid overflow in the exp...
        return total

    def simulate(self, T):
        s = 0
        events = [np.array([]) for _ in range(self.M)]
        while s < T:
            lambda_bar = self.get_rate_sum(events, s)
            w = np.random.exponential(1/lambda_bar)
            s += w
            D = np.random.rand()
            new_lambda_bar = self.get_rate_sum(events, s)
            if D*lambda_bar <= new_lambda_bar:
                k = 0
                new_sum = self.get_rate(k, events, s)
                while D*lambda_bar > new_sum:
                    k += 1
                    new_sum += self.get_rate(k, events, s)
                if s < T:
                    # events[k].append(s) # numpy is faster
                    events[k] = np.append(events[k], s)

        return events
    
    def plot(self, events, T, n_points=1000):
        T_range = np.linspace(0, T, n_points)
        rates = [[self.get_rate(m, events, t) for t in T_range] for m in range(self.M)]
        fig, axs = plt.subplots(self.M, 1, figsize=(10, 6))
        max_rate = max([max(r) for r in rates])

        if self.M == 1:
            axs = [axs] # simple hack to deal with the case where M=1
        for i in range(self.M):
            ax = axs[i]
            ax.plot(T_range, rates[i], label=f'$\\lambda_{i}(t)$', c='blue')
            ax.plot(events[i], [0]*len(events[i]), marker=2, ls='', label=f'events {i}', c='red')
            ax.legend()
            ax.set_ylim(0, max_rate)

        plt.tight_layout()
        plt.show()

    def plot_QQ(self, events):
        # cf. http://lamp.ecp.fr/MAS/fiQuant/ioane_files/HawkesCourseSlides.pdf page 51
        compensator_funcs = [lambda t: self.get_compensator(m, events, t) for m in range(self.M)]
        s = [np.array([compensator_funcs[m](t_i) for t_i in events[m]]) for m in range(self.M)]
        exp = [s[m][1:] - s[m][:-1] for m in range(self.M)]

        for m in range(self.M):
            sts.probplot(exp[m], dist="expon", plot=plt)
            plt.show()


class PriceProcess:

    """Simulates price as X(t) = N_+(t) - N_-(t) where (N_+, N-) is a bivariate Hawkes process in which N_+ excites N-
    and conversely N- excites N+, though the two processes do not excite themselves."""

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
    
    def simulate(self, T):
        """This time we use coupled Hawkes processes."""
        hawkes = MultiHawkesProcess([self.mu, self.mu], [[0, self.alpha], [self.alpha, 0]], [[0, self.beta], [self.beta, 0]])
        events = hawkes.simulate(T)
        pos_events, neg_events = events
        # create time series with index = time and value = price
        all_events = np.concatenate([pos_events, neg_events])
        time_series = pd.Series(index=all_events)
        time_series.loc[pos_events] = 1
        time_series.loc[neg_events] = -1
        time_series = time_series.sort_index()
        time_series = time_series.cumsum()
        return time_series

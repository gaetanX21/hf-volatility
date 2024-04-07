# Implements various price models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GBM:
    """Model price using geometric brownian motion."""
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, T: float, dt: float, S0: float=100) -> np.ndarray:
        """
        Simulate a price path using brownian motion.
        T and dt are expressed in years. (because mu and sigma are annualized)
        """
        N = int(T/dt)
        Z = np.random.normal(size=N)
        delta_ln_S = (self.mu - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
        ln_S = np.cumsum(delta_ln_S)
        ln_S = np.insert(ln_S, 0, 0)
        S = S0*np.exp(ln_S)
        # turn S into time series
        timestep = pd.Timedelta(dt*365, 'D')
        index = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=N+1, freq=timestep)
        S = pd.Series(S, index=index)
        return S
    
    def plot(self, S: np.ndarray) -> None:
        plt.figure()
        plt.plot(S)
        plt.xlabel('Time')
        plt.ylabel('Price')
        # rotate xticks
        plt.xticks(rotation=45)
        plt.title(f'GBM Price Path ($\\mu={self.mu}, \\sigma={self.sigma}$)')
        plt.show()
    

class Heston:
    """
    Model price using Heston stochastic volatility model.
    
    dS_t/S_t = mu*dt + sqrt(nu_t)*dW_1
    dnu_t = kappa*(theta - nu_t)*dt + sigma*sqrt(nu_t)*dW_2

    T and dt are expressed in years. (because mu and sigma are annualized)
    """
    def __init__(self, mu: float, sigma: float, theta: float, rho: float, kappa: float, nu0: float):
        assert 2*kappa*theta > sigma**2, 'Feller condition not satisfied.'
        self.mu = mu # drift
        self.kappa = kappa # mean reversion speed
        self.theta = theta # long-term mean
        self.sigma = sigma # volatility of volatility
        self.rho = rho # correlation between stock and volatility
        self.nu0 = nu0 # initial variance

    def simulate(self, T: float, dt: float, S0: float=100) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a price path using Heston model."""
        N = int(T/dt)
        Z = np.random.normal(size=N)
        W1 = np.random.normal(size=N)
        W2 = np.sqrt(1-self.rho**2)*Z + self.rho*W1
        nu = np.zeros(N)
        nu[0] = self.nu0
        S = np.zeros(N)
        S[0] = S0
        for i in range(1, N):
            nu[i] = nu[i-1] + self.kappa*(self.theta - nu[i-1])*dt + self.sigma*np.sqrt(nu[i-1]*dt)*W1[i]
            S[i] = S[i-1]*np.exp((self.mu - 0.5*nu[i])*dt + np.sqrt(nu[i]*dt)*W2[i])
        # turn S and nu into time series
        timestep = pd.Timedelta(dt*365, 'D')
        assert timestep >= pd.Timedelta(1, 's'), 'Timestep too small (<1s).'
        index = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=N, freq=timestep)
        # round index to nearest second (to avoid numerical rounding errors in pandas)
        index = index.round('S')
        S = pd.Series(S, index=index)
        nu = pd.Series(nu, index=index)
        return S, nu
    
    def simulate_M(self, M: int, T: float, dt: float, S0: float=100) -> tuple[np.ndarray, np.ndarray]:
        """Simulate M price paths using Heston model."""
        N = int(T/dt)
        Z = np.random.normal(size=(M,N))
        W1 = np.random.normal(size=(M,N))
        W2 = np.sqrt(1-self.rho**2)*Z + self.rho*W1
        nu = np.zeros((M,N))
        nu[:,0] = self.nu0
        S = np.zeros((M,N))
        S[:,0] = S0
        for i in range(1, N):
            nu[:,i] = nu[:,i-1] + self.kappa*(self.theta - nu[:,i-1])*dt + self.sigma*np.sqrt(nu[:,i-1]*dt)*W1[:,i]
            S[:,i] = S[:,i-1]*np.exp((self.mu - 0.5*nu[:,i])*dt + np.sqrt(nu[:,i]*dt)*W2[:,i])
        # turn S and nu into time series
        timestep = pd.Timedelta(dt*365, 'D')
        assert timestep >= pd.Timedelta(1, 's'), 'Timestep too small (<1s).'
        index = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=N, freq=timestep)
        # round index to nearest second (to avoid numerical rounding errors in pandas)
        index = index.round('S')
        all_S = [pd.Series(S[i], index=index) for i in range(M)]
        all_nu = [pd.Series(nu[i], index=index) for i in range(M)]
        return all_S, all_nu
    
    def plot(self, S: np.ndarray, nu: np.ndarray) -> None:
        fig, axs = plt.subplots(2)
        plt.xticks(rotation=45)
        axs[0].plot(S)
        axs[0].set_ylabel('Price') 
        axs[0].tick_params(axis='x', labelrotation=45)   
        axs[1].fill_between(range(len(nu)), nu, color='gray', alpha=0.5)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Volatility')
        axs[1].tick_params(axis='x', labelrotation=45)
        plt.suptitle(f'Heston Price Path ($\\mu={self.mu}, \\kappa={self.kappa}, \\theta={self.theta}, \\sigma={self.sigma}, \\rho={self.rho}, \\nu_0={self.nu0}$)')
        plt.tight_layout()
        plt.show()


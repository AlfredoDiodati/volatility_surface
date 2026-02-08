"""
Tests univariate score driven model simulation and computes its scaling coefficients.
"""
import numpy as np

def simulation_gaussian(nsim: int, a: np.ndarray, alpha:float,
    mu:float, k: np.ndarray, sigma2:float):
    """a: k-1 dimensional"""
    n_components = k.shape[0] - 1
    K = np.diag(k) + np.diag(-k[1:], k=-1)
    x = np.zeros((nsim, n_components))
    theta = np.zeros(n_components)
    a_adj = np.zeros_like(theta)
    e = np.zeros((n_components, nsim))
    e[0] = np.random.standard_normal(nsim)
    sigma = np.sqrt(sigma2)
    theta[0] = mu
    a_adj[1:] = a
    for t in range(1, nsim):
        scaled_score = (e[:,t] * sigma)/(sigma2 ** (1.0 - alpha))
        x[t] = x[t-1] + K @ (theta - x[t-1]) + a_adj * scaled_score + e[:,t] * sigma
    return x

def simulation_t(nsim: int, a: np.ndarray, alpha:float,
    mu:float, k: np.ndarray, sigma2:float, nu:float):
    """a: k-1 dimensional
    Simulation from nonstandard t-distribution is made using decomposition into normal and chi-squared
    """
    idx = k.shape[0]
    K = np.diag(k) + np.diag(-k[1:], k=-1)
    x = np.zeros((nsim, idx))
    theta = np.zeros(idx)
    a_adj = np.zeros_like(theta)
    e = np.zeros((idx, nsim))
    e[0] = sigma2 * np.random.standard_normal(nsim) / np.sqrt( np.random.chisquare(nu, size=nsim) / nu)
    theta[0] = mu
    a_adj[1:] = a
    for t in range(1, nsim):
        scaled_score = (sigma2 * (nu + 3.0)/(nu + 1.0))**alpha * (nu + 1.0)/(nu * sigma2 + e[0,t]**2) * e[0,t]
        x[t] = x[t-1] + K @ (theta - x[t-1]) + a_adj * scaled_score + e[:,t]
    return x

if __name__ == '__main__':
    pass
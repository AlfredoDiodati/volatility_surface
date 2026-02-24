"""
Tests specifications of score driven multifractal
adapted from MSM, and compares to simulated MSM draws
"""

import numpy as np

def _logit(x:np.ndarray):
    return 1/(1.0 + np.exp(-x))

def gas_multifractal(nsim:int, rng:np.random.Generator,
    m0:float, psi_bar: float, k:int, f_initial:np.ndarray,
    gamma_1: float, b:float, alpha:float
    )->tuple[np.ndarray, np.ndarray]:
    """
    psi = log sigma2
    gaussian innovation
    """
    draws = rng.standard_normal(size=nsim)
    draws2 = draws**2
    s = 0.5 * (draws2 - 1.0)
    lm0 = np.log(m0)
    m1 = 2.0 - m0
    m01 = m1 / m0
    lm01 = np.log(m01)
    sigma2 = np.empty(nsim, dtype=float)
    phi = np.exp(- gamma_1 * b ** np.arange(k))
    f_t = f_initial
    score_t = np.zeros(k, dtype=float)
    for t in range(nsim):
        f_t = f_t * phi + alpha * score_t
        z_t = _logit(f_t)
        sigma2[t] = np.exp(psi_bar + k * lm0 + lm01* np.sum(z_t))
        # M_t = m0 *(m01) ** z_t
        score_t = s[t] * z_t*(1.0 - z_t) * lm01
    returns = draws * np.sqrt(sigma2)
    return returns, sigma2

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nsim = 1000
    rng = np.random.default_rng(123)

    m0 = 1.4
    psi_bar = np.log(1e-4)   # unconditional scale for variance
    k = 5
    f_initial = np.zeros(k)

    gamma_1 = 0.7
    b = 2.0
    alpha = 0.25

    returns, sigma2 = gas_multifractal(
        nsim=nsim,
        rng=rng,
        m0=1.5,
        psi_bar=1,
        k=5,
        f_initial=f_initial,
        gamma_1=0.98,
        b=b,
        alpha=alpha,
    )

    # Plot: variance path
    plt.figure(figsize=(10, 4))
    plt.plot(returns)
    plt.title("GAS multifractal simulated variance (sigma2)")
    plt.xlabel("t")
    plt.ylabel("sigma2")
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()


    # Also print quick sanity stats
    (float(np.mean(returns)), float(np.std(returns)), float(np.mean(sigma2)), float(np.median(sigma2)))

"""
GAS multifractal with TRUE binomial MSM grid, without sampling latent z_i,t inside the filter.

Setup:
- Each component i is Bernoulli with prob p_i,t = logit(temperature * f_i,t)
- Number of "high" multipliers J_t = sum_i z_i,t has Poisson-binomial pmf w_t[j], j=0..k
- Variance grid: sigma2(j) = exp(psi_bar) * m0^(k-j) * m1^j, with m1 = 2 - m0
- Likelihood: p(y_t | f_t) = sum_j w_t[j] * N(0, sigma2(j))
- Score: exact derivative of log-likelihood w.r.t f_t via chain rule through w_t

Simulation:
- To generate data with discrete sigma2 support, we sample J_t ~ w_t (this is part of the DGP).
- This does NOT break GAS, because the recursion is deterministic given y_t and f_t.
"""

from __future__ import annotations
import numpy as np


def _logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normal_pdf_zero_mean(y: float, var: np.ndarray) -> np.ndarray:
    var = np.asarray(var, dtype=float)
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-0.5 * (y * y) / var)


def _poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
    """
    w[j] = P(sum_i z_i = j), z_i ~ Bernoulli(p_i), independent.
    DP convolution, O(k^2).
    """
    p = np.asarray(p, dtype=float)
    k = int(p.shape[0])
    w = np.zeros(k + 1, dtype=float)
    w[0] = 1.0
    for i in range(k):
        pi = float(p[i])
        w[1 : i + 2] = w[1 : i + 2] * (1.0 - pi) + w[0 : i + 1] * pi
        w[0] *= (1.0 - pi)
    return w


def _leave_one_out_q_from_full_pmf(w: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Compute leave-one-out Poisson-binomial coefficients without recomputing pmf per i.

    Let:
      W(x) = prod_i ((1-p_i) + p_i x) = sum_{j=0}^k w[j] x^j
      Q_i(x) = W(x) / ((1-p_i) + p_i x) = sum_{j=0}^{k-1} q_i[j] x^j

    We obtain q_i by polynomial division recurrence:
      w[j] = (1-p_i) q_i[j] + p_i q_i[j-1], with q_i[-1]=0

    Returns Q with shape (k, k): Q[i, j] = q_i[j] for j=0..k-1
    Total cost O(k^2).
    """
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    k = int(p.shape[0])

    Q = np.zeros((k, k), dtype=float)
    for i in range(k):
        a = 1.0 - float(p[i])
        b = float(p[i])
        if a < 1e-15:
            a = 1e-15  # guard, avoids division blowup if p[i] ~ 1

        qi = Q[i]
        qi[0] = w[0] / a
        for j in range(1, k):
            qi[j] = (w[j] - b * qi[j - 1]) / a
    return Q


def simulation(
    nsim: int,
    rng: np.random.Generator,
    m0: float,
    psi_bar: float,
    k: int,
    f_initial: np.ndarray,
    gamma_1: float,
    b: float,
    alpha: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    - returns: simulated y_t
    - sigma2: simulated discrete variance sigma2(J_t) on MSM grid

    Notes
    - gamma structure matches MSM hierarchy: gamma_i = 1 - (1-gamma_1) * b^{-i}
      then phi_i = 1 - gamma_i
    """
    nsim = int(nsim)
    k = int(k)

    f_t = np.asarray(f_initial, dtype=float).copy()
    if f_t.shape != (k,):
        raise ValueError("f_initial must have shape (k,)")

    m0 = float(m0)
    psi_bar = float(psi_bar)
    gamma_1 = float(gamma_1)
    b = float(b)
    alpha = float(alpha)
    temperature = float(temperature)

    # variance grid over j = 0..k
    j = np.arange(k + 1, dtype=float)
    m_high = m0
    m_low = 2.0 - m0
    sigma2_grid = np.exp(psi_bar) * (m_low ** (k - j)) * (m_high ** j)

    # MSM hierarchy for component persistence
    idx = np.arange(1, k + 1, dtype=float)
    phi = 1.0 - (1.0 - gamma_1) * (b ** (-idx))


    returns = np.empty(nsim, dtype=float)
    sigma2 = np.empty(nsim, dtype=float)

    score_t = np.zeros(k, dtype=float)

    for t in range(nsim):
        # deterministic GAS recursion
        f_t = phi * f_t + alpha * score_t

        p = _logit(temperature * f_t)

        # Poisson-binomial mixture weights on j-grid
        w = _poisson_binomial_pmf(p)

        # simulate discrete j_t from w (DGP), then y_t ~ N(0, sigma2(j_t))
        u = float(rng.random())
        cdf = np.cumsum(w)
        jt = int(np.searchsorted(cdf, u, side="right"))
        if jt > k:
            jt = k
        sig2_t = float(sigma2_grid[jt])
        y_t = float(rng.standard_normal()) * float(np.sqrt(sig2_t))

        returns[t] = y_t
        sigma2[t] = sig2_t

        # exact mixture likelihood and exact score
        dens = _normal_pdf_zero_mean(y_t, sigma2_grid)  # length k+1
        mix = float(np.dot(w, dens))
        if mix < 1e-300:
            mix = 1e-300

        # leave-one-out polynomials Q_i to get dw_j/dp_i efficiently
        Q = _leave_one_out_q_from_full_pmf(w, p)  # shape (k,k), q_i[0..k-1]

        # score_i = d/df_i log mix = (1/mix) * (d mix/dp_i) * dp_i/df_i
        # dp_i/df_i = p_i(1-p_i)*temperature
        for i in range(k):
            qi = Q[i]  # length k

            # dw/dp_i:
            # dw[0] = -q[0]
            # dw[j] = q[j-1] - q[j] for j=1..k-1
            # dw[k] = q[k-1]
            dw = np.empty(k + 1, dtype=float)
            dw[0] = -qi[0]
            if k > 1:
                dw[1:k] = qi[0 : k - 1] - qi[1:k]
            dw[k] = qi[k - 1]

            d_mix_dp = float(np.dot(dw, dens))
            score_t[i] = (d_mix_dp / mix) * p[i] * (1.0 - p[i]) * temperature

    return returns, sigma2

"""Tests that simulated data in the gaussian state space model

Same simulation factors as Zou, Lin and Lucas (2025)
"""

import numpy as np
from ..models import ss

def main():
    moneyness = np.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5])
    ttmaturity = np.array([10, 50, 100, 180], dtype=float) / 255.0
    m_grid, ttm_grid = np.meshgrid(moneyness, ttmaturity)
    Mt = np.stack(np.ones(m_grid.size), m_grid.ravel(), ttm_grid.ravel())
    p = Mt.shape[1]

    sigma2_eps = 0.5
    B = np.diag(np.array([0.98, 0.93, 0.90]))
    H = np.eye(p) * sigma2_eps
    Q = np.diag(np.array([0.001, 0.005, 0.004]))

    params = {
        "B": B,
        "Q_param": Q,
        "H_param": B,
    }

if __name__ == "__main__":
    main()
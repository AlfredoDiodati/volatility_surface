"""Tests that simulated data in the gaussian state space model

Same simulation factors as Zou, Lin and Lucas (2025)
"""

import pickle
import numpy as np
from models import ss

def main():
    moneyness = np.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5])
    ttmaturity = np.array([10, 50, 100, 180], dtype=float) / 255.0

    m_grid, ttm_grid = np.meshgrid(moneyness, ttmaturity, indexing="xy")
    Nt = m_grid.size
    Mt = np.column_stack((
        np.ones(Nt, dtype=float),
        m_grid.ravel(),
        ttm_grid.ravel()))
    p = Mt.shape[1]
    horizon_sim = 3500

    sigma2_eps = 0.5
    B = np.diag(np.array([0.98, 0.93, 0.90]))
    Q = np.diag(np.array([0.001, 0.005, 0.004]))
    H = np.eye(Nt) * sigma2_eps
    bar_beta = np.array([0.37, 0.81, 0.24])

    identity_mat = np.eye(p)
    R = identity_mat.copy()

    a0 = bar_beta
    P0 = Q.copy()
    T0 = (identity_mat - B) @ bar_beta + B @ a0
    Z0 = Mt.copy()
    covariates = np.broadcast_to(Mt, (horizon_sim, Nt, p)).copy()
    params = {
        "B": B,
        "Q_param": Q,
        "H_param": H,
        "bar_beta": bar_beta,
        "covariates": covariates,
        "a": np.array([a0]),
        "P": np.array([P0]),
        "Z": np.array([Z0]),
        "T": np.array([T0]),
        "Q": np.array([Q]),
        "H": np.array([H]),
        "R": np.array([R]),
    }
    draw = ss.simulation(params, horizon_sim, npaths=1)
    y = draw["y"]

    carry0 = (a0, P0, Z0, T0, H, R, Q, 0)
    fitted = ss.fit(y, covariates, params, carry0)
    with open("test/simulation_params.pkl", "wb") as file_handle:
        pickle.dump(fitted, file_handle)
if __name__ == "__main__":
    main()
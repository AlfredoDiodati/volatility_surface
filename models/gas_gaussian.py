"""Base for the implementation of the IV models of Zou, Lin and Lucas (2025)

TODO: generalize to nonbucketed data (covariance in filter, storing eps)
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv

def _filter(logIV:np.ndarray, M:np.ndarray, 
        dates:np.ndarray, dates_unique:np.ndarray, 
        params:dict)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recursion over time of the model.
    dates: array of dates as in the dataset
    dates_unique: array of unique values for dates
    """
    p = B.shape[0]
    B = params["B"]
    C = params["C"]
    bar_beta = params["bar_beta"]
    H = params["H"]
    I = np.eye(p)
    const = (I - B) @ bar_beta
    beta_t = bar_beta

    betas = np.empty_like((B.shape[0], p))
    eps = np.empty_like(betas)
    Hadjs = []

    def beta_update(beta_t:np.ndarray,
        eps_t:np.ndarray, M_t:np.ndarray)->np.ndarray:
        """One step of the filter"""
        A = I
        Hadj = H + M_t @ C @ M_t.T
        Hinv = inv(Hadj)
        scaled_score = A @ inv(M_t.T @ Hinv @ M_t) @ M_t.T @ Hinv @ eps_t
        return const + B @ beta_t + scaled_score, Hadj
    
    for i,t in enumerate(dates_unique):
        idx = dates == t
        logIV_t = logIV[idx]
        M_t = M[idx]
        eps_t = logIV_t - M_t @ beta_t
        beta_t, Hadj_t = beta_update(beta_t, eps_t, M_t)
        betas[i] = beta_t
        eps[i] = eps_t
        Hadjs.append(Hadj_t)
    
    return (betas, eps, np.array(Hadjs))
    
def _fit(data:pd.DataFrame) -> dict:
    """
    correction: function to add back possible constant terms from the likelihood that are not used in optimization
    dates in the dataset must be in YYYYMMDD format to work properly
    """
    logIV = data["logIV"].to_numpy()
    dates = data["DATE"].to_numpy()
    M = data.drop(columns=["DATE", "logIV"])
    dates_unique = np.unique(dates)


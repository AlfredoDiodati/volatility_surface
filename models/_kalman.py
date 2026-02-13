"""Kalman filter implementation.
Supports score-driven in-mean models implementations where the state noise is set to 0.

The actual implementation of the state components needs to return them in matrix form, even when they are vectors or scalars,
so that internal shapes of the filter re

Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

import numpy as np

def Filter(data:np.ndarray, H:np.ndarray, Z:callable | np.ndarray | float, T:callable, R:callable,
        Q: np.ndarray | float = 0.0)->dict:
    """If model is non a state space, leave Q = 0"""

    def _step(yt, Zt, at, Tt, Pt):
        vt = yt - Zt @ at
        L = xp.linalg.cholesky(F_t)
        tmp = xp.linalg.solve(L, Zt @ Pt)
        Kt = xp.linalg.solve(L.T, tmp).T

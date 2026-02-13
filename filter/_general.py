"""General case implementation of the Kalman filter.

Allowes for state variables to have different shapes overtime.
Supports score-driven in-mean models implementations where the state noise is set to 0.

The actual implementation of the state components needs to return them in matrix form, even when they are vectors or scalars,
so that internal shapes of the filter remain consistent

Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

import filter._backend as _backend
from filter._backend import ndarray
xp = _backend.Backend.xp
scan = _backend.Backend.scan

# TODO: caching behaviour numpy vs jax, allow for changing shape

def Filter(data: ndarray, dynamics:callable)->dict:
    """Kalman Filter implementation

    Args:
        data (ndarray): data in compatible ndarray
        dynamics (callable): function that specifies Zt, Tt, Rt and Qt

    Returns:
        dict: _description_
    """
    def _step(carry, yt):
        at, Pt = carry
        Zt, Tt, Ht, Rt, Qt = dynamics(yt, at, Pt)
        vt = yt - Zt @ at
        ZP = Zt @ Pt
        Ft = ZP @ Zt.T + Ht
        L = xp.linalg.cholesky(Ft)
        tmp = xp.linalg.solve(L, ZP)
        Kt = xp.linalg.solve(L.T, tmp).T
        atp1 = Tt @ at + Kt @ vt
        Ptp1 = Tt @ Pt @ Tt.T + Rt @ Qt @ Rt.T - Kt @ ZP
        return (atp1, Ptp1)

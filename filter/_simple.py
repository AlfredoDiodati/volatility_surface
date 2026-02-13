"""Simple implementation of the Kalman filter.

Fixed state shape.

Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

import filter._backend as _backend
from filter._backend import ndarray
xp = _backend.Backend.xp
scan = _backend.Backend.scan

def _filter(data: ndarray, dynamics:callable, carry0 = (0.0, 10e7))->dict:
    """Kalman Filter implementation

    Args:
        data (ndarray): data in compatible ndarray
        dynamics (callable): function that specifies Zt, Tt, Rt and Qt
        carry0 (tuple[float, float]): initial state prediction and variance

    """

    def _step(carry, yt):
        """we carry forward at and Pt for prediction and filter, sum """
        at_pred, Pt_pred = carry
        Zt, Tt, Ht, Rt, Qt = dynamics(yt, at_pred, Pt_pred)
        vt = yt - Zt @ at_pred
        ZP = Zt @ Pt_pred
        Ft = ZP @ Zt.T + Ht
        L = xp.linalg.cholesky(Ft)
        tmp = xp.linalg.solve(L, ZP)
        Kt = xp.linalg.solve(L.T, tmp).T
        at_filt = Tt @ at_pred + Kt @ vt
        Ptp1 = Tt @ Pt_pred @ Tt.T + Rt @ Qt @ Rt.T - Kt @ ZP
        Linv_v = xp.linalg.solve(L, vt)
        quad_t = Linv_v.T @ Linv_v
        logdet_t = 2.0 * xp.sum(xp.log(xp.diag(L)))
        new_carry = (at_filt, Ptp1)
        store_timet = {"a": at_filt,"P": Ptp1,"Z": Zt,"T": Tt,
            "H": Ht,"R": Rt,"Q": Qt,"v": vt,
            "F": Ft,"logdetF": logdet_t,"quad": quad_t,}
        return new_carry, store_timet

    _, ll_terms = scan(_step, carry0, data)

    return ll_terms 


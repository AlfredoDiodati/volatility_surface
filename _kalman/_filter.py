"""Simple implementation of the Kalman filter.

Fixed state shape.

Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

from _backend._np import scan
import numpy as np

def _filter(data: np.ndarray, dynamics:callable, params:dict, carry0)->dict:
    """Kalman Filter implementation

    Args:
        data (ndarray): data in compatible ndarray
        dynamics (callable): function that specifies Zt, Tt, Rt and Qt
        params (dict): parameters of the model
        carry0 (tuple[float, float]): initial state prediction and variance

    """

    def _step(carry, yt):
        """we carry forward at and Pt for prediction and filter, sum """
        at_pred, Pt_pred, Zt, Tt, Ht, Rt, Qt, idx= carry
        Zt, Tt, Ht, Rt, Qt = dynamics(yt, at_pred, Pt_pred, params, Zt, Tt, Ht, Rt, Qt, idx)
        vt = yt - Zt @ at_pred
        ZP = Zt @ Pt_pred
        Ft = ZP @ Zt.T + Ht
        L = np.linalg.cholesky(Ft)
        tmp = np.linalg.solve(L, ZP)
        Kt = np.linalg.solve(L.T, tmp).T
        at_filt = Tt @ at_pred + Kt @ vt
        Ptp1 = Tt @ Pt_pred @ Tt.T + Rt @ Qt @ Rt.T - Kt @ ZP
        Linv_v = np.linalg.solve(L, vt)
        quad_t = Linv_v.T @ Linv_v
        logdet_t = 2.0 * np.sum(np.log(np.diag(L)))
        idx +=1
        new_carry = (at_filt, Ptp1, Zt, Tt, Ht, Rt, Qt, idx)
        store_timet = {"a": at_filt,"P": Ptp1,"Z": Zt,"T": Tt,
            "H": Ht,"R": Rt,"Q": Qt,"v": vt,
            "F": Ft,"logdetF": logdet_t,"quad": quad_t,}
        return new_carry, store_timet

    _, ll_terms = scan(_step, carry0, data)

    return ll_terms 


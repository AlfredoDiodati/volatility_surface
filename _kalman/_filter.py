"""Simple implementation of the Kalman filter.

Fixed state shape.

Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

import numpy as np
from _backend._np import scan
from scipy.optimize import minimize, approx_fprime

def _filter(data: np.ndarray, dynamics:callable, params:dict, carry0:tuple)->dict:
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
        Kt = Tt @ (np.linalg.solve(L.T, tmp).T)
        at_filt = Tt @ at_pred + Kt @ vt
        Ptp1 = Tt @ Pt_pred @ Tt.T + Rt @ Qt @ Rt.T - Kt @ Ft @ Kt.T
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

def _loglikelihood(filter_output:dict):
    """Without constant term"""
    return -0.5 * np.sum(filter_output["logdetF"] + filter_output["quad"])

def _fit(data: np.ndarray, initial_guess: dict, covariates:np.ndarray | None, carry_initial:tuple,
    _dynamics:callable, _link:callable | None = None, _invlink: callable | None = None,
    opt_options:dict | None = None)->dict:
    """
    Args:
        data (np.ndarray)
        params (dict)
        covariates (np.ndarray): _description_
        _dynamics (callable): _description_
        _link (callable | None, optional): function that maps uncostrained, ndarray parameters to
        constrained space and returns them in a dictionary. Defaults to None.
        _invlink (callable | None, optional): inverse of _link. Defaults to None.
    """
    if _link is None: _link = lambda x: x
    if _invlink is None: _invlink = lambda x: x
    initial_guess["covariates"] = covariates
    unc_params = _invlink(initial_guess)
    def _criterion(params):
        constr_params = _link(params)
        kf = _filter(data, _dynamics, constr_params, carry_initial)
        return  - _loglikelihood(kf)
    res = minimize(_criterion, unc_params, options=opt_options, method="BFGS")
    unc_params = res.x
    params = _link(unc_params)
    kf = _filter(data, _dynamics, params, carry_initial)
    out = {
        "loglikelihood": - res.fun,
        "niter": res.nit,
        "is_converged": res.success,
        "gradient": res.jac,
        "hessian_inv": res.hess_inv
    }

    return params | kf | out
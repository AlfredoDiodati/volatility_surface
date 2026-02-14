import numpy as np
from _kalman._filter import _filter as kfilter
from _kalman._filter import _loglikelihood

def _dynamics(_y, _a, _P, params, _Z, bt, H, identity_mat, Q, idx)->dict:
    p = bt.shape[0]
    Mt = params["covariates"][idx * p : (1 + idx)*p, :]
    bar_beta = params["bar_beta"]
    B = params["B"]
    Z = Mt
    T = (identity_mat - B) @ bar_beta + B @ bt
    return Z, T, H, identity_mat, Q

def fit(data: np.ndarray, params: dict, covariates:np.ndarray):
    params["covariates"] = covariates
    def _criterion(params):
        kf = kfilter(data, _dynamics, params)
        return _loglikelihood(kf)
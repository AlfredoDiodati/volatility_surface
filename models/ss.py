import numpy as np
from models._kalman import _filter as kfilter
from models._kalman import _simulation

def _dynamics(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx)->dict:
    p = bt.shape[0]
    Q = params["Q_param"]
    H = params["H_param"]
    Mt = params["covariates"][idx * p : (1 + idx)*p, :]
    bar_beta = params["bar_beta"]
    B = params["B"]
    Z = np.where(np.isnan(y), 0.0, Mt)
    T = (identity_mat - B) @ bar_beta + B @ bt
    return Z, T, H, identity_mat, Q

def simulation(fit_output, nsim, npaths):
    return _simulation(fit_output, nsim, _dynamics, npaths)
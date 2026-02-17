import numpy as np
from models._kalman import _simulation, _fit

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

def fit(data:np.ndarray, covariates:np.ndarray, initial_guess:dict, initialization:tuple, opt_options:dict):
    
    kQ = (initial_guess["Q_param"]).shape[0]
    kH = (initial_guess["H_param"]).shape[0]

    def _cholesky_setup(dimension_of_matrix:int):
        idx = np.tril_indices(dimension_of_matrix)
        d = np.diag_indices(dimension_of_matrix)
        cholesky = np.zeros((dimension_of_matrix,dimension_of_matrix), dtype=float)
        return (cholesky, d, idx)
    
    choleskyQ, dQ, idxQ = _cholesky_setup(kQ)
    choleskyH, dH, idxH = _cholesky_setup(kH)

    def _link_covariance(cholesky, vector, idx, d):
        cholesky.fill(0.0)
        cholesky[idx] = vector
        cholesky[d] = np.exp(cholesky[d])
        return cholesky @ cholesky.T

    def _link_stable_matrix(B_unconstrained: np.ndarray, n: int) -> np.ndarray:
        B_unconstrained = np.asarray(B_unconstrained, dtype=float)
        number_of_q_parameters = n * n
        number_of_upper_triangular_parameters = n * (n - 1) // 2
        unconstrained_matrix_for_q = B_unconstrained[:number_of_q_parameters].reshape(n, n)
        orthogonal_matrix_q = np.linalg.qr(unconstrained_matrix_for_q, mode="reduced")[0]
        constrained_schur_matrix = np.empty((n, n), dtype=float)
        constrained_schur_matrix.fill(0.0)
        upper_row_indices, upper_col_indices = np.triu_indices(n, k=1)
        start_upper = number_of_q_parameters
        end_upper = start_upper + number_of_upper_triangular_parameters
        constrained_schur_matrix[upper_row_indices, upper_col_indices] = B_unconstrained[start_upper:end_upper]
        start_diag = end_upper
        constrained_schur_matrix[np.arange(n), np.arange(n)] = np.tanh(B_unconstrained[start_diag:start_diag + n])
        tmp = orthogonal_matrix_q @ constrained_schur_matrix
        return tmp @ orthogonal_matrix_q.T

    def _link(unconstrained_params:np.ndarray)->dict:
        unc_H = unconstrained_params[:kH]
        unc_Q = unconstrained_params[kH:kQ]
        H = _link_covariance(choleskyH, unc_H, idxH, dH)
        Q = _link_covariance(choleskyQ, unc_Q, idxQ, dQ)
        return {"Q_param":Q,"H_param": H}

    def _invlink_covariance(covariance: np.ndarray):
        L = np.linalg.cholesky(covariance)
        i, j = np.tril_indices(L.shape[0])
        v = L[i, j]
        v[np.cumsum(np.arange(1, L.shape[0] + 1)) - 1] = np.log(v[np.cumsum(np.arange(1, L.shape[0] + 1)) - 1])
        return v

    def _invlink(constrained_params:dict): 
        H = constrained_params["H_param"]
        Q = constrained_params["Q_param"]
        uncH = _invlink_covariance(H)
        uncQ = _invlink_covariance(Q)
        params = np.stack(uncH, uncQ)
        return params

    return _fit(data, initial_guess, covariates, initialization, _dynamics, _link, _invlink, opt_options)

def simulation(fit_output, nsim, npaths):
    return _simulation(fit_output, nsim, _dynamics, npaths)
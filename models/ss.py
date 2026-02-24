import numpy as np
from models._kalman import _simulation, _fit

def _dynamics(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx)->dict:
    Q = params["Q_param"]
    H = params["H_param"]
    Mt = params["covariates"][idx]
    bar_beta = params["bar_beta"]
    B = params["B"]
    Z = Mt.copy()
    Z[np.isnan(y), :] = 0.0
    T = B @ bt
    cache = params.setdefault("_cache", {})
    ct = cache.get("last_term")
    if ct is None:
        ct = (identity_mat - B) @ bar_beta
        cache["ct"] = ct
    return Z, T, H, identity_mat, Q, 0.0, ct

def fit(data:np.ndarray, covariates:np.ndarray, initial_guess:dict, initialization:tuple, opt_options:dict | None = None):
    
    p = (initial_guess["Q_param"]).shape[0]
    pH = (initial_guess["H_param"]).shape[0]

    def _cholesky_setup(dimension_of_matrix:int):
        idx = np.tril_indices(dimension_of_matrix)
        d = np.diag_indices(dimension_of_matrix)
        cholesky = np.zeros((dimension_of_matrix,dimension_of_matrix), dtype=float)
        return (cholesky, d, idx)
    
    choleskyQ, dQ, idxQ = _cholesky_setup(p)
    choleskyH, dH, idxH = _cholesky_setup(pH)

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

    def _link(unconstrained_params: np.ndarray) -> dict:
        len_uncH = pH * (pH + 1) // 2
        len_uncQ = p * (p + 1) // 2
        len_uncB = p * p + (p * (p - 1) // 2) + p

        start_H = 0
        end_H = start_H + len_uncH
        start_Q = end_H
        end_Q = start_Q + len_uncQ
        start_B = end_Q
        end_B = start_B + len_uncB
        start_beta = end_B

        unc_H = unconstrained_params[start_H:end_H]
        unc_Q = unconstrained_params[start_Q:end_Q]
        unc_B = unconstrained_params[start_B:end_B]
        bar_beta = unconstrained_params[start_beta:]

        H = _link_covariance(choleskyH, unc_H, idxH, dH)
        Q = _link_covariance(choleskyQ, unc_Q, idxQ, dQ)
        B = _link_stable_matrix(unc_B, p)

        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta}

    def _invlink_covariance(covariance: np.ndarray):
        L = np.linalg.cholesky(covariance)
        i, j = np.tril_indices(L.shape[0])
        v = L[i, j]
        v[np.cumsum(np.arange(1, L.shape[0] + 1)) - 1] = np.log(v[np.cumsum(np.arange(1, L.shape[0] + 1)) - 1])
        return v

    def _invlink_stable_matrix(B_constrained: np.ndarray, n: int) -> np.ndarray:
        B_constrained = np.asarray(B_constrained, dtype=float)
        _, orthogonal_matrix_q = np.linalg.eigh(B_constrained)
        constrained_schur_matrix = orthogonal_matrix_q.T @ B_constrained @ orthogonal_matrix_q
        upper_row_indices, upper_col_indices = np.triu_indices(n, k=1)
        upper_triangular_part = constrained_schur_matrix[upper_row_indices, upper_col_indices]
        diagonal_part = np.arctanh(np.clip(np.diag(constrained_schur_matrix), -0.999999, 0.999999))
        unconstrained_matrix_for_q = orthogonal_matrix_q
        return np.concatenate([
            unconstrained_matrix_for_q.reshape(n * n),
            upper_triangular_part, diagonal_part])

    def _invlink(constrained_params: dict):
        H = constrained_params["H_param"]
        Q = constrained_params["Q_param"]
        B = constrained_params["B"]
        bar_beta = constrained_params["bar_beta"]

        uncH = _invlink_covariance(H)
        uncQ = _invlink_covariance(Q)
        uncB = _invlink_stable_matrix(B, p)

        params = np.concatenate([uncH, uncQ, uncB, bar_beta])
        return params
    
    return _fit(data, initial_guess, covariates, initialization, _dynamics, _link, _invlink, opt_options)

def simulation(fit_output, nsim, npaths):
    return _simulation(fit_output, nsim, _dynamics, npaths)
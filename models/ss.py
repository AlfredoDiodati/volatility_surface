import jax
import jax.numpy as np
from _kalman import _simulation, _fit

def _dynamics(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx) -> dict:
    Q = params["Q_param"]
    H = params["H_param"]
    Mt = jax.lax.dynamic_index_in_dim(params["covariates"], idx, axis=0, keepdims=False)
    B = params["B"]
    Z = np.asarray(Mt, dtype=float)
    T = B @ bt
    return Z, T, H, identity_mat, Q, 0.0, params["ct"]

def fit(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    initialization: tuple,
    opt_options: dict | None = None,):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    p = (initial_guess["Q_param"]).shape[0]
    pH = (initial_guess["H_param"]).shape[0]
    def _cholesky_setup(dimension_of_matrix: int):
        idx = np.tril_indices(dimension_of_matrix)
        d = np.diag_indices(dimension_of_matrix)
        return (d, idx)
    dQ, idxQ = _cholesky_setup(p)
    dH, idxH = _cholesky_setup(pH)
    def _link_covariance(vector, idx, d, dimension_of_matrix: int):
        cholesky = np.zeros((dimension_of_matrix, dimension_of_matrix), dtype=float)
        cholesky = cholesky.at[idx].set(vector)
        cholesky = cholesky.at[d].set(np.exp(cholesky[d]))
        return cholesky @ cholesky.T

    def _link_stable_matrix(B_unconstrained: np.ndarray, n: int) -> np.ndarray:
        B_unconstrained = np.asarray(B_unconstrained, dtype=float)
        number_of_q_parameters = n * n
        number_of_upper_triangular_parameters = n * (n - 1) // 2
        unconstrained_matrix_for_q = B_unconstrained[:number_of_q_parameters].reshape(n, n)
        orthogonal_matrix_q = np.linalg.qr(unconstrained_matrix_for_q, mode="reduced")[0]
        constrained_schur_matrix = np.zeros((n, n), dtype=float)
        upper_row_indices, upper_col_indices = np.triu_indices(n, k=1)
        start_upper = number_of_q_parameters
        end_upper = start_upper + number_of_upper_triangular_parameters
        constrained_schur_matrix = constrained_schur_matrix.at[upper_row_indices, upper_col_indices].set(
            B_unconstrained[start_upper:end_upper])
        start_diag = end_upper
        constrained_schur_matrix = constrained_schur_matrix.at[np.arange(n), np.arange(n)].set(
            np.tanh(B_unconstrained[start_diag : start_diag + n]))
        tmp = orthogonal_matrix_q @ constrained_schur_matrix
        return tmp @ orthogonal_matrix_q.T

    def _link(unconstrained_params: np.ndarray) -> dict:
        unconstrained_params = np.asarray(unconstrained_params, dtype=float)

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
        H = _link_covariance(unc_H, idxH, dH, pH)
        Q = _link_covariance(unc_Q, idxQ, dQ, p)
        B = _link_stable_matrix(unc_B, p)
        ct = (np.eye(B.shape[0]) - B) @ bar_beta
        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct}

    def _invlink_covariance(covariance: np.ndarray):
        covariance = np.asarray(covariance, dtype=float)
        L = np.linalg.cholesky(covariance)
        i, j = np.tril_indices(L.shape[0])
        v = L[i, j]
        n = L.shape[0]
        diag_positions = np.cumsum(np.arange(1, n + 1)) - 1
        v = v.at[diag_positions].set(np.log(v[diag_positions]))
        return v

    def _invlink_stable_matrix(B_constrained: np.ndarray, n: int) -> np.ndarray:
        B_constrained = np.asarray(B_constrained, dtype=float)
        _, orthogonal_matrix_q = np.linalg.eigh(B_constrained)
        constrained_schur_matrix = orthogonal_matrix_q.T @ B_constrained @ orthogonal_matrix_q
        upper_row_indices, upper_col_indices = np.triu_indices(n, k=1)
        upper_triangular_part = constrained_schur_matrix[upper_row_indices, upper_col_indices]
        diagonal_part = np.arctanh(np.clip(np.diag(constrained_schur_matrix), -0.999999, 0.999999))
        unconstrained_matrix_for_q = orthogonal_matrix_q
        return np.concatenate(
            [unconstrained_matrix_for_q.reshape(n * n), upper_triangular_part, diagonal_part]
        )

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

def simulation(fit_output, nsim, npaths, key: jax.Array):
    if "ct" not in fit_output or fit_output["ct"] is None:
        fit_output["ct"] = (np.eye(fit_output["B"].shape[0]) - fit_output["B"]) @ fit_output["bar_beta"]
    return _simulation(fit_output, nsim, _dynamics, npaths, key)
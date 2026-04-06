import jax
import jax.numpy as np
from models._kalman import _simulation, _fit

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

    def _link_stable_matrix(B_unconstrained: np.ndarray, n: int) -> np.ndarray:
        B_unconstrained = np.asarray(B_unconstrained, dtype=float)
        number_of_skew_parameters = n * (n - 1) // 2
        number_of_upper_triangular_parameters = n * (n - 1) // 2
        lower_row_indices, lower_col_indices = np.tril_indices(n, k=-1)
        skew_symmetric_matrix = np.zeros((n, n), dtype=float)
        skew_symmetric_matrix = skew_symmetric_matrix.at[lower_row_indices, lower_col_indices].set(
            B_unconstrained[:number_of_skew_parameters])
        skew_symmetric_matrix = skew_symmetric_matrix - skew_symmetric_matrix.T
        identity = np.eye(n, dtype=float)
        orthogonal_matrix_q = np.linalg.solve(identity + skew_symmetric_matrix, identity - skew_symmetric_matrix)
        constrained_schur_matrix = np.zeros((n, n), dtype=float)
        upper_row_indices, upper_col_indices = np.triu_indices(n, k=1)
        start_upper = number_of_skew_parameters
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

        len_uncH = 1
        len_uncQ = p
        len_uncB = (p * (p - 1) // 2) + (p * (p - 1) // 2) + p
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
        H = np.exp(unc_H[0])
        Q = np.diag(np.exp(unc_Q))
        B = _link_stable_matrix(unc_B, p)
        ct = (np.eye(B.shape[0]) - B) @ bar_beta
        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct}

    def _invlink_stable_matrix(B_constrained: np.ndarray, n: int) -> np.ndarray:
        B_constrained = np.asarray(B_constrained, dtype=float)
        _, orthogonal_matrix_q = np.linalg.eigh(B_constrained)
        constrained_schur_matrix = orthogonal_matrix_q.T @ B_constrained @ orthogonal_matrix_q
        identity = np.eye(n, dtype=float)
        cayley_matrix = np.linalg.solve(identity + orthogonal_matrix_q, identity - orthogonal_matrix_q)
        lower_row_indices, lower_col_indices = np.tril_indices(n, k=-1)
        skew_lower_triangular_part = cayley_matrix[lower_row_indices, lower_col_indices]
        upper_row_indices, upper_col_indices = np.triu_indices(n, k=1)
        upper_triangular_part = constrained_schur_matrix[upper_row_indices, upper_col_indices]
        diagonal_part = np.arctanh(np.clip(np.diag(constrained_schur_matrix), -0.999999, 0.999999))
        return np.concatenate([skew_lower_triangular_part, upper_triangular_part, diagonal_part])

    def _invlink(constrained_params: dict):
        H = constrained_params["H_param"]
        Q = constrained_params["Q_param"]
        B = constrained_params["B"]
        bar_beta = constrained_params["bar_beta"]
        uncH = np.array([np.log(H[0,0])])
        uncQ = np.log(np.diag(Q))
        uncB = _invlink_stable_matrix(B, p)
        params = np.concatenate([uncH, uncQ, uncB, bar_beta])
        return params

    return _fit(data, initial_guess, covariates, initialization, _dynamics, _link, _invlink, opt_options)

def simulation(fit_output, nsim, npaths, key: jax.Array):
    if "ct" not in fit_output or fit_output["ct"] is None:
        fit_output["ct"] = (np.eye(fit_output["B"].shape[0]) - fit_output["B"]) @ fit_output["bar_beta"]
    return _simulation(fit_output, nsim, _dynamics, npaths, key)
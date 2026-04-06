import jax
import jax.numpy as np
from models._kalman import _filter, _filter_light_univariate, _simulation, _fit

def _dynamics(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx):
    Q = params["Q_param"]
    H = params["H_param"]
    Mt = jax.lax.dynamic_index_in_dim(params["covariates"], idx, axis=0, keepdims=False)
    B = params["B"]
    Z = np.asarray(Mt, dtype=float)
    T = B @ bt
    return Z, T, H, identity_mat, Q, 0.0, params["ct"]

def _dynamics_collapsed(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx):
    Q = params["Q_param"]
    sigma2 = params["H_param"][0, 0]
    Gamma_t = jax.lax.dynamic_index_in_dim(params["covariates"], idx, axis=0, keepdims=False)
    B = params["B"]
    Z = identity_mat
    T = B @ bt
    H = sigma2 * Gamma_t
    return Z, T, H, identity_mat, Q, 0.0, params["ct"]

def _collapsed_correction(constr_params, extra_ll_data):
    sigma2 = constr_params["H_param"][0, 0]
    n, pH_minus_p, sum_logdet_Gamma, sum_resid_sq = (
        extra_ll_data[0], extra_ll_data[1], extra_ll_data[2], extra_ll_data[3]
    )
    return -n / 2 * pH_minus_p * np.log(sigma2) + 0.5 * sum_logdet_Gamma - 0.5 / sigma2 * sum_resid_sq

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

def fit(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    initialization: tuple,
    opt_options: dict | None = None):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    p = (initial_guess["Q_param"]).shape[0]
    pH = (initial_guess["H_param"]).shape[0]

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
        H = np.exp(unc_H[0]) * np.eye(pH, dtype=float)
        Q = np.diag(np.exp(unc_Q))
        B = _link_stable_matrix(unc_B, p)
        ct = (np.eye(B.shape[0]) - B) @ bar_beta
        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct}

    def _invlink(constrained_params: dict):
        H = constrained_params["H_param"]
        Q = constrained_params["Q_param"]
        B = constrained_params["B"]
        bar_beta = constrained_params["bar_beta"]
        uncH = np.array([np.log(H[0, 0])])
        uncQ = np.log(np.diag(Q))
        uncB = _invlink_stable_matrix(B, p)
        params = np.concatenate([uncH, uncQ, uncB, bar_beta])
        return params

    return _fit(
        data, initial_guess, covariates, initialization,
        _dynamics, _link, _invlink, opt_options,
        _filter_fn=_filter_light_univariate,
    )

def fit_collapsed(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    initialization: tuple,
    opt_options: dict | None = None):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    p = (initial_guess["Q_param"]).shape[0]
    pH = (initial_guess["H_param"]).shape[0]

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
        H = np.exp(unc_H[0]) * np.eye(pH, dtype=float)
        Q = np.diag(np.exp(unc_Q))
        B = _link_stable_matrix(unc_B, p)
        ct = (np.eye(B.shape[0]) - B) @ bar_beta
        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct}

    def _invlink(constrained_params: dict):
        H = constrained_params["H_param"]
        Q = constrained_params["Q_param"]
        B = constrained_params["B"]
        bar_beta = constrained_params["bar_beta"]
        uncH = np.array([np.log(H[0, 0])])
        uncQ = np.log(np.diag(Q))
        uncB = _invlink_stable_matrix(B, p)
        params = np.concatenate([uncH, uncQ, uncB, bar_beta])
        return params

    Zt_all = covariates
    ZtTZt = np.einsum("npi,npj->nij", Zt_all, Zt_all)
    Gamma = np.linalg.inv(ZtTZt)
    ystar = np.einsum("nij,npj,np->ni", Gamma, Zt_all, data)
    e = data - np.einsum("npi,ni->np", Zt_all, ystar)
    sum_resid_sq = np.sum(e ** 2)
    sum_logdet_Gamma = np.sum(np.linalg.slogdet(Gamma)[1])
    extra_ll_data = np.array([float(data.shape[0]), float(pH - p), float(sum_logdet_Gamma), float(sum_resid_sq)])

    a1, P1, _Z0, T0, _H0, R0, Q0, _idx = initialization
    carry_collapsed = (
        a1, P1,
        np.eye(p, dtype=float),
        T0,
        initial_guess["H_param"][0, 0] * Gamma[0],
        R0, Q0,
        np.asarray(0, dtype=np.int32),
    )

    result = _fit(
        ystar, initial_guess, Gamma, carry_collapsed,
        _dynamics_collapsed, _link, _invlink, opt_options,
        extra_loglikelihood_fn=_collapsed_correction,
        extra_ll_data=extra_ll_data,
    )

    fitted_params = {k: result[k] for k in ["Q_param", "H_param", "B", "bar_beta", "ct"]}
    fitted_params["covariates"] = covariates
    kf = _filter(data, _dynamics, fitted_params, initialization)
    return fitted_params | kf | {
        "loglikelihood": result["loglikelihood"],
        "niter": result["niter"],
        "is_converged": result["is_converged"],
    }

def simulation(fit_output, nsim, npaths, key: jax.Array):
    return _simulation(fit_output, nsim, _dynamics, npaths, key)
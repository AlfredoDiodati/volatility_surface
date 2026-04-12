import jax
import jax.numpy as np
from models._kalman import _filter, _filter_light_univariate, _simulation, _fit


def _dynamics(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx):
    Q = params["Q_param"]
    H = params["H_param"]
    raw = jax.lax.dynamic_index_in_dim(params["covariates"], idx, axis=0, keepdims=False)
    base_cols = raw[:, :-1]
    bucket_indices = raw[:, -1].astype(int)
    omega_col = params["omega"][bucket_indices]
    B = params["B"]
    Z = np.concatenate([base_cols, omega_col[:, None]], axis=-1)
    T = B @ bt
    return Z, T, H, identity_mat, Q, 0.0, params["ct"]


def _dynamics_collapsed(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx):
    Q = params["Q_param"]
    sigma2 = params["H_param"][0, 0]
    Gamma  = params["Gamma"]   # (P, P) — time-invariant, no per-step indexing
    ystar_t = jax.lax.dynamic_index_in_dim(params["ystar"], idx, axis=0, keepdims=False)
    B = params["B"]
    Z = identity_mat
    T = B @ bt
    H = sigma2 * Gamma
    d = -ystar_t
    return Z, T, H, identity_mat, Q, d, params["ct"]


def _collapsed_correction(constr_params, _extra_ll_data):
    sigma2 = constr_params["H_param"][0, 0]
    return (
        constr_params["n_half_ph_minus_p"] * np.log(sigma2)
        + 0.5 * constr_params["sum_logdet_Gamma"]
        - 0.5 / sigma2 * constr_params["sum_resid_sq"]
    )


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


def _build_Zt_all(base_covariates, bucket_indices, omega):
    def one_t(base_t, bidx_t):
        omega_col = omega[bidx_t]
        return np.concatenate([base_t, omega_col[:, None]], axis=-1)
    return jax.vmap(one_t)(base_covariates, bucket_indices)


def fit(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    initialization: tuple,
    opt_options: dict | None = None):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    p = initial_guess["Q_param"].shape[0]
    pH = initial_guess["H_param"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]

    def _link(unconstrained_params: np.ndarray) -> dict:
        unconstrained_params = np.asarray(unconstrained_params, dtype=float)
        len_uncB = (p * (p - 1) // 2) + (p * (p - 1) // 2) + p
        end_H = 1
        end_Q = end_H + p
        end_B = end_Q + len_uncB
        end_omega = end_B + (n_buckets - 1)
        H = np.exp(unconstrained_params[0]) * np.eye(pH, dtype=float)
        Q = np.diag(np.exp(unconstrained_params[end_H:end_Q]))
        B = _link_stable_matrix(unconstrained_params[end_Q:end_B], p)
        omega = np.concatenate([np.zeros(1), unconstrained_params[end_B:end_omega]])
        bar_beta = unconstrained_params[end_omega:]
        ct = (np.eye(p) - B) @ bar_beta
        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct, "omega": omega}

    def _invlink(constrained_params: dict):
        uncH = np.array([np.log(constrained_params["H_param"][0, 0])])
        uncQ = np.log(np.diag(constrained_params["Q_param"]))
        uncB = _invlink_stable_matrix(constrained_params["B"], p)
        return np.concatenate([uncH, uncQ, uncB, constrained_params["omega"][1:], constrained_params["bar_beta"]])

    return _fit(
        data, initial_guess, covariates, initialization,
        _dynamics, _link, _invlink, opt_options,
        _filter_fn=_filter_light_univariate,
    )


def _build_Zt(base_covariates_2d, bucket_indices_2d, omega):
    """Build time-invariant loading matrix. Returns (N, P)."""
    omega_col = omega[bucket_indices_2d]                               # (N,)
    return np.concatenate([base_covariates_2d, omega_col[:, None]], axis=-1)  # (N, P)


def fit_collapsed(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    initialization: tuple,
    opt_options: dict | None = None):
    """
    Expects covariates as (N, P_BASE+1) — time-invariant, single copy.
    Avoids O(T·N·P) intermediates by computing Zt once and using batch matmuls.
    """
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)  # (N, P_BASE+1)

    p         = initial_guess["Q_param"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    N_obs     = initial_guess.get("N_obs", covariates.shape[0])
    T_obs     = data.shape[0]

    base_covariates = covariates[:, :-1]              # (N, P_BASE)
    bucket_indices  = covariates[:, -1].astype(np.int32)  # (N,)
    dummy_data      = np.zeros((T_obs, p), dtype=float)

    def _link(unconstrained_params: np.ndarray) -> dict:
        unconstrained_params = np.asarray(unconstrained_params, dtype=float)
        len_uncB  = (p * (p - 1) // 2) + (p * (p - 1) // 2) + p
        end_H     = 1
        end_Q     = end_H + p
        end_B     = end_Q + len_uncB
        end_omega = end_B + (n_buckets - 1)
        H        = np.exp(unconstrained_params[0]) * np.eye(1, dtype=float)  # (1,1)
        Q        = np.diag(np.exp(unconstrained_params[end_H:end_Q]))
        B        = _link_stable_matrix(unconstrained_params[end_Q:end_B], p)
        omega    = np.concatenate([np.zeros(1), unconstrained_params[end_B:end_omega]])
        bar_beta = unconstrained_params[end_omega:]
        ct       = (np.eye(p) - B) @ bar_beta

        # Zt computed once — (N, P), not (T, N, P)
        Zt    = _build_Zt(base_covariates, bucket_indices, omega)
        ZtTZt = Zt.T @ Zt                           # (P, P)
        Gamma = np.linalg.inv(ZtTZt)               # (P, P) — time-invariant

        # ystar_t = Gamma @ Zt^T @ y_t = (y_t @ Zt) @ Gamma (Gamma symmetric)
        ystar = (data @ Zt) @ Gamma                 # (T, P)

        # ‖e‖² = ‖y‖² − ‖P_Z y‖² = ‖y‖² − tr(ystar^T ZtTZt ystar)
        # avoids materialising the (T, N) residual matrix
        sum_resid_sq = np.sum(data ** 2) - np.sum(ystar * (ystar @ ZtTZt))

        return {
            "Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct,
            "omega": omega,
            "Gamma": Gamma,                                               # (P, P)
            "ystar": ystar,                                               # (T, P)
            "n_half_ph_minus_p": np.asarray(-float(T_obs) / 2 * float(N_obs - p)),
            "sum_logdet_Gamma":  np.asarray(float(T_obs) * np.linalg.slogdet(Gamma)[1]),
            "sum_resid_sq":      sum_resid_sq,
        }

    def _invlink(constrained_params: dict):
        uncH = np.array([np.log(constrained_params["H_param"][0, 0])])
        uncQ = np.log(np.diag(constrained_params["Q_param"]))
        uncB = _invlink_stable_matrix(constrained_params["B"], p)
        return np.concatenate([uncH, uncQ, uncB, constrained_params["omega"][1:], constrained_params["bar_beta"]])

    # Initial values (outside _link so they don't re-run every opt step)
    initial_Zt    = _build_Zt(base_covariates, bucket_indices, initial_guess["omega"])
    initial_ZtTZt = initial_Zt.T @ initial_Zt                  # (P, P)
    initial_Gamma = np.linalg.inv(initial_ZtTZt)               # (P, P)
    initial_ystar = (data @ initial_Zt) @ initial_Gamma        # (T, P)
    initial_sum_resid_sq = (
        np.sum(data ** 2) - np.sum(initial_ystar * (initial_ystar @ initial_ZtTZt))
    )

    initial_guess_augmented = initial_guess | {
        "Gamma":              initial_Gamma,
        "ystar":              initial_ystar,
        "n_half_ph_minus_p":  np.asarray(-float(T_obs) / 2 * float(N_obs - p)),
        "sum_logdet_Gamma":   np.asarray(float(T_obs) * np.linalg.slogdet(initial_Gamma)[1]),
        "sum_resid_sq":       initial_sum_resid_sq,
    }

    a1, P1, _Z0, T0, _H0, R0, Q0, _idx = initialization
    carry_collapsed = (
        a1, P1,
        np.eye(p, dtype=float),
        T0,
        initial_guess["H_param"][0, 0] * initial_Gamma,   # (P, P) — Gamma is now (P,P)
        R0, Q0,
        np.asarray(0, dtype=np.int32),
    )

    result = _fit(
        dummy_data, initial_guess_augmented, initial_Gamma, carry_collapsed,
        _dynamics_collapsed, _link, _invlink, opt_options,
        extra_loglikelihood_fn=_collapsed_correction,
        extra_ll_data=np.zeros(4),
    )

    fitted_params = {k: result[k] for k in ["Q_param", "H_param", "B", "bar_beta", "ct", "omega"]}
    # Broadcast (N, P+1) → (T, N, P+1) for _dynamics, which indexes by time step.
    # broadcast_to is zero-copy; XLA optimises the indexing to a slice of the base array.
    fitted_params["covariates"] = np.broadcast_to(
        covariates[None], (T_obs,) + covariates.shape
    )
    kf = _filter(data, _dynamics, fitted_params, initialization)
    return fitted_params | kf | {
        "loglikelihood": result["loglikelihood"],
        "niter":         result["niter"],
        "is_converged":  result["is_converged"],
    }

def simulation(fit_output, nsim, npaths, key: jax.Array):
    return _simulation(fit_output, nsim, _dynamics, npaths, key)
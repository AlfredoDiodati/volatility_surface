import jax
import jax.numpy as np
from models._kalman import _filter_light_univariate, _simulation, _fit

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
    Gamma_t = jax.lax.dynamic_index_in_dim(params["Gamma"], idx, axis=0, keepdims=False)
    ystar_t = jax.lax.dynamic_index_in_dim(params["ystar"], idx, axis=0, keepdims=False)
    B = params["B"]
    Z = identity_mat
    T = B @ bt
    H = sigma2 * Gamma_t
    d = -ystar_t
    return Z, T, H, identity_mat, Q, d, params["ct"]

def _collapsed_correction(constr_params, _extra_ll_data):
    sigma2 = constr_params["H_param"][0, 0]
    return (
        constr_params["n_half_ph_minus_p"] * np.log(sigma2)
        + 0.5 * constr_params["sum_logdet_Gamma"]
        - 0.5 / sigma2 * constr_params["sum_resid_sq"]
    )


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
    opt_options: dict | None = None,
):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    p = initial_guess["Q_param"].shape[0]
    pH = initial_guess["H_param"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]

    def _link(unconstrained_params: np.ndarray) -> dict:
        unconstrained_params = np.asarray(unconstrained_params, dtype=float)
        end_H = 1
        end_Q = end_H + p
        end_B = end_Q + p
        end_omega = end_B + (n_buckets - 1)
        H = np.exp(unconstrained_params[0]) * np.eye(pH, dtype=float)
        Q = np.diag(np.exp(unconstrained_params[end_H:end_Q]))
        B = np.diag(np.tanh(unconstrained_params[end_Q:end_B]))
        omega = np.concatenate([np.zeros(1), unconstrained_params[end_B:end_omega]])
        bar_beta = unconstrained_params[end_omega:]
        ct = (np.eye(p) - B) @ bar_beta
        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct, "omega": omega}

    def _invlink(constrained_params: dict):
        uncH = np.array([np.log(constrained_params["H_param"][0, 0])])
        uncQ = np.log(np.diag(constrained_params["Q_param"]))
        uncB = np.arctanh(np.clip(np.diag(constrained_params["B"]), -0.999999, 0.999999))
        return np.concatenate([uncH, uncQ, uncB, constrained_params["omega"][1:], constrained_params["bar_beta"]])

    return _fit(
        data, initial_guess, covariates, initialization,
        _dynamics, _link, _invlink, opt_options or {},
        _filter_fn=_filter_light_univariate,
    )

def fit_collapsed(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    initialization: tuple,
    opt_options: dict | None = None,
    maxiter: int = 5000,
):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    p = initial_guess["Q_param"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    T_obs, max_n = data.shape
    y_mask = ~np.isnan(data)
    y_masked = np.where(y_mask, data, 0.0)
    Z_mask = y_mask[:, :, None]
    base_covariates = covariates[:, :, :-1]
    bucket_cube = covariates[:, :, -1].astype(np.int32)
    total_obs = np.sum(y_mask)
    n_half_total = (total_obs - T_obs * p) / 2.0
    dummy_data = np.zeros((T_obs, p), dtype=float)

    def _build_Zt_cube(omega):
        omega_cube = omega[bucket_cube]
        Zt_full = np.concatenate([base_covariates, omega_cube[:, :, None]], axis=-1)
        return Zt_full * Z_mask

    def _link(unconstrained_params: np.ndarray) -> dict:
        unconstrained_params = np.asarray(unconstrained_params, dtype=float)
        end_H = 1
        end_Q = end_H + p
        end_B = end_Q + p
        end_omega = end_B + (n_buckets - 1)
        H = np.exp(unconstrained_params[0]) * np.eye(1, dtype=float)
        Q = np.diag(np.exp(unconstrained_params[end_H:end_Q]))
        B = np.diag(np.tanh(unconstrained_params[end_Q:end_B]))
        omega = np.concatenate([np.zeros(1), unconstrained_params[end_B:end_omega]])
        bar_beta = unconstrained_params[end_omega:]
        ct = (np.eye(p) - B) @ bar_beta
        Zt_cube = _build_Zt_cube(omega)
        ZtTZt = np.einsum("tnp,tnq->tpq", Zt_cube, Zt_cube)
        Gamma = np.linalg.inv(ZtTZt + 1e-8 * np.eye(p))
        ystar = np.einsum("tpq,tnq,tn->tp", Gamma, Zt_cube, y_masked)
        sum_yy = np.sum(y_masked ** 2)
        sum_ZyZy = np.sum(ystar * np.einsum("tpq,tq->tp", ZtTZt, ystar))
        sum_resid_sq = sum_yy - sum_ZyZy
        return {
            "Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct,
            "omega": omega,
            "Gamma": Gamma,
            "ystar": ystar,
            "n_half_ph_minus_p": np.asarray(-n_half_total),
            "sum_logdet_Gamma": np.sum(np.linalg.slogdet(Gamma)[1]),
            "sum_resid_sq": sum_resid_sq,
        }

    def _invlink(constrained_params: dict):
        uncH = np.array([np.log(constrained_params["H_param"][0, 0])])
        uncQ = np.log(np.diag(constrained_params["Q_param"]))
        uncB = np.arctanh(np.clip(np.diag(constrained_params["B"]), -0.999999, 0.999999))
        return np.concatenate([
            uncH, uncQ, uncB,
            constrained_params["omega"][1:],
            constrained_params["bar_beta"],
        ])

    initial_Zt_cube = _build_Zt_cube(initial_guess["omega"])
    initial_ZtTZt = np.einsum("tnp,tnq->tpq", initial_Zt_cube, initial_Zt_cube)
    initial_Gamma = np.linalg.inv(initial_ZtTZt + 1e-8 * np.eye(p))
    initial_ystar = np.einsum("tpq,tnq,tn->tp", initial_Gamma, initial_Zt_cube, y_masked)
    init_sum_yy = np.sum(y_masked ** 2)
    init_sum_ZyZy = np.sum(initial_ystar * np.einsum("tpq,tq->tp", initial_ZtTZt, initial_ystar))
    initial_guess_augmented = initial_guess | {
        "Gamma": initial_Gamma,
        "ystar": initial_ystar,
        "n_half_ph_minus_p": -n_half_total,
        "sum_logdet_Gamma": np.sum(np.linalg.slogdet(initial_Gamma)[1]),
        "sum_resid_sq": init_sum_yy - init_sum_ZyZy,
    }
    a1, P1, _Z0, T0, _H0, R0, Q0, _idx = initialization
    carry_collapsed = (
        a1, P1,
        np.eye(p, dtype=float),
        T0,
        initial_guess["H_param"][0, 0] * initial_Gamma[0],
        R0, Q0,
        np.asarray(0, dtype=np.int32),
    )
    result = _fit(
        dummy_data, initial_guess_augmented, initial_Gamma, carry_collapsed,
        _dynamics_collapsed, _link, _invlink, opt_options or {},
        maxiter=maxiter,
        extra_loglikelihood_fn=_collapsed_correction,
        extra_ll_data=np.zeros(4),
    )
    fitted_params = {k: result[k] for k in ["Q_param", "H_param", "B", "bar_beta", "ct", "omega"]}
    return fitted_params | {k: result[k] for k in ["logdetF", "quad", "a", "P", "att", "Ptt", "v", "F", "K"]} | {
        "loglikelihood": result["loglikelihood"],
        "niter": result["niter"],
        "is_converged": result["is_converged"],
    }

def forecast(fit_result, covariates, y_test, q_alpha):
    a0 = fit_result["att"][-1]
    P0 = fit_result["Ptt"][-1]
    B = fit_result["B"]
    Q = fit_result["Q_param"]
    ct = fit_result["ct"]
    sigma2 = fit_result["H_param"][0, 0]
    omega = fit_result["omega"]

    covariates = np.asarray(covariates, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    base_covariates = covariates[:, :, :-1]
    bucket_indices = covariates[:, :, -1].astype(np.int32)

    Z_all = _build_Zt_all(base_covariates, bucket_indices, omega)

    def _step(carry, inputs):
        Z_h, y_h = inputs
        a_t, P_t = carry
        a_pred = B @ a_t + ct
        P_pred = B @ P_t @ B.T + Q
        mask_h = ~np.isnan(y_h)
        n_h = np.sum(mask_h)
        y_hat = Z_h @ a_pred
        P_mean = y_hat.sum() / n_h
        z_sum = Z_h.sum(axis=0)
        F_sum = n_h * sigma2 + z_sum @ P_pred @ z_sum
        VaR_h = P_mean + q_alpha / n_h * np.sqrt(F_sum)
        y_m = np.where(mask_h, y_h, 0.0)
        v = (y_m - Z_h @ a_pred) * mask_h
        F_vec = np.einsum("ip,pq,iq->i", Z_h, P_pred, Z_h) + sigma2
        oos_ll = -0.5 * (n_h * np.log(2 * np.pi) + np.sum(mask_h * np.log(F_vec)) + np.sum(mask_h * v ** 2 / F_vec))
        return (a_pred, P_pred), (y_hat, P_mean, VaR_h, oos_ll)

    _, (predictions, P_means, VaR, log_liks) = jax.lax.scan(_step, (a0, P0), (Z_all, y_test))

    return predictions, P_means, VaR, log_liks


def simulation(fit_output, nsim, npaths, key: jax.Array):
    return _simulation(fit_output, nsim, _dynamics, npaths, key)

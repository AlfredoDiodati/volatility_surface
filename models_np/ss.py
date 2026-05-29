import numpy as np
from models_np._kalman import _filter_light_univariate, _fit


def _dynamics(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx):
    Q = params["Q_param"]
    H = params["H_param"]
    raw = params["covariates"][idx]
    base_cols = raw[:, :-1]
    bucket_indices = raw[:, -1].astype(int)
    omega_col = params["omega"][bucket_indices]
    B = params["B"]
    Z = np.concatenate([base_cols, omega_col[:, None]], axis=-1)
    T = B
    return Z, T, H, identity_mat, Q, 0.0, params["ct"]


def _dynamics_collapsed(y, _a, _P, params, _Z, bt, _H, identity_mat, _Q, idx):
    Q = params["Q_param"]
    sigma2 = params["H_param"][0, 0]
    Gamma_t = params["Gamma"][idx]
    ystar_t = params["ystar"][idx]
    B = params["B"]
    Z = identity_mat
    T = B
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
    omega_cols = omega[bucket_indices]
    return np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)


def fit(data, M, initial_guess, initialization, opt_options=None):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = initial_guess["Q_param"].shape[0]
    pH = initial_guess["H_param"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    _eye_pH = np.eye(pH, dtype=float)

    def _link(unconstrained_params):
        unconstrained_params = np.asarray(unconstrained_params, dtype=float)
        end_H = 1
        end_Q = end_H + p
        end_B = end_Q + p
        end_omega = end_B + (n_buckets - 1)
        H = np.exp(unconstrained_params[0]) * _eye_pH
        Q = np.diag(np.exp(unconstrained_params[end_H:end_Q]))
        b_diag = np.tanh(unconstrained_params[end_Q:end_B])
        B = np.diag(b_diag)
        omega = np.concatenate([np.zeros(1), unconstrained_params[end_B:end_omega]])
        bar_beta = unconstrained_params[end_omega:]
        ct = (1.0 - b_diag) * bar_beta
        return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct, "omega": omega}

    def _invlink(constrained_params):
        uncH = np.array([np.log(constrained_params["H_param"][0, 0])])
        uncQ = np.log(np.diag(constrained_params["Q_param"]))
        uncB = np.arctanh(np.clip(np.diag(constrained_params["B"]), -0.999999, 0.999999))
        return np.concatenate([uncH, uncQ, uncB, constrained_params["omega"][1:], constrained_params["bar_beta"]])

    return _fit(
        data, initial_guess, M, initialization,
        _dynamics, _link, _invlink, opt_options or {},
        _filter_fn=_filter_light_univariate,
    )


def fit_collapsed(data, M, initial_guess, initialization, opt_options=None, maxiter=5000):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = initial_guess["Q_param"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    T_obs, max_n = data.shape
    y_mask = ~np.isnan(data)
    y_masked = np.where(y_mask, data, 0.0)
    Z_mask = y_mask[:, :, None]
    base_covariates = M[:, :, :-1]
    bucket_cube = M[:, :, -1].astype(np.int32)
    total_obs = np.sum(y_mask)
    n_half_total = (total_obs - T_obs * p) / 2.0
    dummy_data = np.zeros((T_obs, p), dtype=float)
    _eye_p = np.eye(p, dtype=float)
    _eye_1 = np.eye(1, dtype=float)
    _sum_yy = np.sum(y_masked ** 2)

    def _build_Zt_cube(omega):
        omega_cube = omega[bucket_cube]
        Zt_full = np.concatenate([base_covariates, omega_cube[:, :, None]], axis=-1)
        return Zt_full * Z_mask

    def _link(unconstrained_params):
        unconstrained_params = np.asarray(unconstrained_params, dtype=float)
        end_H = 1
        end_Q = end_H + p
        end_B = end_Q + p
        end_omega = end_B + (n_buckets - 1)
        H = np.exp(unconstrained_params[0]) * _eye_1
        Q = np.diag(np.exp(unconstrained_params[end_H:end_Q]))
        b_diag = np.tanh(unconstrained_params[end_Q:end_B])
        B = np.diag(b_diag)
        omega = np.concatenate([np.zeros(1), unconstrained_params[end_B:end_omega]])
        bar_beta = unconstrained_params[end_omega:]
        ct = (1.0 - b_diag) * bar_beta
        Zt_cube = _build_Zt_cube(omega)
        ZtTZt = Zt_cube.swapaxes(1, 2) @ Zt_cube
        Gamma = np.linalg.inv(ZtTZt + 1e-8 * _eye_p)
        ZtY = (Zt_cube.swapaxes(1, 2) @ y_masked[:, :, None]).squeeze(-1)
        ystar = (Gamma @ ZtY[:, :, None]).squeeze(-1)
        Zy = (Zt_cube @ ystar[:, :, None]).squeeze(-1)
        sum_resid_sq = _sum_yy - np.sum(Zy ** 2)
        return {
            "Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": ct,
            "omega": omega,
            "Gamma": Gamma,
            "ystar": ystar,
            "n_half_ph_minus_p": -n_half_total,
            "sum_logdet_Gamma": np.sum(np.linalg.slogdet(Gamma)[1]),
            "sum_resid_sq": sum_resid_sq,
        }

    def _invlink(constrained_params):
        uncH = np.array([np.log(constrained_params["H_param"][0, 0])])
        uncQ = np.log(np.diag(constrained_params["Q_param"]))
        uncB = np.arctanh(np.clip(np.diag(constrained_params["B"]), -0.999999, 0.999999))
        return np.concatenate([
            uncH, uncQ, uncB,
            constrained_params["omega"][1:],
            constrained_params["bar_beta"],
        ])

    initial_Zt_cube = _build_Zt_cube(initial_guess["omega"])
    initial_ZtTZt = initial_Zt_cube.swapaxes(1, 2) @ initial_Zt_cube
    initial_Gamma = np.linalg.inv(initial_ZtTZt + 1e-8 * _eye_p)
    initial_ZtY = (initial_Zt_cube.swapaxes(1, 2) @ y_masked[:, :, None]).squeeze(-1)
    initial_ystar = (initial_Gamma @ initial_ZtY[:, :, None]).squeeze(-1)
    initial_Zy = (initial_Zt_cube @ initial_ystar[:, :, None]).squeeze(-1)
    initial_guess_augmented = initial_guess | {
        "Gamma": initial_Gamma,
        "ystar": initial_ystar,
        "n_half_ph_minus_p": -n_half_total,
        "sum_logdet_Gamma": np.sum(np.linalg.slogdet(initial_Gamma)[1]),
        "sum_resid_sq": _sum_yy - np.sum(initial_Zy ** 2),
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
    return (
        fitted_params
        | {k: result[k] for k in ["logdetF", "quad", "a", "P", "att", "Ptt", "v", "F", "K"]}
        | {"loglikelihood": result["loglikelihood"], "niter": result["niter"], "is_converged": result["is_converged"]}
    )


def forecast(fit_result, M, y_test, q_alpha):
    a0 = fit_result["att"][-1]
    P0 = fit_result["Ptt"][-1]
    B = fit_result["B"]
    Q = fit_result["Q_param"]
    ct = fit_result["ct"]
    sigma2 = fit_result["H_param"][0, 0]
    omega = fit_result["omega"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    Z_all = _build_Zt_all(base_covariates, bucket_indices, omega)

    T_steps = len(Z_all)
    N = Z_all.shape[1]
    b = B.diagonal()
    h_inv = 1.0 / sigma2
    _eye_p = np.eye(b.shape[0])
    _log_sigma2 = np.log(sigma2)
    _log_2pi = np.log(2 * np.pi)
    predictions = np.empty((T_steps, N))
    P_means = np.empty(T_steps)
    VaR = np.empty(T_steps)
    log_liks = np.empty(T_steps)

    a_t, P_t = a0.copy(), P0.copy()
    for t in range(T_steps):
        Z_h = Z_all[t]
        y_h = y_test[t]
        a_pred = b * a_t + ct
        P_pred = b[:, None] * P_t * b + Q
        mask_h = ~np.isnan(y_h)
        n_h = np.sum(mask_h)
        y_hat = Z_h @ a_pred
        P_mean = y_hat.sum() / n_h
        z_sum = Z_h.sum(axis=0)
        F_sum = n_h * sigma2 + z_sum @ P_pred @ z_sum
        y_m = np.where(mask_h, y_h, 0.0)
        v = (y_m - y_hat) * mask_h
        Z_masked = np.where(mask_h[:, None], Z_h, 0.0)
        PZm = P_pred @ Z_masked.T
        Inn = _eye_p + h_inv * PZm @ Z_masked
        ZtV = Z_masked.T @ v
        sol = np.linalg.solve(Inn, PZm @ np.column_stack([v, Z_masked]))
        c, D = sol[:, 0], sol[:, 1:]
        _, log_det_corr = np.linalg.slogdet(Inn)
        quad = h_inv * np.dot(v, v) - h_inv ** 2 * ZtV @ c
        predictions[t] = y_hat
        P_means[t] = P_mean
        VaR[t] = P_mean + q_alpha / n_h * np.sqrt(F_sum)
        log_liks[t] = -0.5 * (n_h * (_log_2pi + _log_sigma2) + log_det_corr + quad)
        a_t = a_pred + h_inv * c
        P_t = P_pred - h_inv * D @ P_pred

    return predictions, P_means, VaR, log_liks


def forecast_rolling_h(fit_result, M, y_test, eval_horizons):
    a0 = fit_result["att"][-1]
    P0 = fit_result["Ptt"][-1]
    B = fit_result["B"]
    Q = fit_result["Q_param"]
    ct = fit_result["ct"]
    sigma2 = fit_result["H_param"][0, 0]
    omega = fit_result["omega"]
    b = B.diagonal()
    mu = ct / (1.0 - b)

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cols = omega[bucket_indices]
    Z_all = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)

    b_h_stack = np.array([b ** h for h in eval_horizons])
    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)
    Z_update = Z_all[:T_half]
    h_inv = 1.0 / sigma2
    _eye_p = np.eye(b.shape[0])
    n_horizons = len(eval_horizons)
    N = Z_all.shape[1]
    preds_all = np.empty((T_half, n_horizons, N))

    a_t, P_t = a0.copy(), P0.copy()
    for t in range(T_half):
        Z_h_t = Z_h_scan[t]
        Z_upd_t = Z_update[t]
        y_h = y_test[t]

        dev = a_t - mu
        a_h_stack = mu + b_h_stack * dev
        preds_all[t] = np.einsum("hnp,hp->hn", Z_h_t, a_h_stack)

        a_pred = b * a_t + ct
        P_pred = b[:, None] * P_t * b + Q
        mask_h = ~np.isnan(y_h)
        y_m = np.where(mask_h, y_h, 0.0)
        v = (y_m - Z_upd_t @ a_pred) * mask_h
        Z_masked = np.where(mask_h[:, None], Z_upd_t, 0.0)
        PZm = P_pred @ Z_masked.T
        Inner = _eye_p + h_inv * PZm @ Z_masked
        sol = np.linalg.solve(Inner, PZm @ np.column_stack([v, Z_masked]))
        c, D = sol[:, 0], sol[:, 1:]
        a_t = a_pred + h_inv * c
        P_t = P_pred - h_inv * D @ P_pred

    return preds_all.transpose(1, 0, 2)


def forecast_h(fit_result, M, q_alpha):
    a0 = fit_result["att"][-1]
    P0 = fit_result["Ptt"][-1]
    B = fit_result["B"]
    Q = fit_result["Q_param"]
    ct = fit_result["ct"]
    sigma2 = fit_result["H_param"][0, 0]
    omega = fit_result["omega"]

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    Z_all = _build_Zt_all(base_covariates, bucket_indices, omega)
    T_steps = len(Z_all)
    n_h = Z_all.shape[1]
    b = B.diagonal()

    predictions = np.empty((T_steps, n_h))
    P_means = np.empty(T_steps)
    VaR = np.empty(T_steps)

    a_t, P_t = a0.copy(), P0.copy()
    for t in range(T_steps):
        Z_h = Z_all[t]
        a_pred = b * a_t + ct
        P_pred = b[:, None] * P_t * b + Q
        y_hat = Z_h @ a_pred
        P_mean = y_hat.sum() / n_h
        z_sum = Z_h.sum(axis=0)
        F_sum = n_h * sigma2 + z_sum @ P_pred @ z_sum
        predictions[t] = y_hat
        P_means[t] = P_mean
        VaR[t] = P_mean + q_alpha / n_h * np.sqrt(F_sum)
        a_t, P_t = a_pred, P_pred

    return predictions, P_means, VaR


def simulation(fit_result, M, n, rng):
    B = fit_result["B"]
    Q = fit_result["Q_param"]
    ct = fit_result["ct"]
    sigma2 = fit_result["H_param"][0, 0]
    omega = fit_result["omega"]
    a0 = fit_result["att"][-1]

    M = np.asarray(M, dtype=float)
    T, N_obs, _ = M.shape
    base = M[:, :, :-1]
    bidx = M[:, :, -1].astype(np.int32)
    design = np.concatenate([base, omega[bidx][:, :, None]], axis=-1)

    sqrt_sigma = np.sqrt(sigma2)
    b = B.diagonal()
    p_state = b.shape[0]
    y_paths = np.empty((n, T, N_obs))

    for i in range(n):
        eta_draws = rng.multivariate_normal(np.zeros(p_state), Q, size=T)
        eps_draws = rng.standard_normal(size=(T, N_obs)) * sqrt_sigma
        a_t = a0.copy()
        for t in range(T):
            y_paths[i, t] = design[t] @ a_t + eps_draws[t]
            a_t = b * a_t + ct + eta_draws[t]

    return y_paths

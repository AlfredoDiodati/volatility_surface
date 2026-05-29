import numpy as np
from models_np._kalman import _filter_light_univariate, _fit


def _solve_weights_ff(eta, alpha, K):
    p = len(eta)
    indices = np.arange(K + 1, dtype=float)
    ws = np.empty((p, K + 1))
    lambdas = np.empty((p, K + 1))
    for i in range(p):
        eta_k = float(eta[i])
        alpha_k = float(alpha[i])
        lambdas[i] = eta_k * alpha_k ** (-indices)
        c = np.zeros(K + 1)
        c[0] = 1.0
        for k in range(1, K + 1):
            diff_k = k - indices[:k]
            c[k] = 1.0 - np.sum(c[:k] * alpha_k ** (-eta_k * (k - 1)) * np.exp(eta_k * (1.0 - alpha_k ** diff_k)))
        c_ordered = np.flip(c)
        w_tilde = c_ordered * alpha_k ** (-indices * eta_k)
        ws[i] = w_tilde / np.sum(w_tilde)
    return ws.T, lambdas.T


def _dynamics(y, _a, _P, params, _Z, _T, _H, _R, _Q, idx):
    raw = params["covariates"][idx]
    base_cols = raw[:, :-1]
    bucket_indices = raw[:, -1].astype(int)
    omega_col = params["omega"][bucket_indices]
    M_t = np.concatenate([base_cols, omega_col[:, None]], axis=-1)
    ws = params["ws"]
    Z_t = (M_t[:, None, :] * ws[None, :, :]).reshape(M_t.shape[0], -1)
    d_t = M_t @ params["beta_bar"]
    return Z_t, params["T_aug"], params["H_obs"], _R, params["Q_aug"], d_t, 0.0


def _build_M_base(M, omega):
    base_cols = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cube = omega[bucket_indices]
    return np.concatenate([base_cols, omega_cube[:, :, None]], axis=-1)


def fit(data, M, initial_guess, K, opt_options=None, maxiter=5000):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    _, max_n, _ = M.shape
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    state_dim = (K + 1) * p
    maxiter = int(maxiter)
    opt_options = opt_options or {}

    def _link(theta):
        i = 0
        beta_bar = theta[i:i + p]; i += p
        sigma2 = np.exp(theta[i]); i += 1
        q_diag = np.exp(theta[i:i + p]); i += p
        omega = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        eta = np.exp(theta[i:i + p]); i += p
        alpha = 1.0 + np.logaddexp(0.0, theta[i])
        ws, lambdas = _solve_weights_ff(eta, np.full(p, alpha), K)
        T_aug = np.diag(np.exp(-lambdas).ravel())
        Q_aug = np.diag(np.tile(q_diag, K + 1))
        H_obs = sigma2 * np.eye(max_n, dtype=float)
        return {
            "beta_bar": beta_bar,
            "sigma2": sigma2,
            "Q_param": np.diag(q_diag),
            "omega": omega,
            "eta": eta,
            "alpha": alpha,
            "ws": ws,
            "lambdas": lambdas,
            "T_aug": T_aug,
            "Q_aug": Q_aug,
            "H_obs": H_obs,
        }

    def _invlink(params):
        return np.concatenate([
            params["beta_bar"],
            np.array([np.log(params["sigma2"])]),
            np.log(np.diag(params["Q_param"])),
            params["omega"][1:],
            np.log(params["eta"]),
            np.array([np.log(np.exp(params["alpha"] - 1.0) - 1.0)]),
        ])

    init = _link(_invlink(initial_guess))
    carry_initial = (
        np.zeros(state_dim, dtype=float),
        10.0 * np.eye(state_dim, dtype=float),
        np.zeros((max_n, state_dim), dtype=float),
        init["T_aug"],
        init["H_obs"],
        np.eye(state_dim, dtype=float),
        init["Q_aug"],
        np.asarray(0, dtype=np.int32),
    )

    result = _fit(
        data, initial_guess, M, carry_initial,
        _dynamics, _link, _invlink, opt_options,
        maxiter=maxiter,
        _filter_fn=_filter_light_univariate,
    )

    param_keys = ["beta_bar", "sigma2", "Q_param", "omega", "eta", "alpha", "ws", "lambdas", "T_aug", "Q_aug"]
    kf_keys = ["logdetF", "quad", "a", "P", "att", "Ptt", "v", "F", "K"]
    return (
        {k: result[k] for k in param_keys}
        | {k: result[k] for k in kf_keys}
        | {"loglikelihood": result["loglikelihood"], "niter": result["niter"], "is_converged": result["is_converged"]}
    )


def forecast(fit_result, M, y_test, q_alpha):
    a0 = fit_result["att"][-1]
    P0 = fit_result["Ptt"][-1]
    T_aug = fit_result["T_aug"]
    Q_aug = fit_result["Q_aug"]
    sigma2 = fit_result["sigma2"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H, N, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    t_diag = T_aug.diagonal()
    h_inv = 1.0 / sigma2
    state_dim = t_diag.shape[0]
    _eye_state = np.eye(state_dim)
    _log_sigma2 = np.log(sigma2)
    _log_2pi = np.log(2 * np.pi)
    predictions = np.empty((H, N))
    P_means = np.empty(H)
    VaR = np.empty(H)
    log_liks = np.empty(H)

    a_t, P_t = a0.copy(), P0.copy()
    for t in range(H):
        Z_h = Z_all[t]
        d_h = d_all[t]
        y_h = y_test[t]
        a_pred = t_diag * a_t
        P_pred = t_diag[:, None] * P_t * t_diag + Q_aug
        mask_h = ~np.isnan(y_h)
        n_h = np.sum(mask_h)
        y_hat = Z_h @ a_pred + d_h
        P_mean = y_hat.sum() / n_h
        z_sum = Z_h.sum(axis=0)
        F_sum = n_h * sigma2 + z_sum @ P_pred @ z_sum
        v = (y_h - y_hat) * mask_h
        Z_masked = np.where(mask_h[:, None], Z_h, 0.0)
        PZm = P_pred @ Z_masked.T
        Inn = _eye_state + h_inv * PZm @ Z_masked
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
    T_aug = fit_result["T_aug"]
    Q_aug = fit_result["Q_aug"]
    sigma2 = fit_result["sigma2"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H_ext, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    t_diag = T_aug.diagonal()
    t_h_stack = np.array([t_diag ** h for h in eval_horizons])

    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    d_h_stack = np.stack([d_all[h - 1:T_half + h - 1] for h in eval_horizons])
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)
    d_h_scan = d_h_stack.transpose(1, 0, 2)
    Z_update = Z_all[:T_half]
    d_update = d_all[:T_half]
    h_inv = 1.0 / sigma2
    state_dim = t_diag.shape[0]
    _eye_state = np.eye(state_dim)
    n_horizons = len(eval_horizons)
    preds_all = np.empty((T_half, n_horizons, N))

    a_t, P_t = a0.copy(), P0.copy()
    for t in range(T_half):
        Z_h_t = Z_h_scan[t]
        d_h_t = d_h_scan[t]
        Z_upd_t = Z_update[t]
        d_upd_t = d_update[t]
        y_h = y_test[t]

        a_h_stack = t_h_stack * a_t
        preds_all[t] = np.einsum("hnp,hp->hn", Z_h_t, a_h_stack) + d_h_t

        a_pred = t_diag * a_t
        P_pred = t_diag[:, None] * P_t * t_diag + Q_aug
        mask_h = ~np.isnan(y_h)
        y_hat = Z_upd_t @ a_pred + d_upd_t
        v = (y_h - y_hat) * mask_h
        Z_masked = np.where(mask_h[:, None], Z_upd_t, 0.0)
        PZm = P_pred @ Z_masked.T
        Inner = _eye_state + h_inv * PZm @ Z_masked
        sol = np.linalg.solve(Inner, PZm @ np.column_stack([v, Z_masked]))
        c, D = sol[:, 0], sol[:, 1:]
        a_t = a_pred + h_inv * c
        P_t = P_pred - h_inv * D @ P_pred

    return preds_all.transpose(1, 0, 2)


def forecast_h(fit_result, M):
    a0 = fit_result["att"][-1]
    T_aug = fit_result["T_aug"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    H, N, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    t_diag = T_aug.diagonal()
    predictions = np.empty((H, N))
    P_means = np.empty(H)

    a_t = a0.copy()
    for t in range(H):
        a_pred = t_diag * a_t
        y_hat = Z_all[t] @ a_pred + d_all[t]
        predictions[t] = y_hat
        P_means[t] = y_hat.sum() / N
        a_t = a_pred

    return predictions, P_means


def simulate_panel(params, M, n, rng, b_0=None):
    T_aug = params["T_aug"]
    Q_aug = params["Q_aug"]
    sigma2 = params["sigma2"]
    omega = params["omega"]
    beta_bar = params["beta_bar"]
    ws = params["ws"]
    state_dim = T_aug.shape[0]

    M = np.asarray(M, dtype=float)
    T_obs, N_obs, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(T_obs, N_obs, -1)
    d = np.einsum("tnp,p->tn", M_base, beta_bar)

    if b_0 is None:
        b_0 = np.zeros(state_dim, dtype=float)

    sqrt_sigma = np.sqrt(sigma2)
    t_diag = T_aug.diagonal()
    L_Q = np.linalg.cholesky(Q_aug)
    y_paths = np.empty((n, T_obs, N_obs))

    for i in range(n):
        eta_draws = rng.standard_normal(size=(T_obs, state_dim))
        eps_draws = rng.standard_normal(size=(T_obs, N_obs))
        b_t = b_0.copy()
        for t in range(T_obs):
            y_paths[i, t] = Z[t] @ b_t + d[t] + sqrt_sigma * eps_draws[t]
            b_t = t_diag * b_t + L_Q @ eta_draws[t]

    return y_paths

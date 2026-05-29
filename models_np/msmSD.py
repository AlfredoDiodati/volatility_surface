import numpy as np
from scipy.special import gammaln, ndtri, betainc, expit, logsumexp
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


def _t_unit_var_ppf(alpha, nu):
    a = nu / 2.0
    p = np.where(alpha <= 0.5, alpha, 1.0 - alpha)
    z = ndtri(p)
    x = z + (z**3 + z) / (4.0 * nu)
    for _ in range(15):
        x2 = x * x
        u = nu / (nu + x2)
        cdf = 0.5 * betainc(a, 0.5, u)
        log_pdf = (
            gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * nu)
            - ((nu + 1.0) / 2.0) * np.log1p(x2 / nu)
        )
        x = x - (cdf - p) / np.exp(log_pdf)
    q_std = np.where(alpha <= 0.5, x, -x)
    return q_std * np.sqrt((nu - 2.0) / nu)


def _build_msm_states(K, m0):
    h = 2 ** K
    bits = np.arange(h, dtype=np.int32)
    bit_matrix = (bits[:, None] >> np.arange(K, dtype=np.int32)[None, :]) & 1
    return np.prod(np.where(bit_matrix, 2.0 - m0, m0), axis=1)


def _build_log_transition_tensor(gamma_k):
    I2 = np.eye(2)
    outer = np.full((2, 2), 0.5)
    return np.array([np.log((1.0 - gk) * I2 + gk * outer) for gk in gamma_k])


def _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_flat):
    log_pi = log_pi_flat.copy()
    for k in range(log_P.shape[0]):
        log_pi_k = log_P[k]
        from0 = log_pi[idx0_all[k]] + log_pi_k[0, bits_matrix[k]]
        from1 = log_pi[idx1_all[k]] + log_pi_k[1, bits_matrix[k]]
        log_pi = logsumexp(np.stack([from0, from1], axis=0), axis=0)
    return log_pi


def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, score_power, state0):
    b_diag = np.diag(params["B"])
    A_diag = np.diag(params["A"])
    sigma2 = params["sigma2"]
    sigma_0 = params["sigma_0"]
    omega_load = params["omega_load"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]

    p_tilde = beta_bar.shape[0]
    p_full = p_tilde + 1
    h_inv = 1.0 / sigma2
    L_C = np.linalg.cholesky(params["C"])
    _eye_pfull = np.eye(p_full)
    _log_sigma2 = np.log(sigma2)

    h_states = 2 ** K
    g_vals = _build_msm_states(K, params["m0"])
    log_P = _build_log_transition_tensor(params["gamma_k"])

    j_range = np.arange(h_states, dtype=np.int32)
    k_range = np.arange(K, dtype=np.int32)
    masks = np.int32(1) << k_range
    bits_matrix = (j_range[None, :] >> k_range[:, None]) & 1
    idx0_all = j_range[None, :] & (~masks[:, None])
    idx1_all = j_range[None, :] | masks[:, None]

    log_pi_prev, beta_tilde_t = state0
    log_pi_prev = np.asarray(log_pi_prev, dtype=float)
    beta_tilde_t = np.asarray(beta_tilde_t, dtype=float).copy()
    T = y_masked.shape[0]
    log_liks = np.empty(T)
    betas_prev = np.empty((T, p_full))

    for t in range(T):
        log_pi_pred = _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_prev)
        sigma_vals = sigma_0 * g_vals

        omega_col = omega_load[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        mask_t = mask_f[t]
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)

        ZtHinvZ = h_inv * (Z_mask.T @ Z_mask)
        WLC = ZtHinvZ @ L_C
        Inner_mat = _eye_pfull + L_C.T @ WLC
        L_Inner = np.linalg.cholesky(Inner_mat)
        log_det_Sigma = N_t * _log_sigma2 + 2.0 * np.sum(np.log(np.diag(L_Inner)))
        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

        beta_full_base = np.concatenate([beta_tilde_t, np.zeros(1)])
        eps_base = (y_masked[t] - Z_t @ beta_full_base) * mask_t
        omega_col_masked = omega_col * mask_t

        ZtHinv_eps_base = h_inv * (Z_mask.T @ eps_base)
        ZtHinv_omega = h_inv * (Z_mask.T @ omega_col_masked)
        eps_base_sq = h_inv * np.dot(eps_base, eps_base)
        cross = h_inv * np.dot(eps_base, omega_col_masked)
        omega_sq = h_inv * np.dot(omega_col_masked, omega_col_masked)
        wb_base = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps_base, lower=True)
        wb_omega = solve_triangular(L_Inner, L_C.T @ ZtHinv_omega, lower=True)

        wb_all = wb_base[None, :] - sigma_vals[:, None] * wb_omega[None, :]
        mahal_all = (eps_base_sq - 2.0 * sigma_vals * cross + sigma_vals**2 * omega_sq
                     - np.sum(wb_all**2, axis=1))
        ll_all = (
            gammaln((nu + N_t) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * (N_t * np.log((nu - 2.0) * np.pi) + log_det_Sigma
                     + (nu + N_t) * np.log1p(mahal_all / (nu - 2.0)))
        )
        ZtSigmaInv_eps_all = ((ZtHinv_eps_base[None, :] - sigma_vals[:, None] * ZtHinv_omega[None, :])
                              - wb_all @ V_fisher)
        nabla_all = ((nu + N_t) / ((nu - 2.0) + mahal_all))[:, None] * ZtSigmaInv_eps_all[:, :p_tilde]

        log_weights = log_pi_pred + ll_all
        log_marg = logsumexp(log_weights)
        log_pi_t = log_weights - log_marg

        pi_t = np.exp(log_pi_t)
        nabla_t = (pi_t[:, None] * nabla_all).sum(axis=0)

        fisher_tilde = fisher_t[:p_tilde, :p_tilde]
        eigvals, eigvecs = np.linalg.eigh(fisher_tilde)
        scaling_matrix = (eigvecs * (eigvals ** (-score_power))) @ eigvecs.T
        xi_tilde = A_diag * (scaling_matrix @ nabla_t)

        sigma_t = (pi_t * sigma_vals).sum()
        betas_prev[t] = np.concatenate([beta_tilde_t, np.array([sigma_t])])
        log_liks[t] = log_marg

        log_pi_prev = log_pi_t
        beta_tilde_t = beta_bar + b_diag * (beta_tilde_t - beta_bar) + xi_tilde

    pi_T = np.exp(log_pi_prev)
    sigma_T = sigma_0 * (pi_T * g_vals).sum()
    beta_T = np.concatenate([beta_tilde_t, np.array([sigma_T])])
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, log_liks, (log_pi_prev, beta_tilde_t)


def fit(data, M, initial_guess, K, score_power, opt_options=None, maxiter=5000):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p_tilde = initial_guess["beta_bar"].shape[0]
    p_full = p_tilde + 1
    n_buckets = initial_guess["omega_load"].shape[0]
    maxiter = int(maxiter)
    opt_options = opt_options or {}
    mask_bool = ~np.isnan(data)
    y_masked = np.where(mask_bool, data, 0.0)
    mask_f = mask_bool.astype(float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    log_pi_init = np.full(2 ** K, -K * np.log(2.0))

    def _link(theta):
        theta = np.asarray(theta, dtype=float)
        i = 0
        beta_bar = theta[i:i + p_tilde]; i += p_tilde
        B = np.diag(np.tanh(theta[i:i + p_tilde])); i += p_tilde
        A = np.diag(theta[i:i + p_tilde]); i += p_tilde
        sigma2 = np.exp(theta[i]); i += 1
        sigma_0 = np.exp(theta[i]); i += 1
        omega_load = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        C = np.diag(np.exp(theta[i:i + p_full])); i += p_full
        nu = np.exp(theta[i]) + 2.0; i += 1
        m0 = 1.0 + expit(theta[i]); i += 1
        gamma_K = expit(theta[i]); i += 1
        b = 1.0 + np.exp(theta[i]); i += 1
        k_idx = np.arange(1, K + 1, dtype=float)
        gamma_k = 1.0 - (1.0 - gamma_K) ** (b ** (k_idx - K))
        return {
            "beta_bar": beta_bar, "B": B, "A": A,
            "sigma2": sigma2, "sigma_0": sigma_0,
            "omega_load": omega_load, "C": C, "nu": nu,
            "m0": m0, "gamma_K": gamma_K, "b": b, "gamma_k": gamma_k,
        }

    def _invlink(params):
        b_diag = np.diag(params["B"])
        return np.concatenate([
            params["beta_bar"],
            np.arctanh(np.clip(b_diag, -0.999999, 0.999999)),
            np.diag(params["A"]),
            np.array([np.log(params["sigma2"]), np.log(params["sigma_0"])]),
            params["omega_load"][1:],
            np.log(np.diag(params["C"])),
            np.array([
                np.log(params["nu"] - 2.0),
                np.log(params["m0"] - 1.0) - np.log(2.0 - params["m0"]),
                np.log(params["gamma_K"]) - np.log(1.0 - params["gamma_K"]),
                np.log(params["b"] - 1.0),
            ]),
        ])

    def _criterion(theta):
        params = _link(theta)
        _, lls, _ = _filter(
            y_masked, base_covariates, bucket_indices, mask_f,
            params, K, score_power, (log_pi_init, params["beta_bar"])
        )
        return -np.sum(lls)

    theta0 = np.asarray(_invlink(initial_guess))
    result = minimize(_criterion, theta0, method="L-BFGS-B", options={"maxiter": maxiter, **opt_options})
    params_opt = _link(result.x)
    betas, _, (log_pi_T, beta_tilde_T) = _filter(
        y_masked, base_covariates, bucket_indices, mask_f,
        params_opt, K, score_power, (log_pi_init, params_opt["beta_bar"])
    )
    return params_opt | {
        "betas": betas,
        "log_pi_T": log_pi_T,
        "beta_tilde_T": beta_tilde_T,
        "log_likelihood": -result.fun,
        "niter": result.nit,
        "is_converged": result.success,
        "score_power": score_power,
        "K": K,
    }


def forecast_h(fit_result, M, K):
    beta_bar = fit_result["beta_bar"]
    b_diag = np.diag(fit_result["B"])
    sigma_0 = fit_result["sigma_0"]
    omega_load = fit_result["omega_load"]

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    n_h = base_covariates.shape[1]
    T = base_covariates.shape[0]

    g_vals = _build_msm_states(K, fit_result["m0"])
    log_P = _build_log_transition_tensor(fit_result["gamma_k"])
    h_states = 2 ** K
    j_range = np.arange(h_states, dtype=np.int32)
    k_range = np.arange(K, dtype=np.int32)
    masks = np.int32(1) << k_range
    bits_matrix = (j_range[None, :] >> k_range[:, None]) & 1
    idx0_all = j_range[None, :] & (~masks[:, None])
    idx1_all = j_range[None, :] | masks[:, None]

    log_pi_t = fit_result["log_pi_T"].copy()
    beta_tilde_t = fit_result["beta_tilde_T"].copy()
    predictions = np.empty((T, n_h))
    P = np.empty(T)

    for t in range(T):
        log_pi_pred = _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_t)
        pi_pred = np.exp(log_pi_pred)
        sigma_t = sigma_0 * (pi_pred * g_vals).sum()
        beta_full_t = np.concatenate([beta_tilde_t, np.array([sigma_t])])
        beta_tilde_next = beta_bar + b_diag * (beta_tilde_t - beta_bar)
        omega_col = omega_load[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        predictions_t = Z_t @ beta_full_t
        predictions[t] = predictions_t
        P[t] = predictions_t.sum() / n_h
        log_pi_t = log_pi_pred
        beta_tilde_t = beta_tilde_next

    return predictions, P


def forecast_rolling_h(fit_result, M, y_test, K, score_power, eval_horizons):
    B = fit_result["B"]
    beta_bar = fit_result["beta_bar"]
    sigma_0 = fit_result["sigma_0"]
    omega_load = fit_result["omega_load"]
    nu = fit_result["nu"]
    sigma2 = fit_result["sigma2"]
    b_diag = np.diag(B)
    A_diag = np.diag(fit_result["A"])

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape
    p_tilde = beta_bar.shape[0]
    p_full = p_tilde + 1
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cols = omega_load[bucket_indices]
    Z_all = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)

    g_vals = _build_msm_states(K, fit_result["m0"])
    log_P = _build_log_transition_tensor(fit_result["gamma_k"])
    h_states = 2 ** K
    j_range = np.arange(h_states, dtype=np.int32)
    k_range = np.arange(K, dtype=np.int32)
    masks = np.int32(1) << k_range
    bits_matrix = (j_range[None, :] >> k_range[:, None]) & 1
    idx0_all = j_range[None, :] & (~masks[:, None])
    idx1_all = j_range[None, :] | masks[:, None]

    b_h_stack = np.array([b_diag ** (h - 1) for h in eval_horizons])
    n_horizons = len(eval_horizons)

    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)

    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    h_inv = 1.0 / sigma2
    L_C = np.linalg.cholesky(fit_result["C"])
    _eye_pfull = np.eye(p_full)
    _log_sigma2 = np.log(sigma2)

    preds_all = np.empty((T_half, n_horizons, N))
    log_pi_t = fit_result["log_pi_T"].copy()
    beta_tilde_t = fit_result["beta_tilde_T"].copy()

    for t in range(T_half):
        dev_tilde = beta_tilde_t - beta_bar
        beta_tilde_h_all = beta_bar + b_h_stack * dev_tilde[None, :]

        sigma_h_all = np.empty(n_horizons)
        for i, h in enumerate(eval_horizons):
            log_pi_h = log_pi_t.copy()
            for _ in range(h - 1):
                log_pi_h = _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_h)
            sigma_h_all[i] = sigma_0 * (np.exp(log_pi_h) * g_vals).sum()

        beta_full_h_all = np.concatenate([beta_tilde_h_all, sigma_h_all[:, None]], axis=-1)
        preds_all[t] = np.einsum("hnp,hp->hn", Z_h_scan[t], beta_full_h_all)

        log_pi_pred = _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_t)

        omega_col = omega_load[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        mask_t = mask_f[t]
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)

        ZtHinvZ = h_inv * (Z_mask.T @ Z_mask)
        WLC = ZtHinvZ @ L_C
        Inner_mat = _eye_pfull + L_C.T @ WLC
        L_Inner = np.linalg.cholesky(Inner_mat)
        log_det_Sigma = N_t * _log_sigma2 + 2.0 * np.sum(np.log(np.diag(L_Inner)))
        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

        sigma_vals = sigma_0 * g_vals
        beta_full_base = np.concatenate([beta_tilde_t, np.zeros(1)])
        eps_base = (y_masked[t] - Z_t @ beta_full_base) * mask_t
        omega_col_masked = omega_col * mask_t

        ZtHinv_eps_base = h_inv * (Z_mask.T @ eps_base)
        ZtHinv_omega = h_inv * (Z_mask.T @ omega_col_masked)
        eps_base_sq = h_inv * np.dot(eps_base, eps_base)
        cross = h_inv * np.dot(eps_base, omega_col_masked)
        omega_sq = h_inv * np.dot(omega_col_masked, omega_col_masked)
        wb_base = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps_base, lower=True)
        wb_omega = solve_triangular(L_Inner, L_C.T @ ZtHinv_omega, lower=True)

        wb_all = wb_base[None, :] - sigma_vals[:, None] * wb_omega[None, :]
        mahal_all = (eps_base_sq - 2.0 * sigma_vals * cross + sigma_vals**2 * omega_sq
                     - np.sum(wb_all**2, axis=1))
        ll_all = (
            gammaln((nu + N_t) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * (N_t * np.log((nu - 2.0) * np.pi) + log_det_Sigma
                     + (nu + N_t) * np.log1p(mahal_all / (nu - 2.0)))
        )
        ZtSigmaInv_eps_all = ((ZtHinv_eps_base[None, :] - sigma_vals[:, None] * ZtHinv_omega[None, :])
                              - wb_all @ V_fisher)
        nabla_all = ((nu + N_t) / ((nu - 2.0) + mahal_all))[:, None] * ZtSigmaInv_eps_all[:, :p_tilde]

        log_weights = log_pi_pred + ll_all
        log_pi_next = log_weights - logsumexp(log_weights)

        pi_next = np.exp(log_pi_next)
        nabla_t = (pi_next[:, None] * nabla_all).sum(axis=0)
        fisher_tilde = fisher_t[:p_tilde, :p_tilde]
        eigvals, eigvecs = np.linalg.eigh(fisher_tilde)
        scaling_matrix = (eigvecs * (eigvals ** (-score_power))) @ eigvecs.T
        xi_tilde = A_diag * (scaling_matrix @ nabla_t)

        log_pi_t = log_pi_next
        beta_tilde_t = beta_bar + b_diag * (beta_tilde_t - beta_bar) + xi_tilde

    return preds_all.transpose(1, 0, 2)


def forecast(fit_result, M, y_test, K, score_power, alpha):
    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = M.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, _ = _filter(
        y_masked, base_covariates, bucket_indices, mask_f,
        fit_result, K, score_power,
        state0=(fit_result["log_pi_T"], fit_result["beta_tilde_T"])
    )

    omega_cols = fit_result["omega_load"][bucket_indices]
    Z = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    predictions = np.einsum("hni,hi->hn", Z, betas[:H])

    n_obs_h = mask_bool.sum(axis=1)
    P = predictions.sum(axis=1) / n_obs_h
    z_sum = Z.sum(axis=1)
    F_sum = n_obs_h * fit_result["sigma2"] + np.einsum("hi,ij,hj->h", z_sum, fit_result["C"], z_sum)
    VaR = P + _t_unit_var_ppf(alpha, fit_result["nu"]) / n_obs_h * np.sqrt(F_sum)

    return predictions, P, VaR, log_liks


def simulate_panel(params, M, n, rng, K, m_k_0=None, beta_tilde_0=None):
    b_diag = np.diag(params["B"])
    A_diag = np.diag(params["A"])
    sigma2 = params["sigma2"]
    sigma_0 = params["sigma_0"]
    omega_load = params["omega_load"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    m0 = params["m0"]
    gamma_k = params["gamma_k"]
    score_power = params["score_power"]
    p_tilde = beta_bar.shape[0]
    p_full = p_tilde + 1

    M = np.asarray(M, dtype=float)
    T, N_obs, _ = M.shape
    base = M[:, :, :-1]
    bidx = M[:, :, -1].astype(np.int32)
    design = np.concatenate([base, omega_load[bidx][:, :, None]], axis=-1)

    if m_k_0 is None: m_k_0 = np.full(K, m0)
    if beta_tilde_0 is None: beta_tilde_0 = beta_bar

    h_inv = 1.0 / sigma2
    sqrt_sigma = np.sqrt(sigma2)
    L_C = np.linalg.cholesky(params["C"])
    _eye_pfull = np.eye(p_full)

    y_paths = np.empty((n, T, N_obs))
    beta_paths = np.empty((n, T, p_full))
    msm_paths = np.empty((n, T, K))

    for i in range(n):
        switch_draws = rng.random((T, K)) < gamma_k
        direction_draws = rng.random((T, K)) < 0.5
        g_draws = rng.gamma(nu / 2.0, size=T)
        z_draws = rng.standard_normal((T, N_obs))
        w_draws = rng.standard_normal((T, p_full))

        m_k_t = np.asarray(m_k_0, dtype=float).copy()
        beta_tilde_t = np.asarray(beta_tilde_0, dtype=float).copy()

        for t in range(T):
            new_vals = np.where(direction_draws[t], 2.0 - m0, m0)
            m_k_t = np.where(switch_draws[t], new_vals, m_k_t)
            sigma_t = sigma_0 * np.prod(m_k_t)
            beta_full_t = np.concatenate([beta_tilde_t, np.array([sigma_t])])

            t_scale = np.sqrt(nu / (2.0 * g_draws[t]))
            design_t = design[t]
            eps_t = t_scale * (sqrt_sigma * z_draws[t] + design_t @ (L_C @ w_draws[t]))
            y_t = design_t @ beta_full_t + eps_t

            ZtHinvZ = h_inv * (design_t.T @ design_t)
            WLC = ZtHinvZ @ L_C
            Inner_mat = _eye_pfull + L_C.T @ WLC
            L_Inner = np.linalg.cholesky(Inner_mat)
            V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
            ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
            fisher_tilde = (nu / (nu - 2.0)) * ((nu + N_obs) / (nu + N_obs + 2.0)) * ZtSigmaInvZ[:p_tilde, :p_tilde]

            ZtHinv_eps = h_inv * (design_t.T @ eps_t)
            wb = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
            mahal = h_inv * np.dot(eps_t, eps_t) - np.dot(wb, wb)
            ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ wb
            nabla_t = ((nu + N_obs) / ((nu - 2.0) + mahal)) * ZtSigmaInv_eps[:p_tilde]

            eigvals, eigvecs = np.linalg.eigh(fisher_tilde)
            xi_tilde = A_diag * ((eigvecs * (eigvals ** (-score_power))) @ eigvecs.T @ nabla_t)

            beta_paths[i, t] = beta_full_t
            y_paths[i, t] = y_t
            msm_paths[i, t] = m_k_t
            beta_tilde_t = beta_bar + b_diag * (beta_tilde_t - beta_bar) + xi_tilde

    return y_paths, beta_paths, msm_paths

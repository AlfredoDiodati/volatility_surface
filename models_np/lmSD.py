import numpy as np
from scipy.special import gammaln, ndtri, betainc, expit
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


def _compute_weights(d, T):
    k = np.arange(1, T, dtype=float)
    ratios = (k[:, None] - 1.0 - d[None, :]) / k[:, None]
    w_rest = np.cumprod(ratios, axis=0)
    return np.concatenate([np.ones((1, len(d))), w_rest], axis=0)


def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, state0):
    b_diag = np.diag(params["B"])
    A_diag = np.diag(params["A"])
    sigma2 = params["sigma2"]
    omega = params["omega"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C
    diag_C_inv = np.diag(C_inv)
    sum_log_C = np.sum(np.log(C))

    beta0, score_buf0, write_idx0 = state0
    T_buf = score_buf0.shape[0]
    T = y_masked.shape[0]
    p = beta_bar.shape[0]
    weights = _compute_weights(params["d"], T_buf)
    _k_range = np.arange(T_buf)

    betas = np.empty((T, p))
    log_liks = np.empty(T)

    score_buf = score_buf0.copy()
    beta_t = np.asarray(beta0, dtype=float).copy()
    write_idx = int(write_idx0)

    for t in range(T):
        omega_col = omega[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        mask_t = mask_f[t]
        Z_mask = Z_t * mask_t[:, None]
        eps_t = (y_masked[t] - Z_t @ beta_t) * mask_t
        N_t = np.sum(mask_t)

        V_t = h_inv * (Z_mask.T @ Z_mask)
        G_t = h_inv * (Z_mask.T @ eps_t)
        S = diag_C_inv + V_t

        mahal_H = h_inv * np.dot(eps_t, eps_t)
        S_inv = np.linalg.solve(S, np.concatenate([G_t[:, None], V_t], axis=1))
        S_inv_G = S_inv[:, 0]
        S_inv_V = S_inv[:, 1:]
        mahal_F = mahal_H - G_t @ S_inv_G

        weight = (1.0 + (N_t + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
        g_tilde = G_t - V_t @ S_inv_G
        V_tilde = V_t - V_t @ S_inv_V
        s_t = weight * np.linalg.solve(V_tilde, g_tilde)

        score_buf[write_idx] = s_t
        positions = (write_idx - _k_range) % T_buf
        conv = A_diag * (weights * score_buf[positions]).sum(axis=0)
        write_idx = (write_idx + 1) % T_buf

        log_det_F = N_t * np.log(sigma2) + sum_log_C + np.linalg.slogdet(S)[1]
        log_constants = gammaln((nu + N_t) / 2.0) - gammaln(nu / 2.0) - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
        log_liks[t] = log_constants - 0.5 * log_det_F - 0.5 * (nu + N_t) * np.log(1.0 + mahal_F / (nu - 2.0))

        betas[t] = beta_t
        beta_t = beta_bar + b_diag * (beta_t - beta_bar) + conv

    return betas, log_liks, beta_t, score_buf, write_idx


def fit(data, M, initial_guess, opt_options=None, maxiter=5000):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    T = data.shape[0]
    maxiter = int(maxiter)
    opt_options = opt_options or {}
    mask_bool = ~np.isnan(data)
    y_masked = np.where(mask_bool, data, 0.0)
    mask_f = mask_bool.astype(float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)

    def _link(theta):
        theta = np.asarray(theta, dtype=float)
        i = 0
        beta_bar = theta[i:i + p]; i += p
        B = np.diag(np.tanh(theta[i:i + p])); i += p
        A = np.diag(theta[i:i + p]); i += p
        d = 0.5*np.tanh(theta[i:i + p]); i += p
        sigma2 = np.exp(theta[i]); i += 1
        omega = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        C_diag = np.exp(theta[i:i + p]); i += p
        nu = np.exp(theta[i]) + 2.0
        return {"beta_bar": beta_bar, "B": B, "A": A, "d": d, "sigma2": sigma2,
                "omega": omega, "C": C_diag, "nu": nu}

    def _invlink(params):
        B_diag = np.diag(params["B"])
        unc_B = np.arctanh(np.clip(B_diag, -0.999999, 0.999999))
        unc_A = np.diag(params["A"])
        unc_d = np.arctanh(2.0*params["d"])
        unc_s2 = np.log(params["sigma2"])
        unc_omega = params["omega"][1:]
        unc_C = np.log(params["C"])
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([
            params["beta_bar"], unc_B, unc_A, unc_d,
            np.array([unc_s2]), unc_omega, unc_C, np.array([unc_nu]),
        ])

    def _criterion(theta):
        params = _link(theta)
        _, lls, _, _, _ = _filter(
            y_masked, base_covariates, bucket_indices, mask_f,
            params, (params["beta_bar"], np.zeros((T, p)), 0)
        )
        return -np.sum(lls)

    theta0 = np.asarray(_invlink(initial_guess))
    result = minimize(_criterion, theta0, method="L-BFGS-B", options={"maxiter": maxiter, **opt_options})
    params_opt = _link(result.x)
    betas, _, beta_T, score_buf_circ, write_idx_T = _filter(
        y_masked, base_covariates, bucket_indices, mask_f,
        params_opt, (params_opt["beta_bar"], np.zeros((T, p)), 0)
    )
    score_buf_T = score_buf_circ[(write_idx_T - 1 - np.arange(T)) % T]
    return params_opt | {
        "betas": betas,
        "beta_T": beta_T,
        "score_buf_T": score_buf_T,
        "log_likelihood": -result.fun,
        "niter": result.nit,
        "is_converged": result.success,
    }


def simulate(params, M, horizon, rng, beta_0=None, score_buf_size=None, score_buf_0=None):
    b_diag = np.diag(params["B"])
    A_diag = np.diag(params["A"])
    d = params["d"]
    sigma2 = params["sigma2"]
    omega = params["omega"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    p = beta_bar.shape[0]

    M = np.asarray(M, dtype=float)
    base_fixed = M[:, :-1]
    bidx_fixed = M[:, -1].astype(np.int32)
    design = np.concatenate([base_fixed, omega[bidx_fixed][:, None]], axis=-1)
    N = design.shape[0]

    if score_buf_size is None: score_buf_size = horizon
    if beta_0 is None: beta_0 = beta_bar
    if score_buf_0 is None: score_buf_0 = np.zeros((score_buf_size, p))

    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C
    sqrt_C = np.sqrt(C)
    sqrt_sigma = np.sqrt(sigma2)

    V = h_inv * (design.T @ design)
    S_base = np.diag(C_inv) + V
    S_base_inv = np.linalg.inv(S_base)
    V_tilde = V - V @ (S_base_inv @ V)
    V_tilde_inv = np.linalg.inv(V_tilde)

    weights = _compute_weights(d, score_buf_size)
    N_sim = score_buf_0.shape[0]
    _k_sim = np.arange(N_sim)

    g_samp = rng.chisquare(nu, size=horizon)
    w_samp = rng.standard_normal(size=(horizon, p))
    z_samp = rng.standard_normal(size=(horizon, N))

    y_sim = np.empty((horizon, N))
    beta_sim = np.empty((horizon, p))

    score_buf = score_buf_0.copy()
    beta_t = np.asarray(beta_0, dtype=float).copy()
    write_idx = 0

    for t in range(horizon):
        scale = np.sqrt((nu - 2.0) / g_samp[t])
        eps_t = scale * (design @ (sqrt_C * w_samp[t]) + sqrt_sigma * z_samp[t])
        y_t = design @ beta_t + eps_t

        G_t = h_inv * (design.T @ eps_t)
        S_inv_G = S_base_inv @ G_t
        mahal_H = h_inv * np.dot(eps_t, eps_t)
        mahal_F = mahal_H - G_t @ S_inv_G
        wt = (1.0 + (N + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
        g_tilde = G_t - V @ S_inv_G
        s_t = wt * (V_tilde_inv @ g_tilde)

        score_buf[write_idx] = s_t
        positions = (write_idx - _k_sim) % N_sim
        conv = A_diag * (weights * score_buf[positions]).sum(axis=0)
        write_idx = (write_idx + 1) % N_sim

        beta_sim[t] = beta_t
        y_sim[t] = y_t
        beta_t = beta_bar + b_diag * (beta_t - beta_bar) + conv

    return y_sim, beta_sim


def forecast_rolling_h(fit_result, M, y_test, eval_horizons):
    B = fit_result["B"]
    beta_bar = fit_result["beta_bar"]
    omega = fit_result["omega"]
    p = beta_bar.shape[0]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    score_buf_ext = np.concatenate([fit_result["score_buf_T"], np.zeros((T_half, p))], axis=0)
    T_train = fit_result["score_buf_T"].shape[0]
    state0 = (fit_result["beta_T"], score_buf_ext, T_train)

    betas_origins, _, _, _, _ = _filter(
        y_masked, base_covariates[:T_half], bucket_indices[:T_half],
        mask_f, fit_result, state0,
    )

    b_diag = np.diag(B)
    b_h_stack = np.array([b_diag ** (h - 1) for h in eval_horizons])
    deviations = betas_origins - beta_bar
    beta_h = beta_bar + b_h_stack[:, None, :] * deviations[None, :, :]

    omega_cols = omega[bucket_indices]
    Z_all = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    return np.einsum("htnp,htp->htn", Z_h_stack, beta_h)


def forecast_h(fit_result, M):
    B = fit_result["B"]
    A = fit_result["A"]
    beta_bar = fit_result["beta_bar"]
    omega = fit_result["omega"]
    p = beta_bar.shape[0]
    b_diag = np.diag(B)
    A_diag = np.diag(A)

    M = np.asarray(M, dtype=float)
    H = M.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    n_h = base_covariates.shape[1]

    score_buf_T = fit_result["score_buf_T"]
    T_buf = score_buf_T.shape[0]
    score_buf_init = np.concatenate([score_buf_T, np.zeros((H, p))], axis=0)

    N_buf = T_buf + H
    weights = _compute_weights(fit_result["d"], N_buf)
    _k_fh = np.arange(N_buf)

    predictions = np.empty((H, n_h))
    P = np.empty(H)

    beta_h = fit_result["beta_T"].copy()
    read_idx = 0

    for t in range(H):
        read_idx_new = (read_idx + 1) % N_buf
        positions = (read_idx_new - _k_fh) % N_buf
        conv = A_diag * (weights * score_buf_init[positions]).sum(axis=0)
        beta_next = beta_bar + b_diag * (beta_h - beta_bar) + conv
        omega_col = omega[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        predictions_t = Z_t @ beta_next
        predictions[t] = predictions_t
        P[t] = predictions_t.sum() / n_h
        beta_h = beta_next
        read_idx = read_idx_new

    return predictions, P


def simulate_panel(params, M, n, rng, beta_0=None, score_buf_size=None, score_buf_0=None):
    b_diag = np.diag(params["B"])
    A_diag = np.diag(params["A"])
    d = params["d"]
    sigma2 = params["sigma2"]
    omega = params["omega"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    p = beta_bar.shape[0]

    M = np.asarray(M, dtype=float)
    T, N_obs, _ = M.shape
    base = M[:, :, :-1]
    bidx = M[:, :, -1].astype(np.int32)
    design = np.concatenate([base, omega[bidx][:, :, None]], axis=-1)

    if score_buf_size is None: score_buf_size = T
    if beta_0 is None: beta_0 = beta_bar
    if score_buf_0 is None: score_buf_0 = np.zeros((score_buf_size, p))

    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C
    diag_C_inv = np.diag(C_inv)
    sqrt_C = np.sqrt(C)
    sqrt_sigma = np.sqrt(sigma2)

    weights = _compute_weights(d, score_buf_size)
    N_pan = score_buf_0.shape[0]
    _k_pan = np.arange(N_pan)

    y_paths = np.empty((n, T, N_obs))
    beta_paths = np.empty((n, T, p))

    for i in range(n):
        g_samp = rng.chisquare(nu, size=T)
        w_samp = rng.standard_normal(size=(T, p))
        z_samp = rng.standard_normal(size=(T, N_obs))

        score_buf = score_buf_0.copy()
        beta_t = np.asarray(beta_0, dtype=float).copy()
        write_idx = 0

        for t in range(T):
            design_t = design[t]
            V_t = h_inv * (design_t.T @ design_t)
            S_t = diag_C_inv + V_t

            scale = np.sqrt((nu - 2.0) / g_samp[t])
            eps_t = scale * (design_t @ (sqrt_C * w_samp[t]) + sqrt_sigma * z_samp[t])
            y_t = design_t @ beta_t + eps_t

            G_t = h_inv * (design_t.T @ eps_t)
            S_inv = np.linalg.solve(S_t, np.concatenate([G_t[:, None], V_t], axis=1))
            S_inv_G = S_inv[:, 0]
            S_inv_V_t = S_inv[:, 1:]
            V_tilde_t = V_t - V_t @ S_inv_V_t
            mahal_H = h_inv * np.dot(eps_t, eps_t)
            mahal_F = mahal_H - G_t @ S_inv_G
            wt = (1.0 + (N_obs + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
            g_tilde = G_t - V_t @ S_inv_G
            s_t = wt * np.linalg.solve(V_tilde_t, g_tilde)

            score_buf[write_idx] = s_t
            positions = (write_idx - _k_pan) % N_pan
            conv = A_diag * (weights * score_buf[positions]).sum(axis=0)
            write_idx = (write_idx + 1) % N_pan

            beta_paths[i, t] = beta_t
            y_paths[i, t] = y_t
            beta_t = beta_bar + b_diag * (beta_t - beta_bar) + conv

    return y_paths, beta_paths


def forecast(fit_result, M, y_test, alpha):
    p = fit_result["beta_bar"].shape[0]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = M.shape[0]
    score_buf_init = np.concatenate([
        fit_result["score_buf_T"],
        np.zeros((H, p)),
    ], axis=0)
    T_train = fit_result["score_buf_T"].shape[0]
    state0 = (fit_result["beta_T"], score_buf_init, T_train)

    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, _, _, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, fit_result, state0)

    omega_cols = fit_result["omega"][bucket_indices]
    Z = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    predictions = np.einsum("hni,hi->hn", Z, betas[:H])

    n_obs_h = mask_bool.sum(axis=1)
    P = predictions.sum(axis=1) / n_obs_h
    z_sum = Z.sum(axis=1)
    F_sum = n_obs_h * fit_result["sigma2"] + (z_sum ** 2 * fit_result["C"]).sum(axis=1)
    q = _t_unit_var_ppf(alpha, fit_result["nu"])
    VaR = P + q / n_obs_h * np.sqrt(F_sum)

    return predictions, P, VaR, log_liks

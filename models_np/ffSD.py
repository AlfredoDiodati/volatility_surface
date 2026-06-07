import numpy as np
from scipy.special import gammaln, ndtri, betainc, expit
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


def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, state0):
    A_diag = np.diag(params["A"])
    sigma2 = params["sigma2"]
    omega_load = params["omega_load"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]

    p = beta_bar.shape[0]
    h_inv = 1.0 / sigma2
    L_C = np.linalg.cholesky(params["C"])
    _eye_p = np.eye(p)
    ws, lambdas = _solve_weights_ff(params["eta"], params["alpha"], K)
    exp_neg_lambdas = np.exp(-lambdas)
    _log_sigma2 = np.log(sigma2)

    T = y_masked.shape[0]
    betas_prev = np.empty((T, p))
    log_liks = np.empty(T)

    b_t = state0.copy()
    for t in range(T):
        beta_t = beta_bar + np.sum(ws * b_t, axis=0)

        omega_col = omega_load[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        mask_t = mask_f[t]
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)
        eps_t = (y_masked[t] - Z_t @ beta_t) * mask_t

        ZtHinvZ = h_inv * (Z_mask.T @ Z_mask)
        WLC = ZtHinvZ @ L_C
        Inner_mat = _eye_p + L_C.T @ WLC
        L_Inner = np.linalg.cholesky(Inner_mat)
        log_det_Sigma = N_t * _log_sigma2 + 2.0 * np.sum(np.log(np.diag(L_Inner)))

        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

        ZtHinv_eps = h_inv * (Z_mask.T @ eps_t)
        woodbury_eps = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
        mahal_Sigma = h_inv * np.dot(eps_t, eps_t) - np.dot(woodbury_eps, woodbury_eps)
        ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ woodbury_eps

        log_liks[t] = (
            gammaln((nu + N_t) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
            - 0.5 * log_det_Sigma
            - 0.5 * (nu + N_t) * np.log(1.0 + mahal_Sigma / (nu - 2.0))
        )

        score_weight = (nu + N_t) / ((nu - 2.0) + mahal_Sigma)
        nabla_t = score_weight * ZtSigmaInv_eps
        eigvals, eigvecs = np.linalg.eigh(fisher_t)
        scaled_score = A_diag * ((eigvecs * (eigvals ** -1.0)) @ eigvecs.T @ nabla_t)

        betas_prev[t] = beta_t
        b_t = exp_neg_lambdas * b_t + ws * scaled_score[None, :]

    beta_T = beta_bar + np.sum(ws * b_t, axis=0)
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, log_liks, b_t


def fit(data, M, initial_guess, K, opt_options=None, maxiter=5000):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega_load"].shape[0]
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
        A = np.diag(theta[i:i + p]); i += p
        sigma2 = np.exp(theta[i]); i += 1
        omega_load = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        eta = 2.0 * expit(theta[i:i + p]); i += p
        alpha = np.full(p, np.logaddexp(0.0, theta[i]) + 1.0); i += 1
        C = np.diag(np.exp(theta[i:i + p])); i += p
        nu = np.exp(theta[i]) + 2.0
        return {"beta_bar": beta_bar, "A": A, "sigma2": sigma2,
                "omega_load": omega_load, "eta": eta, "alpha": alpha, "C": C, "nu": nu}

    def _invlink(params):
        unc_s2 = np.log(params["sigma2"])
        unc_omega_load = params["omega_load"][1:]
        eta = params["eta"]
        unc_eta = np.log( eta / (2.0 - eta))
        unc_alpha = np.log(np.exp(params["alpha"][0] - 1.0) - 1.0)
        unc_C = np.log(np.diag(params["C"]))
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([
            params["beta_bar"], np.diag(params["A"]), np.array([unc_s2]),
            unc_omega_load, unc_eta, np.array([unc_alpha]),
            unc_C, np.array([unc_nu]),
        ])

    def _criterion(theta):
        params = _link(theta)
        try:
            _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, np.zeros((K + 1, p)))
            val = -np.sum(lls)
            return val if np.isfinite(val) else 1e20
        except (np.linalg.LinAlgError, ValueError):
            return 1e20

    theta0 = np.asarray(_invlink(initial_guess))
    result = minimize(_criterion, theta0, method="L-BFGS-B", options={"maxiter": maxiter, **opt_options})
    params_opt = _link(result.x)
    betas, _, b_T = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt, K, np.zeros((K + 1, p)))
    return params_opt | {
        "betas": betas,
        "b_T": b_T,
        "log_likelihood": -result.fun,
        "niter": result.nit,
        "is_converged": result.success,
    }


def standard_errors(fit_result, data, M, K):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = fit_result["beta_bar"].shape[0]
    n_buckets = fit_result["omega_load"].shape[0]

    mask_bool = ~np.isnan(data)
    y_masked = np.where(mask_bool, data, 0.0)
    mask_f = mask_bool.astype(float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)

    def _link_arr(theta):
        theta = np.asarray(theta, dtype=float)
        i = 0
        beta_bar = theta[i:i + p]; i += p
        A_diag = theta[i:i + p]; i += p
        sigma2 = np.exp(theta[i]); i += 1
        omega_load = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        eta = 2.0 * expit(theta[i:i + p]); i += p
        alpha_val = np.logaddexp(0.0, theta[i]) + 1.0; i += 1
        C_diag = np.exp(theta[i:i + p]); i += p
        nu = np.exp(theta[i]) + 2.0
        return np.concatenate([beta_bar, A_diag, np.array([sigma2]),
                                omega_load[1:], eta, np.array([alpha_val]),
                                C_diag, np.array([nu])])

    def _invlink(params):
        unc_s2 = np.log(params["sigma2"])
        unc_omega_load = params["omega_load"][1:]
        unc_eta = np.log(params["eta"] / (2.0 - params["eta"]))
        unc_alpha = np.log(np.exp(params["alpha"][0] - 1.0) - 1.0)
        unc_C = np.log(np.diag(params["C"]))
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([params["beta_bar"], np.diag(params["A"]), np.array([unc_s2]),
                                unc_omega_load, unc_eta, np.array([unc_alpha]),
                                unc_C, np.array([unc_nu])])

    def _criterion(theta):
        theta = np.asarray(theta, dtype=float)
        i = 0
        beta_bar = theta[i:i + p]; i += p
        A = np.diag(theta[i:i + p]); i += p
        sigma2 = np.exp(theta[i]); i += 1
        omega_load = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        eta = 2.0 * expit(theta[i:i + p]); i += p
        alpha = np.full(p, np.logaddexp(0.0, theta[i]) + 1.0); i += 1
        C = np.diag(np.exp(theta[i:i + p])); i += p
        nu = np.exp(theta[i]) + 2.0
        params = {"beta_bar": beta_bar, "A": A, "sigma2": sigma2,
                  "omega_load": omega_load, "eta": eta, "alpha": alpha, "C": C, "nu": nu}
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, np.zeros((K + 1, p)))
        return -np.sum(lls)

    theta_opt = _invlink(fit_result)
    n = len(theta_opt)
    eps = 1e-5

    def _central_grad(theta):
        g = np.zeros(n)
        for j in range(n):
            e_j = np.zeros(n); e_j[j] = eps
            g[j] = (_criterion(theta + e_j) - _criterion(theta - e_j)) / (2 * eps)
        return g

    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        H[i] = (_central_grad(theta_opt + e) - _central_grad(theta_opt - e)) / (2 * eps)
    H = (H + H.T) / 2

    eigvals, eigvecs = np.linalg.eigh(H)
    H_pd = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    cov_theta = np.linalg.inv(H_pd)

    f0 = _link_arr(theta_opt)
    m = len(f0)
    J = np.zeros((m, n))
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        J[:, i] = (_link_arr(theta_opt + e) - f0) / eps

    cov_natural = J @ cov_theta @ J.T
    se_flat = np.sqrt(np.abs(np.diag(cov_natural)))

    i = 0
    se = {}
    se["beta_bar"] = se_flat[i:i + p]; i += p
    se["A_diag"] = se_flat[i:i + p]; i += p
    se["sigma2"] = se_flat[i]; i += 1
    se["omega_load"] = se_flat[i:i + n_buckets - 1]; i += n_buckets - 1
    eta_idx = i
    se["eta"] = se_flat[i:i + p]; i += p
    se["alpha"] = se_flat[i:i + 1]; i += 1
    se["C_diag"] = se_flat[i:i + p]; i += p
    se["nu"] = se_flat[i]

    cov_eta = cov_natural[eta_idx:eta_idx + p, eta_idx:eta_idx + p]
    return se, cov_eta


def forecast(fit_result, M, y_test, K, alpha):
    state0 = fit_result["b_T"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = M.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, fit_result, K, state0)

    omega_cols = fit_result["omega_load"][bucket_indices]
    Z = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    predictions = np.einsum("hni,hi->hn", Z, betas[:H])

    n_obs_h = mask_bool.sum(axis=1)
    P = predictions.sum(axis=1) / n_obs_h
    z_sum = Z.sum(axis=1)
    F_sum = n_obs_h * fit_result["sigma2"] + np.einsum("hi,ij,hj->h", z_sum, fit_result["C"], z_sum)
    q = _t_unit_var_ppf(alpha, fit_result["nu"])
    VaR = P + q / n_obs_h * np.sqrt(F_sum)

    return predictions, P, VaR, log_liks


def forecast_rolling_h(fit_result, M, y_test, K, eval_horizons):
    beta_bar = fit_result["beta_bar"]
    omega_load = fit_result["omega_load"]
    nu = fit_result["nu"]
    sigma2 = fit_result["sigma2"]
    A_diag = np.diag(fit_result["A"])

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cols = omega_load[bucket_indices]
    M_base = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)

    ws, lambdas = _solve_weights_ff(fit_result["eta"], fit_result["alpha"], K)
    exp_neg_lambdas = np.exp(-lambdas)
    exp_neg_h_stack = np.array([np.exp(-(h - 1) * lambdas) for h in eval_horizons])

    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H_ext, N, -1)
    d_all = M_base @ beta_bar

    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    d_h_stack = np.stack([d_all[h - 1:T_half + h - 1] for h in eval_horizons])
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)
    d_h_scan = d_h_stack.transpose(1, 0, 2)

    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    p = beta_bar.shape[0]
    h_inv = 1.0 / sigma2
    L_C = np.linalg.cholesky(fit_result["C"])
    _eye_p = np.eye(p)
    n_horizons = len(eval_horizons)
    preds_all = np.empty((T_half, n_horizons, N))

    b_t = fit_result["b_T"].copy()
    for t in range(T_half):
        b_h_all = exp_neg_h_stack * b_t[None, :, :]
        preds_all[t] = np.einsum("hnp,hp->hn", Z_h_scan[t], b_h_all.reshape(n_horizons, -1)) + d_h_scan[t]

        beta_t = beta_bar + np.sum(ws * b_t, axis=0)
        omega_col = omega_load[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        mask_t = mask_f[t]
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)
        eps_t = (y_masked[t] - Z_t @ beta_t) * mask_t

        ZtHinvZ = h_inv * (Z_mask.T @ Z_mask)
        WLC = ZtHinvZ @ L_C
        Inner_mat = _eye_p + L_C.T @ WLC
        L_Inner = np.linalg.cholesky(Inner_mat)

        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

        ZtHinv_eps = h_inv * (Z_mask.T @ eps_t)
        woodbury_eps = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
        mahal_Sigma = h_inv * np.dot(eps_t, eps_t) - np.dot(woodbury_eps, woodbury_eps)
        ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ woodbury_eps

        score_weight = (nu + N_t) / ((nu - 2.0) + mahal_Sigma)
        nabla_t = score_weight * ZtSigmaInv_eps
        eigvals, eigvecs = np.linalg.eigh(fisher_t)
        scaled_score = A_diag * ((eigvecs * (eigvals ** -1.0)) @ eigvecs.T @ nabla_t)

        b_t = exp_neg_lambdas * b_t + ws * scaled_score[None, :]

    return preds_all.transpose(1, 0, 2)


def forecast_h(fit_result, M, K):
    beta_bar = fit_result["beta_bar"]
    omega_load = fit_result["omega_load"]

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    T = base_covariates.shape[0]
    n_h = base_covariates.shape[1]

    ws, lambdas = _solve_weights_ff(fit_result["eta"], fit_result["alpha"], K)
    exp_neg_lambdas = np.exp(-lambdas)

    predictions = np.empty((T, n_h))
    P = np.empty(T)

    b_t = fit_result["b_T"].copy()
    for t in range(T):
        beta_t = beta_bar + np.sum(ws * b_t, axis=0)
        b_t = exp_neg_lambdas * b_t
        omega_col = omega_load[bucket_indices[t]]
        Z_t = np.concatenate([base_covariates[t], omega_col[:, None]], axis=-1)
        predictions_t = Z_t @ beta_t
        predictions[t] = predictions_t
        P[t] = predictions_t.sum() / n_h

    return predictions, P


def simulate_panel(params, M, n, rng, K, b_0=None):
    A_diag = np.diag(params["A"])
    sigma2 = params["sigma2"]
    omega_load = params["omega_load"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    p = beta_bar.shape[0]

    M = np.asarray(M, dtype=float)
    T, N_obs, _ = M.shape
    base = M[:, :, :-1]
    bidx = M[:, :, -1].astype(np.int32)
    design = np.concatenate([base, omega_load[bidx][:, :, None]], axis=-1)

    if b_0 is None:
        b_0 = np.zeros((K + 1, p))

    h_inv = 1.0 / sigma2
    sqrt_sigma = np.sqrt(sigma2)
    L_C = np.linalg.cholesky(params["C"])
    ws, lambdas = _solve_weights_ff(params["eta"], params["alpha"], K)
    exp_neg_lambdas = np.exp(-lambdas)
    _eye_p = np.eye(p)

    y_paths = np.empty((n, T, N_obs))
    beta_paths = np.empty((n, T, p))

    for i in range(n):
        g_samp = rng.chisquare(nu, size=T)
        w_samp = rng.standard_normal(size=(T, p))
        z_samp = rng.standard_normal(size=(T, N_obs))

        b_t = b_0.copy()
        for t in range(T):
            design_t = design[t]
            beta_t = beta_bar + np.sum(ws * b_t, axis=0)

            scale = np.sqrt((nu - 2.0) / g_samp[t])
            eps_t = scale * (design_t @ L_C @ w_samp[t] + sqrt_sigma * z_samp[t])
            y_t = design_t @ beta_t + eps_t

            ZtHinvZ = h_inv * (design_t.T @ design_t)
            WLC = ZtHinvZ @ L_C
            Inner_mat = _eye_p + L_C.T @ WLC
            L_Inner = np.linalg.cholesky(Inner_mat)

            V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
            ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
            fisher_t = (nu / (nu - 2.0)) * ((nu + N_obs) / (nu + N_obs + 2.0)) * ZtSigmaInvZ

            ZtHinv_eps = h_inv * (design_t.T @ eps_t)
            woodbury_eps = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
            mahal_Sigma = h_inv * np.dot(eps_t, eps_t) - np.dot(woodbury_eps, woodbury_eps)
            ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ woodbury_eps

            score_weight = (nu + N_obs) / ((nu - 2.0) + mahal_Sigma)
            nabla_t = score_weight * ZtSigmaInv_eps
            eigvals, eigvecs = np.linalg.eigh(fisher_t)
            scaled_score = A_diag * ((eigvecs * (eigvals ** -1.0)) @ eigvecs.T @ nabla_t)

            beta_paths[i, t] = beta_t
            y_paths[i, t] = y_t
            b_t = exp_neg_lambdas * b_t + ws * scaled_score[None, :]

    return y_paths, beta_paths

import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular
from models._solver import adam, adam_adj, sgn, lbfgs

_SOLVERS = {"adam": adam, "adam_adj": adam_adj, "sgn": sgn, "lbfgs": lbfgs}


def _t_unit_var_ppf(alpha, nu):
    a = nu / 2.0
    p = np.where(alpha <= 0.5, alpha, 1.0 - alpha)
    z = jax.scipy.special.ndtri(p)
    x0 = z + (z ** 3 + z) / (4.0 * nu)

    def _newton(x, _):
        x2 = x * x
        u = nu / (nu + x2)
        cdf = 0.5 * jax.scipy.special.betainc(a, 0.5, u)
        log_pdf = (
            gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * nu)
            - ((nu + 1.0) / 2.0) * np.log1p(x2 / nu)
        )
        return x - (cdf - p) / np.exp(log_pdf), None

    x_star, _ = lax.scan(_newton, x0, None, length=15)
    q_std = np.where(alpha <= 0.5, x_star, -x_star)
    return q_std * np.sqrt((nu - 2.0) / nu)

def _solve_weights_ff(eta, alpha, K):
    def _single(ea):
        eta_k = ea[0]
        alpha_k = ea[1]
        indices = np.arange(K + 1)
        lambdas_k = eta_k * np.power(alpha_k, -indices)

        def _rec_step(c_stack, k):
            i_range = np.arange(K + 1)
            mask = i_range < k
            diff = k - i_range
            terms = (
                c_stack
                * np.power(alpha_k, -eta_k * (k - 1))
                * np.exp(eta_k * (1.0 - np.power(alpha_k, diff)))
            )
            c_next = 1.0 - np.sum(np.where(mask, terms, 0.0))
            return c_stack.at[k].set(c_next), None

        c_init = np.zeros(K + 1).at[0].set(1.0)
        c_final, _ = lax.scan(_rec_step, c_init, np.arange(1, K + 1))
        c_ordered = np.flip(c_final)
        w_tilde = c_ordered * np.power(alpha_k, -indices * eta_k)
        w_k = w_tilde / np.sum(w_tilde)
        return w_k, lambdas_k

    ws, lambdas = jax.vmap(_single)(np.stack([eta, alpha], axis=-1))
    return ws.T, lambdas.T

def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, state0):
    A = params["A"]
    sigma2 = params["sigma2"]
    omega_load = params["omega_load"]
    eta = params["eta"]
    alpha = params["alpha"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]

    p = beta_bar.shape[0]
    h_inv = 1.0 / sigma2
    L_C = np.linalg.cholesky(C)

    ws, lambdas = _solve_weights_ff(eta, alpha, K)

    def _step(state, inputs):
        b_t = state
        y_t, base_t, bidx_t, mask_t = inputs

        beta_t = beta_bar + np.sum(ws * b_t, axis=0)

        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)
        eps_t = (y_t - Z_t @ beta_t) * mask_t

        ZtZ = Z_mask.T @ Z_mask
        ZtHinvZ = h_inv * ZtZ
        WLC = ZtHinvZ @ L_C
        Inner_mat = np.eye(p) + L_C.T @ WLC
        L_Inner = np.linalg.cholesky(Inner_mat)
        log_det_Sigma = N_t * np.log(sigma2) + 2.0 * np.sum(np.log(np.diag(L_Inner)))

        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

        ZtHinv_eps = h_inv * (Z_mask.T @ eps_t)
        woodbury_eps = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
        mahal_Sigma = h_inv * (eps_t @ eps_t) - (woodbury_eps @ woodbury_eps)
        ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ woodbury_eps

        ll_t = (
            gammaln((nu + N_t) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
            - 0.5 * log_det_Sigma
            - 0.5 * (nu + N_t) * np.log(1.0 + mahal_Sigma / (nu - 2.0))
        )

        score_weight = (nu + N_t) / ((nu - 2.0) + mahal_Sigma)
        nabla_t = score_weight * ZtSigmaInv_eps

        eigvals, eigvecs = np.linalg.eigh(fisher_t)
        scaled_score = A @ (eigvecs * (eigvals ** -1.0)) @ eigvecs.T @ nabla_t

        b_next = b_t + np.expm1(-lambdas) * b_t + ws * scaled_score[None, :]

        return b_next, (ll_t, beta_t)

    init = state0

    b_T, (lls, betas_prev) = lax.scan(
        _step, init,
        (y_masked, base_covariates, bucket_indices, mask_f),
    )
    beta_T = beta_bar + np.sum(ws * b_T, axis=0)
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, lls, b_T

def fit(
    data: np.ndarray,
    M: np.ndarray,
    initial_guess: dict,
    K: int,
    opt_options: dict | None = None,
    maxiter: int = 5000,
    solver: str = "adam",
):
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
        idx = 0
        beta_bar = theta[idx:idx + p]
        idx += p
        A_diag = theta[idx:idx + p]
        idx += p
        A = np.diag(A_diag)
        sigma2 = np.exp(theta[idx])
        idx += 1
        omega_load = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]])
        idx += n_buckets - 1
        eta = np.exp(theta[idx:idx + p])
        idx += p
        alpha = np.full(p, jax.nn.softplus(theta[idx]) + 1.0)
        idx += 1
        C_diag = np.exp(theta[idx:idx + p])
        idx += p
        C = np.diag(C_diag)
        nu = np.exp(theta[idx]) + 2.0
        return {
            "beta_bar": beta_bar,
            "A": A,
            "sigma2": sigma2,
            "omega_load": omega_load,
            "eta": eta,
            "alpha": alpha,
            "C": C,
            "nu": nu,
        }

    def _invlink(params):
        A_diag = np.diag(params["A"])
        unc_s2 = np.log(params["sigma2"])
        unc_omega_load = params["omega_load"][1:]
        unc_eta = np.log(params["eta"])
        unc_alpha = np.log(np.exp(params["alpha"][0] - 1.0) - 1.0)
        C_diag = np.diag(params["C"])
        unc_C = np.log(C_diag)
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([
            params["beta_bar"],
            A_diag,
            np.array([unc_s2]),
            unc_omega_load,
            unc_eta,
            np.array([unc_alpha]),
            unc_C,
            np.array([unc_nu]),
        ])

    def _criterion(theta):
        params = _link(theta)
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, np.zeros((K + 1, p)))
        return -np.sum(lls)

    theta0 = np.asarray(_invlink(initial_guess))
    theta_opt, niter, final_loss, is_converged = _SOLVERS[solver](_criterion, theta0, opt_options, maxiter)
    params_opt = _link(theta_opt)
    betas, _, b_T = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt, K, np.zeros((K + 1, p)))
    return params_opt | {
        "betas": betas,
        "b_T": b_T,
        "log_likelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
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

    def _link(theta):
        idx = 0
        beta_bar = theta[idx:idx + p]; idx += p
        A_diag = theta[idx:idx + p]; idx += p
        sigma2 = np.exp(theta[idx]); idx += 1
        omega_load = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]]); idx += n_buckets - 1
        eta = np.exp(theta[idx:idx + p]); idx += p
        alpha = np.full(p, jax.nn.softplus(theta[idx]) + 1.0); idx += 1
        C_diag = np.exp(theta[idx:idx + p]); idx += p
        nu = np.exp(theta[idx]) + 2.0
        return np.concatenate([beta_bar, A_diag, np.array([sigma2]),
                                omega_load[1:], eta, np.array([alpha[0]]),
                                C_diag, np.array([nu])])

    def _invlink(params):
        A_diag = np.diag(params["A"])
        unc_s2 = np.log(params["sigma2"])
        unc_omega_load = params["omega_load"][1:]
        unc_eta = np.log(params["eta"])
        unc_alpha = np.log(np.exp(params["alpha"][0] - 1.0) - 1.0)
        unc_C = np.log(np.diag(params["C"]))
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([params["beta_bar"], A_diag, np.array([unc_s2]),
                                unc_omega_load, unc_eta, np.array([unc_alpha]),
                                unc_C, np.array([unc_nu])])

    def _criterion(theta):
        idx = 0
        beta_bar = theta[idx:idx + p]; idx += p
        A = np.diag(theta[idx:idx + p]); idx += p
        sigma2 = np.exp(theta[idx]); idx += 1
        omega_load = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]]); idx += n_buckets - 1
        eta = np.exp(theta[idx:idx + p]); idx += p
        alpha = np.full(p, jax.nn.softplus(theta[idx]) + 1.0); idx += 1
        C = np.diag(np.exp(theta[idx:idx + p])); idx += p
        nu = np.exp(theta[idx]) + 2.0
        params = {"beta_bar": beta_bar, "A": A, "sigma2": sigma2,
                  "omega_load": omega_load, "eta": eta, "alpha": alpha, "C": C, "nu": nu}
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, np.zeros((K + 1, p)))
        return -np.sum(lls)

    theta_opt = _invlink(fit_result)
    H = jax.hessian(_criterion)(theta_opt)
    eigvals, eigvecs = np.linalg.eigh(H)
    H_pd = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    cov_theta = np.linalg.inv(H_pd)
    J = jax.jacobian(_link)(theta_opt)
    cov_natural = J @ cov_theta @ J.T
    se_flat = np.sqrt(np.abs(np.diag(cov_natural)))

    idx = 0
    se = {}
    se["beta_bar"]   = se_flat[idx:idx + p];             idx += p
    se["A_diag"]     = se_flat[idx:idx + p];             idx += p
    se["sigma2"]     = se_flat[idx];                     idx += 1
    se["omega_load"] = se_flat[idx:idx + n_buckets - 1]; idx += n_buckets - 1
    eta_idx = idx
    se["eta"]        = se_flat[idx:idx + p];             idx += p
    se["alpha"]      = se_flat[idx:idx + 1];             idx += 1
    se["C_diag"]     = se_flat[idx:idx + p];             idx += p
    se["nu"]         = se_flat[idx]

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

    betas, log_liks, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, fit_result, K, state0=state0)

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
    eta = fit_result["eta"]
    alpha = fit_result["alpha"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cols = omega_load[bucket_indices]
    M_base = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)

    ws, lambdas = _solve_weights_ff(eta, alpha, K)

    exp_neg_h_lambdas_list = []
    for h in eval_horizons:
        exp_neg_h_lambdas_list.append(np.exp(-(h - 1) * lambdas))
    exp_neg_h_stack = np.stack(exp_neg_h_lambdas_list)

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
    h_inv = 1.0 / fit_result["sigma2"]
    L_C = np.linalg.cholesky(fit_result["C"])

    def _step(b_t, inputs):
        Z_h_t, d_h_t, y_t, base_t, bidx_t, mask_t = inputs

        beta_t = beta_bar + np.sum(ws * b_t, axis=0)
        b_h_all = exp_neg_h_stack * b_t[None, :, :]
        preds_h = np.einsum("hnp,hp->hn", Z_h_t, b_h_all.reshape(b_h_all.shape[0], -1)) + d_h_t

        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)
        eps_t = (y_t - Z_t @ beta_t) * mask_t

        ZtZ = Z_mask.T @ Z_mask
        ZtHinvZ = h_inv * ZtZ
        WLC = ZtHinvZ @ L_C
        Inner_mat = np.eye(p) + L_C.T @ WLC
        L_Inner = np.linalg.cholesky(Inner_mat)

        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (fit_result["nu"] / (fit_result["nu"] - 2.0)) * ((fit_result["nu"] + N_t) / (fit_result["nu"] + N_t + 2.0)) * ZtSigmaInvZ

        ZtHinv_eps = h_inv * (Z_mask.T @ eps_t)
        woodbury_eps = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
        mahal_Sigma = h_inv * (eps_t @ eps_t) - (woodbury_eps @ woodbury_eps)
        ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ woodbury_eps

        score_weight = (fit_result["nu"] + N_t) / ((fit_result["nu"] - 2.0) + mahal_Sigma)
        nabla_t = score_weight * ZtSigmaInv_eps

        eigvals, eigvecs = np.linalg.eigh(fisher_t)
        scaled_score = fit_result["A"] @ (eigvecs * (eigvals ** -1.0)) @ eigvecs.T @ nabla_t

        b_next = b_t + np.expm1(-lambdas) * b_t + ws * scaled_score[None, :]
        return b_next, preds_h

    _, preds_all = lax.scan(
        _step, fit_result["b_T"],
        (Z_h_scan, d_h_scan, y_masked, base_covariates[:T_half], bucket_indices[:T_half], mask_f),
    )
    return preds_all.transpose(1, 0, 2)


def forecast_h(fit_result, M, K):
    beta_bar = fit_result["beta_bar"]
    omega_load = fit_result["omega_load"]
    eta = fit_result["eta"]
    alpha = fit_result["alpha"]

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    n_h = base_covariates.shape[1]

    ws, lambdas = _solve_weights_ff(eta, alpha, K)
    exp_neg_lambdas = np.exp(-lambdas)

    def _step(b_t, inputs):
        base_t, bidx_t = inputs
        beta_t = beta_bar + np.sum(ws * b_t, axis=0)
        b_next = exp_neg_lambdas * b_t
        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        predictions_t = Z_t @ beta_t
        P_t = predictions_t.sum() / n_h
        return b_next, (predictions_t, P_t)

    _, (predictions, P) = lax.scan(_step, fit_result["b_T"], (base_covariates, bucket_indices))
    return predictions, P

def simulate_panel(params, M, n, key, K, b_0=None):
    A = params["A"]
    sigma2 = params["sigma2"]
    omega_load = params["omega_load"]
    eta = params["eta"]
    alpha = params["alpha"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    p = beta_bar.shape[0]

    M = np.asarray(M, dtype=float)
    T, N_obs, _ = M.shape
    base = M[:, :, :-1]
    bidx = M[:, :, -1].astype(np.int32)
    design = np.concatenate([base, omega_load[bidx][:, :, None]], axis=-1)

    if b_0 is None: b_0 = np.zeros((K + 1, p))

    h_inv = 1.0 / sigma2
    sqrt_sigma = np.sqrt(sigma2)
    L_C = np.linalg.cholesky(C)
    ws, lambdas = _solve_weights_ff(eta, alpha, K)
    exp_neg_lambdas = np.exp(-lambdas)

    keys = jax.random.split(key, n)

    def _one_path(k):
        k1, k2, k3 = jax.random.split(k, 3)
        g_samp = jax.random.chisquare(k1, nu, shape=(T,))
        w_samp = jax.random.normal(k2, shape=(T, p))
        z_samp = jax.random.normal(k3, shape=(T, N_obs))

        def step(b_t, inputs):
            g, w, z, design_t = inputs

            beta_t = beta_bar + np.sum(ws * b_t, axis=0)

            scale = np.sqrt((nu - 2.0) / g)
            eps_t = scale * (design_t @ L_C @ w + sqrt_sigma * z)
            y_t = design_t @ beta_t + eps_t

            ZtHinvZ = h_inv * (design_t.T @ design_t)
            WLC = ZtHinvZ @ L_C
            Inner_mat = np.eye(p) + L_C.T @ WLC
            L_Inner = np.linalg.cholesky(Inner_mat)

            V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
            ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
            fisher_t = (nu / (nu - 2.0)) * ((nu + N_obs) / (nu + N_obs + 2.0)) * ZtSigmaInvZ

            ZtHinv_eps = h_inv * (design_t.T @ eps_t)
            woodbury_eps = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
            mahal_Sigma = h_inv * (eps_t @ eps_t) - (woodbury_eps @ woodbury_eps)
            ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ woodbury_eps

            score_weight = (nu + N_obs) / ((nu - 2.0) + mahal_Sigma)
            nabla_t = score_weight * ZtSigmaInv_eps

            eigvals, eigvecs = np.linalg.eigh(fisher_t)
            scaled_score = A @ (eigvecs * (eigvals ** -1.0)) @ eigvecs.T @ nabla_t

            b_next = exp_neg_lambdas * b_t + ws * scaled_score[None, :]
            return b_next, (y_t, beta_t)

        _, (y_path, beta_path) = lax.scan(step, b_0, (g_samp, w_samp, z_samp, design))
        return y_path, beta_path

    return jax.vmap(_one_path)(keys)
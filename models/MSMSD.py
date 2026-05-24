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


def _build_msm_states(K, m0):
    h = 2 ** K
    bits = np.arange(h, dtype=np.int32)
    bit_matrix = (bits[:, None] >> np.arange(K, dtype=np.int32)[None, :]) & 1
    return np.prod(np.where(bit_matrix, 2.0 - m0, m0), axis=1)


def _build_log_transition_tensor(gamma_k):
    I2 = np.eye(2)
    outer = np.ones((2, 1)) * np.array([0.5, 0.5])[None, :]

    def _make_log_pi(gk):
        return np.log((1.0 - gk) * I2 + gk * outer)

    return jax.vmap(_make_log_pi)(gamma_k)

def _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_flat):
    def _k_step(log_pi, k_data):
        log_pi_k, idx0_k, idx1_k, bits_k = k_data
        from0 = log_pi[idx0_k] + log_pi_k[0, bits_k]
        from1 = log_pi[idx1_k] + log_pi_k[1, bits_k]
        return jax.scipy.special.logsumexp(np.stack([from0, from1], axis=0), axis=0), None

    log_pi_out, _ = lax.scan(_k_step, log_pi_flat, (log_P, idx0_all, idx1_all, bits_matrix))
    return log_pi_out

def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, score_power, state0):
    B = params["B"]
    A = params["A"]
    sigma2 = params["sigma2"]
    sigma_0 = params["sigma_0"]
    omega_load = params["omega_load"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    m0 = params["m0"]
    gamma_k = params["gamma_k"]

    p_tilde = beta_bar.shape[0]
    p_full = p_tilde + 1
    h_inv = 1.0 / sigma2
    IminusB = np.eye(p_tilde) - B
    L_C = np.linalg.cholesky(C)

    h_states = 2 ** K
    g_vals = _build_msm_states(K, m0)
    log_P = _build_log_transition_tensor(gamma_k)

    j_range = np.arange(h_states, dtype=np.int32)
    k_range = np.arange(K, dtype=np.int32)
    masks = np.int32(1) << k_range
    bits_matrix = ((j_range[None, :] >> k_range[:, None]) & 1)
    idx0_all = j_range[None, :] & (~masks[:, None])
    idx1_all = j_range[None, :] | masks[:, None]

    def _step(state, inputs):
        log_pi_prev, beta_tilde_t = state
        y_t, base_t, bidx_t, mask_t = inputs

        log_pi_pred = _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_prev)

        sigma_vals = sigma_0 * g_vals

        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)

        ZtHinvZ = h_inv * (Z_mask.T @ Z_mask)
        WLC = ZtHinvZ @ L_C
        Inner_mat = np.eye(p_full) + L_C.T @ WLC
        L_Inner = np.linalg.cholesky(Inner_mat)
        log_det_Sigma = N_t * np.log(sigma2) + 2.0 * np.sum(np.log(np.diag(L_Inner)))
        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

        beta_full_base = np.concatenate([beta_tilde_t, np.zeros(1)])
        eps_base = (y_t - Z_t @ beta_full_base) * mask_t
        omega_col_masked = omega_col * mask_t

        ZtHinv_eps_base = h_inv * (Z_mask.T @ eps_base)
        ZtHinv_omega = h_inv * (Z_mask.T @ omega_col_masked)
        eps_base_sq = h_inv * (eps_base @ eps_base)
        cross = h_inv * (eps_base @ omega_col_masked)
        omega_sq = h_inv * (omega_col_masked @ omega_col_masked)
        wb_base = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps_base, lower=True)
        wb_omega = solve_triangular(L_Inner, L_C.T @ ZtHinv_omega, lower=True)

        def _state_ll_and_score(sigma_j):
            wb_j = wb_base - sigma_j * wb_omega
            mahal_j = (eps_base_sq - 2.0 * sigma_j * cross + sigma_j ** 2 * omega_sq) - (wb_j @ wb_j)
            ll_j = (
                gammaln((nu + N_t) / 2.0) - gammaln(nu / 2.0)
                - 0.5 * (N_t * np.log((nu - 2.0) * np.pi) + log_det_Sigma
                         + (nu + N_t) * np.log1p(mahal_j / (nu - 2.0)))
            )
            ZtSigmaInv_eps_j = (ZtHinv_eps_base - sigma_j * ZtHinv_omega) - V_fisher.T @ wb_j
            nabla_j = ((nu + N_t) / ((nu - 2.0) + mahal_j)) * ZtSigmaInv_eps_j[:p_tilde]
            return ll_j, nabla_j

        ll_all, nabla_all = jax.vmap(_state_ll_and_score)(sigma_vals)

        log_weights = log_pi_pred + ll_all
        log_marg = jax.scipy.special.logsumexp(log_weights)
        log_pi_t = log_weights - log_marg

        pi_t = np.exp(log_pi_t)
        nabla_t = (pi_t[:, None] * nabla_all).sum(axis=0)

        fisher_tilde = fisher_t[:p_tilde, :p_tilde]
        eigvals, eigvecs = np.linalg.eigh(fisher_tilde)
        scaling_matrix = (eigvecs * (eigvals ** (-score_power))) @ eigvecs.T
        xi_tilde = A @ scaling_matrix @ nabla_t

        beta_tilde_next = IminusB @ beta_bar + B @ beta_tilde_t + xi_tilde
        sigma_t = (pi_t * sigma_vals).sum()
        beta_full_t = np.concatenate([beta_tilde_t, np.array([sigma_t])])

        return (log_pi_t, beta_tilde_next), (log_marg, beta_full_t)

    (log_pi_T, beta_tilde_T), (log_liks, betas_prev) = lax.scan(
        _step, state0, (y_masked, base_covariates, bucket_indices, mask_f),
    )

    pi_T = np.exp(log_pi_T)
    sigma_T = sigma_0 * (pi_T * g_vals).sum()
    beta_T = np.concatenate([beta_tilde_T, np.array([sigma_T])])
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, log_liks, (log_pi_T, beta_tilde_T)

def fit(
    data,
    M,
    initial_guess: dict,
    K: int,
    score_power: float,
    opt_options: dict | None = None,
    maxiter: int = 5000,
    solver: str = "adam",
):
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
        idx = 0
        beta_bar = theta[idx:idx + p_tilde]; idx += p_tilde
        B = np.diag(np.tanh(theta[idx:idx + p_tilde])); idx += p_tilde
        A = np.diag(theta[idx:idx + p_tilde]); idx += p_tilde
        sigma2 = np.exp(theta[idx]); idx += 1
        sigma_0 = np.exp(theta[idx]); idx += 1
        omega_load = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]]); idx += n_buckets - 1
        C = np.diag(np.exp(theta[idx:idx + p_full])); idx += p_full
        nu = np.exp(theta[idx]) + 2.0; idx += 1
        m0 = 1.0 + jax.nn.sigmoid(theta[idx]); idx += 1
        gamma_K = jax.nn.sigmoid(theta[idx]); idx += 1
        b = 1.0 + np.exp(theta[idx]); idx += 1
        k_idx = np.arange(1, K + 1, dtype=float)
        gamma_k = 1.0 - (1.0 - gamma_K) ** (b ** (k_idx - K))
        return {
            "beta_bar": beta_bar, "B": B, "A": A,
            "sigma2": sigma2, "sigma_0": sigma_0,
            "omega_load": omega_load, "C": C, "nu": nu,
            "m0": m0, "gamma_K": gamma_K, "b": b, "gamma_k": gamma_k,
        }

    def _invlink(params):
        return np.concatenate([
            params["beta_bar"],
            np.arctanh(np.diag(params["B"])),
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
    theta_opt, niter, final_loss, is_converged = _SOLVERS[solver](_criterion, theta0, opt_options, maxiter)
    params_opt = _link(theta_opt)
    betas, _, (log_pi_T, beta_tilde_T) = _filter(
        y_masked, base_covariates, bucket_indices, mask_f,
        params_opt, K, score_power, (log_pi_init, params_opt["beta_bar"])
    )
    return params_opt | {
        "betas": betas,
        "log_pi_T": log_pi_T,
        "beta_tilde_T": beta_tilde_T,
        "log_likelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
        "score_power": score_power,
        "K": K,
    }

def forecast_h(fit_result, M, K):
    beta_bar = fit_result["beta_bar"]
    B = fit_result["B"]
    IminusB = np.eye(beta_bar.shape[0]) - B
    sigma_0 = fit_result["sigma_0"]
    omega_load = fit_result["omega_load"]
    m0 = fit_result["m0"]
    gamma_k = fit_result["gamma_k"]

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    n_h = base_covariates.shape[1]

    g_vals = _build_msm_states(K, m0)
    log_P = _build_log_transition_tensor(gamma_k)
    h_states = 2 ** K
    j_range = np.arange(h_states, dtype=np.int32)
    k_range = np.arange(K, dtype=np.int32)
    masks = np.int32(1) << k_range
    bits_matrix = ((j_range[None, :] >> k_range[:, None]) & 1)
    idx0_all = j_range[None, :] & (~masks[:, None])
    idx1_all = j_range[None, :] | masks[:, None]

    def _step(state, inputs):
        log_pi_t, beta_tilde_t = state
        base_t, bidx_t = inputs
        log_pi_pred = _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_t)
        pi_pred = np.exp(log_pi_pred)
        sigma_t = sigma_0 * (pi_pred * g_vals).sum()
        beta_full_t = np.concatenate([beta_tilde_t, np.array([sigma_t])])
        beta_tilde_next = IminusB @ beta_bar + B @ beta_tilde_t
        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        predictions_t = Z_t @ beta_full_t
        P_t = predictions_t.sum() / n_h
        return (log_pi_pred, beta_tilde_next), (predictions_t, P_t)

    state0 = (fit_result["log_pi_T"], fit_result["beta_tilde_T"])
    _, (predictions, P) = lax.scan(_step, state0, (base_covariates, bucket_indices))
    return predictions, P

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

def simulate_panel(params, M, n, key, K, m_k_0=None, beta_tilde_0=None):
    B = params["B"]
    A = params["A"]
    sigma2 = params["sigma2"]
    sigma_0 = params["sigma_0"]
    omega_load = params["omega_load"]
    C = params["C"]
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
    IminusB = np.eye(p_tilde) - B
    L_C = np.linalg.cholesky(C)

    keys = jax.random.split(key, n)

    def _one_path(k):
        step_keys = jax.random.split(k, T)

        def step(state, inputs):
            m_k_prev, beta_tilde_t = state
            design_t, key_t = inputs

            k1, k2, k3, k4, k5 = jax.random.split(key_t, 5)

            switch = jax.random.bernoulli(k1, gamma_k)
            new_vals = np.where(jax.random.bernoulli(k2, 0.5, shape=(K,)), 2.0 - m0, m0)
            m_k_t = np.where(switch, new_vals, m_k_prev)

            sigma_t = sigma_0 * np.prod(m_k_t)
            beta_full_t = np.concatenate([beta_tilde_t, np.array([sigma_t])])

            t_scale = np.sqrt(nu / (2.0 * jax.random.gamma(k5, nu / 2.0)))
            eps_t = t_scale * (sqrt_sigma * jax.random.normal(k3, (N_obs,)) + design_t @ (L_C @ jax.random.normal(k4, (p_full,))))
            y_t = design_t @ beta_full_t + eps_t

            ZtHinvZ = h_inv * (design_t.T @ design_t)
            WLC = ZtHinvZ @ L_C
            Inner_mat = np.eye(p_full) + L_C.T @ WLC
            L_Inner = np.linalg.cholesky(Inner_mat)
            V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
            ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
            fisher_tilde = (nu / (nu - 2.0)) * ((nu + N_obs) / (nu + N_obs + 2.0)) * ZtSigmaInvZ[:p_tilde, :p_tilde]

            ZtHinv_eps = h_inv * (design_t.T @ eps_t)
            wb = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
            mahal = h_inv * (eps_t @ eps_t) - (wb @ wb)
            ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ wb
            nabla_t = ((nu + N_obs) / ((nu - 2.0) + mahal)) * ZtSigmaInv_eps[:p_tilde]

            eigvals, eigvecs = np.linalg.eigh(fisher_tilde)
            xi_tilde = A @ ((eigvecs * (eigvals ** (-score_power))) @ eigvecs.T) @ nabla_t
            beta_tilde_next = IminusB @ beta_bar + B @ beta_tilde_t + xi_tilde

            return (m_k_t, beta_tilde_next), (y_t, beta_full_t, m_k_t)

        _, (y_path, beta_path, msm_path) = lax.scan(
            step, (m_k_0, beta_tilde_0), (design, step_keys)
        )
        return y_path, beta_path, msm_path

    return jax.vmap(_one_path)(keys)
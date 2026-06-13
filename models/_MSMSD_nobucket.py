import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular
from models._solver import lbfgs
from models.MSMSD import _build_msm_states, _build_log_transition_tensor, _transition_step_log


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


def _msm_step_nb(y_t, Z_t, mask_t, log_pi_prev, beta_t, params,
                 K, score_power, h_inv, L_C, p, IminusB,
                 log_P, idx0_all, idx1_all, bits_matrix, g_vals,
                 l_per_state, sigma_distinct):
    log_pi_pred = _transition_step_log(log_P, idx0_all, idx1_all, bits_matrix, log_pi_prev)

    Z_mask = Z_t * mask_t[:, None]
    N_t = np.sum(mask_t)
    eps_t = (y_t - Z_t @ beta_t) * mask_t

    ZtHinvZ = h_inv * (Z_mask.T @ Z_mask)
    WLC = ZtHinvZ @ L_C
    Inner_mat = np.eye(p) + L_C.T @ WLC
    L_Inner = np.linalg.cholesky(Inner_mat)
    log_det_Sigma = N_t * np.log(params["sigma2"]) + 2.0 * np.sum(np.log(np.diag(L_Inner)))
    V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
    ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
    nu = params["nu"]
    fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

    ZtHinv_eps = h_inv * (Z_mask.T @ eps_t)
    wb = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
    mahal = h_inv * (eps_t @ eps_t) - (wb @ wb)
    ll_t = (
        gammaln((nu + N_t) / 2.0) - gammaln(nu / 2.0)
        - 0.5 * (N_t * np.log((nu - 2.0) * np.pi) + log_det_Sigma
                 + (nu + N_t) * np.log1p(mahal / (nu - 2.0)))
    )
    nabla_t = ((nu + N_t) / ((nu - 2.0) + mahal)) * (ZtHinv_eps - V_fisher.T @ wb)

    log_weights = log_pi_pred + ll_t
    log_marg = jax.scipy.special.logsumexp(log_weights)
    log_pi_t = log_weights - log_marg

    pi_t = np.exp(log_pi_t)
    sigma_vals = params["sigma_0"] * g_vals
    sigma_t = (pi_t * sigma_vals).sum()

    eigvals, eigvecs = np.linalg.eigh(fisher_t)
    scaling_matrix = (eigvecs * (eigvals ** (-score_power))) @ eigvecs.T
    xi_tilde = params["A"] @ scaling_matrix @ nabla_t
    beta_next = IminusB @ params["beta_bar"] + params["B"] @ beta_t + xi_tilde

    return log_pi_t, beta_next, log_marg, sigma_t


def _filter(y_masked, base_covariates, mask_f, params, K, score_power, state0):
    p = params["beta_bar"].shape[0]
    h_inv = 1.0 / params["sigma2"]
    IminusB = np.eye(p) - params["B"]
    L_C = np.linalg.cholesky(params["C"])

    h_states = 2 ** K
    g_vals = _build_msm_states(K, params["m0"])
    log_P = _build_log_transition_tensor(params["gamma_k"])

    j_range = np.arange(h_states, dtype=np.int32)
    k_range = np.arange(K, dtype=np.int32)
    masks = np.int32(1) << k_range
    bits_matrix = ((j_range[None, :] >> k_range[:, None]) & 1)
    idx0_all = j_range[None, :] & (~masks[:, None])
    idx1_all = j_range[None, :] | masks[:, None]
    l_per_state = bits_matrix.sum(axis=0)
    l_range = np.arange(K + 1, dtype=np.int32)
    sigma_distinct = params["sigma_0"] * (params["m0"] ** (K - l_range)) * ((2.0 - params["m0"]) ** l_range)

    def _step(state, inputs):
        log_pi_prev, beta_t = state
        y_t, Z_t, mask_t = inputs
        log_pi_t, beta_next, log_marg, sigma_t = _msm_step_nb(
            y_t, Z_t, mask_t, log_pi_prev, beta_t, params,
            K, score_power, h_inv, L_C, p, IminusB,
            log_P, idx0_all, idx1_all, bits_matrix, g_vals,
            l_per_state, sigma_distinct,
        )
        return (log_pi_t, beta_next), (log_marg, beta_t, sigma_t)

    (log_pi_T, beta_tilde_T), (log_liks, betas_prev, sigma_prev) = lax.scan(
        _step, state0, (y_masked, base_covariates, mask_f),
    )

    pi_T = np.exp(log_pi_T)
    sigma_T = params["sigma_0"] * (pi_T * g_vals).sum()
    betas = np.concatenate([betas_prev, beta_tilde_T[None]], axis=0)
    sigmas = np.concatenate([sigma_prev, np.array([sigma_T])], axis=0)
    return betas, log_liks, sigmas, (log_pi_T, beta_tilde_T)

def fit(
    data,
    M,
    initial_guess: dict,
    K: int,
    score_power: float,
    opt_options: dict | None = None,
    maxiter: int = 5000,
):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = initial_guess["beta_bar"].shape[0]
    maxiter = int(maxiter)
    opt_options = opt_options or {}
    mask_bool = ~np.isnan(data)
    y_masked = np.where(mask_bool, data, 0.0)
    mask_f = mask_bool.astype(float)
    base_covariates = M[:, :, :-1]
    log_pi_init = np.full(2 ** K, -K * np.log(2.0))

    def _link(theta):
        idx = 0
        beta_bar = theta[idx:idx + p]; idx += p
        B = np.diag(np.tanh(theta[idx:idx + p])); idx += p
        A = np.diag(theta[idx:idx + p]); idx += p
        sigma2 = np.exp(theta[idx]); idx += 1
        sigma_0 = np.exp(theta[idx]); idx += 1
        C = np.diag(np.exp(theta[idx:idx + p])); idx += p
        nu = np.exp(theta[idx]) + 2.0; idx += 1
        m0 = 1.0 + jax.nn.sigmoid(theta[idx]); idx += 1
        gamma_K = jax.nn.sigmoid(theta[idx]); idx += 1
        b = 1.0 + np.exp(theta[idx]); idx += 1
        k_idx = np.arange(1, K + 1, dtype=float)
        gamma_k = 1.0 - (1.0 - gamma_K) ** (b ** (k_idx - K))
        return {
            "beta_bar": beta_bar, "B": B, "A": A,
            "sigma2": sigma2, "sigma_0": sigma_0,
            "C": C, "nu": nu,
            "m0": m0, "gamma_K": gamma_K, "b": b, "gamma_k": gamma_k,
        }

    def _invlink(params):
        return np.concatenate([
            params["beta_bar"],
            np.arctanh(np.diag(params["B"])),
            np.diag(params["A"]),
            np.array([np.log(params["sigma2"]), np.log(params["sigma_0"])]),
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
        _, lls, _, _ = _filter(
            y_masked, base_covariates, mask_f,
            params, K, score_power, (log_pi_init, params["beta_bar"])
        )
        return -np.sum(lls)

    theta0 = np.asarray(_invlink(initial_guess))
    theta_opt, niter, final_loss, is_converged = lbfgs(_criterion, theta0, opt_options, maxiter)
    params_opt = _link(theta_opt)
    betas, _, sigmas, (log_pi_T, beta_tilde_T) = _filter(
        y_masked, base_covariates, mask_f,
        params_opt, K, score_power, (log_pi_init, params_opt["beta_bar"])
    )
    return params_opt | {
        "betas": betas,
        "sigmas": sigmas,
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

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    n_h = base_covariates.shape[1]

    def _step(beta_t, Z_t):
        beta_next = IminusB @ beta_bar + B @ beta_t
        predictions_t = Z_t @ beta_t
        P_t = predictions_t.sum() / n_h
        return beta_next, (predictions_t, P_t)

    _, (predictions, P) = lax.scan(_step, fit_result["beta_tilde_T"], base_covariates)
    return predictions, P

def forecast_rolling_h(fit_result, M, y_test, K, score_power, eval_horizons):
    B = fit_result["B"]
    beta_bar = fit_result["beta_bar"]
    m0 = fit_result["m0"]
    gamma_k = fit_result["gamma_k"]
    IminusB = np.eye(B.shape[0]) - B

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    p = beta_bar.shape[0]
    base_covariates = M[:, :, :-1]
    Z_all = base_covariates

    g_vals = _build_msm_states(K, m0)
    log_P = _build_log_transition_tensor(gamma_k)
    h_states = 2 ** K
    j_range = np.arange(h_states, dtype=np.int32)
    k_range = np.arange(K, dtype=np.int32)
    masks = np.int32(1) << k_range
    bits_matrix = ((j_range[None, :] >> k_range[:, None]) & 1)
    idx0_all = j_range[None, :] & (~masks[:, None])
    idx1_all = j_range[None, :] | masks[:, None]
    l_per_state = bits_matrix.sum(axis=0)
    l_range = np.arange(K + 1, dtype=np.int32)
    sigma_distinct = fit_result["sigma_0"] * (m0 ** (K - l_range)) * ((2.0 - m0) ** l_range)

    max_h = max(eval_horizons)
    def _B_power(Bk, _):
        return Bk @ B, Bk @ B
    _, B_powers = lax.scan(_B_power, np.eye(p), None, length=max_h)
    h_indices = np.array(eval_horizons, dtype=np.int32) - 1
    B_h_stack = B_powers[h_indices]

    h_shifts = np.array(eval_horizons, dtype=np.int32) - 1
    t_idx = np.arange(T_half)
    Z_h_stack = Z_all[h_shifts[:, None] + t_idx[None, :]]
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)

    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    h_inv = 1.0 / fit_result["sigma2"]
    L_C = np.linalg.cholesky(fit_result["C"])

    def _step(state, inputs):
        log_pi_t, beta_t = state
        Z_h_t, y_t, Z_t, mask_t = inputs

        dev = beta_t - beta_bar
        beta_h_all = beta_bar + np.einsum("hpq,q->hp", B_h_stack, dev)
        preds_h = np.einsum("hnp,hp->hn", Z_h_t, beta_h_all)

        log_pi_next, beta_next, _, _ = _msm_step_nb(
            y_t, Z_t, mask_t, log_pi_t, beta_t, fit_result,
            K, score_power, h_inv, L_C, p, IminusB,
            log_P, idx0_all, idx1_all, bits_matrix, g_vals,
            l_per_state, sigma_distinct,
        )

        return (log_pi_next, beta_next), preds_h

    state0 = (fit_result["log_pi_T"], fit_result["beta_tilde_T"])
    _, preds_all = lax.scan(
        _step, state0,
        (Z_h_scan, y_masked, base_covariates[:T_half], mask_f),
    )
    return preds_all.transpose(1, 0, 2)


def forecast(fit_result, M, y_test, K, score_power, alpha):
    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = M.shape[0]
    base_covariates = M[:, :, :-1]
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, sigmas, _ = _filter(
        y_masked, base_covariates, mask_f,
        fit_result, K, score_power,
        state0=(fit_result["log_pi_T"], fit_result["beta_tilde_T"])
    )

    Z = base_covariates
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
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    m0 = params["m0"]
    gamma_k = params["gamma_k"]
    score_power = params["score_power"]
    p = beta_bar.shape[0]

    M = np.asarray(M, dtype=float)
    T, N_obs, _ = M.shape
    design = M[:, :, :-1]

    if m_k_0 is None: m_k_0 = np.full(K, m0)
    if beta_tilde_0 is None: beta_tilde_0 = beta_bar

    h_inv = 1.0 / sigma2
    sqrt_sigma = np.sqrt(sigma2)
    IminusB = np.eye(p) - B
    L_C = np.linalg.cholesky(C)

    keys = jax.random.split(key, n)

    def _one_path(k):
        step_keys = jax.random.split(k, T)

        def step(state, inputs):
            m_k_prev, beta_t = state
            design_t, key_t = inputs

            k1, k2, k3, k4, k5 = jax.random.split(key_t, 5)

            switch = jax.random.bernoulli(k1, gamma_k)
            new_vals = np.where(jax.random.bernoulli(k2, 0.5, shape=(K,)), 2.0 - m0, m0)
            m_k_t = np.where(switch, new_vals, m_k_prev)

            t_scale = np.sqrt((nu - 2.0) / (2.0 * jax.random.gamma(k5, nu / 2.0)))
            eps_t = t_scale * (sqrt_sigma * jax.random.normal(k3, (N_obs,)) + design_t @ (L_C @ jax.random.normal(k4, (p,))))
            y_t = design_t @ beta_t + eps_t

            ZtHinvZ = h_inv * (design_t.T @ design_t)
            WLC = ZtHinvZ @ L_C
            Inner_mat = np.eye(p) + L_C.T @ WLC
            L_Inner = np.linalg.cholesky(Inner_mat)
            V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
            ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
            fisher_tilde = (nu / (nu - 2.0)) * ((nu + N_obs) / (nu + N_obs + 2.0)) * ZtSigmaInvZ

            ZtHinv_eps = h_inv * (design_t.T @ eps_t)
            wb = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
            mahal = h_inv * (eps_t @ eps_t) - (wb @ wb)
            ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ wb
            nabla_t = ((nu + N_obs) / ((nu - 2.0) + mahal)) * ZtSigmaInv_eps

            eigvals, eigvecs = np.linalg.eigh(fisher_tilde)
            xi_tilde = A @ ((eigvecs * (eigvals ** (-score_power))) @ eigvecs.T) @ nabla_t
            beta_next = IminusB @ beta_bar + B @ beta_t + xi_tilde

            return (m_k_t, beta_next), (y_t, beta_t, m_k_t)

        _, (y_path, beta_path, msm_path) = lax.scan(
            step, (m_k_0, beta_tilde_0), (design, step_keys)
        )
        return y_path, beta_path, msm_path

    return jax.vmap(_one_path)(keys)

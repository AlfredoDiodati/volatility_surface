import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular

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

def _solve_weights_midas(a, b, eta, alpha_scalar, K):
    def _single(params):
        a_k, b_k, eta_k = params
        indices = np.arange(K + 1)
        x = indices / K
        eps = 1e-10
        x = np.clip(x, eps, 1.0 - eps)
        log_w = (a_k - 1.0) * np.log(x) + (b_k - 1.0) * np.log(1.0 - x)
        w_k = np.exp(log_w - jax.scipy.special.logsumexp(log_w))
        lambdas_k = eta_k * np.power(alpha_scalar, -indices)
        return w_k, lambdas_k

    ws, lambdas = jax.vmap(_single)(np.stack([a, b, eta], axis=-1))
    return ws.T, lambdas.T

def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, score_power, state0=None):
    A = params["A"]
    sigma2 = params["sigma2"]
    omega_load = params["omega_load"]
    eta = params["eta"]
    alpha = params["alpha"]
    a_midas = params["a_midas"]
    b_midas = params["b_midas"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]

    p = beta_bar.shape[0]
    K1 = K + 1
    h_inv = 1.0 / np.maximum(sigma2, 1e-10)
    L_C = np.linalg.cholesky(C + np.eye(p) * 1e-10)

    ws, lambdas = _solve_weights_midas(a_midas, b_midas, eta, alpha[0], K)

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
        Inner_mat = np.eye(p) + L_C.T @ WLC + np.eye(p) * 1e-10
        L_Inner = np.linalg.cholesky(Inner_mat)
        
        log_det_Sigma = N_t * np.log(sigma2) + 2.0 * np.sum(np.log(np.diag(L_Inner)))

        V_fisher = solve_triangular(L_Inner, WLC.T, lower=True)
        ZtSigmaInvZ = ZtHinvZ - V_fisher.T @ V_fisher
        fisher_t = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * ZtSigmaInvZ

        ZtHinv_eps = h_inv * (Z_mask.T @ eps_t)
        woodbury_eps = solve_triangular(L_Inner, L_C.T @ ZtHinv_eps, lower=True)
        mahal_Sigma = np.maximum(h_inv * (eps_t @ eps_t) - (woodbury_eps @ woodbury_eps), 0.0)
        ZtSigmaInv_eps = ZtHinv_eps - V_fisher.T @ woodbury_eps

        ll_t = (
            gammaln((nu + N_t) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
            - 0.5 * log_det_Sigma
            - 0.5 * (nu + N_t) * np.log(1.0 + mahal_Sigma / (nu - 2.0) + 1e-10)
        )

        score_weight = (nu + N_t) / ((nu - 2.0) + mahal_Sigma + 1e-10)
        nabla_t = score_weight * ZtSigmaInv_eps

        eigvals, eigvecs = np.linalg.eigh(fisher_t + np.eye(p) * 1e-10)
        scaling_matrix = (eigvecs * (np.maximum(eigvals, 1e-10) ** (-score_power))) @ eigvecs.T
        scaled_score = np.clip(scaling_matrix @ nabla_t, -1e3, 1e3)

        xi_t = A @ scaled_score
        b_next = b_t + np.expm1(-lambdas) * b_t + xi_t[None, :]

        return b_next, (ll_t, beta_t)

    init = np.zeros((K1, p)) if state0 is None else state0

    b_T, (lls, betas_prev) = lax.scan(
        _step, init,
        (y_masked, base_covariates, bucket_indices, mask_f),
    )
    beta_T = beta_bar + np.sum(ws * b_T, axis=0)
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, lls, b_T

def fit(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    K: int,
    score_power: float,
    opt_options: dict | None = None,
    maxiter: int = 5000,
):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega_load"].shape[0]
    maxiter = int(maxiter)
    opt_options = opt_options or {}
    lr = opt_options.get("learning_rate", 1e-3)
    tol = opt_options.get("tol", 1e-6)
    b1 = opt_options.get("beta1", 0.9)
    b2 = opt_options.get("beta2", 0.999)
    eps = opt_options.get("eps", 1e-8)
    mask_bool = ~np.isnan(data)
    y_masked = np.where(mask_bool, data, 0.0)
    mask_f = mask_bool.astype(float)
    base_covariates = covariates[:, :, :-1]
    bucket_indices = covariates[:, :, -1].astype(np.int32)

    def _link(theta):
        idx = 0
        beta_bar = theta[idx:idx + p]
        idx += p
        A_diag = jax.nn.softplus(theta[idx:idx + p])
        idx += p
        A = np.diag(A_diag)
        sigma2 = jax.nn.softplus(theta[idx]) + 1e-6
        idx += 1
        omega_load = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]])
        idx += n_buckets - 1
        eta = jax.nn.softplus(theta[idx:idx + p]) + 1e-6
        idx += p
        alpha_val = jax.nn.softplus(theta[idx]) + 1.0001
        alpha = np.full(p, alpha_val)
        idx += 1
        a_midas = jax.nn.softplus(theta[idx:idx + p]) + 1.0001
        idx += p
        b_midas = jax.nn.softplus(theta[idx:idx + p]) + 1.0001
        idx += p
        C_diag = jax.nn.softplus(theta[idx:idx + p]) + 1e-10
        idx += p
        C = np.diag(C_diag)
        nu = jax.nn.softplus(theta[idx]) + 2.0001
        return {
            "beta_bar": beta_bar, "A": A, "sigma2": sigma2, "omega_load": omega_load,
            "eta": eta, "alpha": alpha, "a_midas": a_midas, "b_midas": b_midas, "C": C, "nu": nu,
        }

    def _invlink(params):
        def inv_sp(x): 
            return np.log(np.exp(np.maximum(x, 1e-10)) - 1.0)

        A_diag = np.diag(params["A"])
        return np.concatenate([
            params["beta_bar"],
            inv_sp(A_diag),
            np.array([inv_sp(params["sigma2"] - 1e-6)]),
            params["omega_load"][1:],
            inv_sp(params["eta"] - 1e-6),
            np.array([inv_sp(params["alpha"][0] - 1.0001)]),
            inv_sp(params["a_midas"] - 1.0001),
            inv_sp(params["b_midas"] - 1.0001),
            inv_sp(np.diag(params["C"]) - 1e-10),
            np.array([inv_sp(params["nu"] - 2.0001)]),
        ])
        
    def _criterion(theta):
        params = _link(theta)
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, score_power)
        return -np.sum(lls)

    value_and_grad = jax.value_and_grad(_criterion)

    def _adam_step(state):
        theta, m, v, i, prev_loss, converged = state
        loss, g = value_and_grad(theta)
        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * g * g
        i1 = i + 1
        mhat = m_new / (1.0 - b1 ** i1)
        vhat = v_new / (1.0 - b2 ** i1)
        theta_new = theta - lr * mhat / (np.sqrt(vhat) + eps)
        converged_new = np.abs(loss - prev_loss) / (np.abs(prev_loss) + 1.0) < tol
        return (theta_new, m_new, v_new, i1, loss, converged_new)

    def _not_converged(state):
        _, _, _, i, loss, converged = state
        return (i < maxiter) & ~converged & (np.isfinite(loss) | (i == 0))

    theta0 = np.asarray(_invlink(initial_guess))
    state0 = (theta0, np.zeros_like(theta0), np.zeros_like(theta0), np.asarray(0, dtype=np.int32), np.asarray(np.inf), np.asarray(False))
    theta_opt, _, _, niter, final_loss, is_converged = lax.while_loop(_not_converged, _adam_step, state0)
    params_opt = _link(theta_opt)
    betas, _, b_T = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt, K, score_power)
    return params_opt | {"betas": betas, "b_T": b_T, "log_likelihood": -final_loss, "niter": niter, "is_converged": is_converged, "score_power": score_power}

def forecast(fit_result, covariates, y_test, K, score_power, target_alpha):
    state0 = fit_result["b_T"]
    covariates = np.asarray(covariates, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = covariates.shape[0]
    base_covariates = covariates[:, :, :-1]
    bucket_indices = covariates[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, fit_result, K, score_power, state0=state0)

    omega_cols = fit_result["omega_load"][bucket_indices]
    Z = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    predictions = np.einsum("hni,hi->hn", Z, betas[:H])

    n_obs_h = mask_f.sum(axis=1)
    P = predictions.sum(axis=1) / np.where(n_obs_h > 0, n_obs_h, 1.0)
    z_sum = Z.sum(axis=1)
    F_sum = n_obs_h * fit_result["sigma2"] + np.einsum("hi,ij,hj->h", z_sum, fit_result["C"], z_sum)
    q = _t_unit_var_ppf(target_alpha, fit_result["nu"])
    VaR = P + q / np.where(n_obs_h > 0, n_obs_h, 1.0) * np.sqrt(F_sum)

    return predictions, P, VaR, log_liks
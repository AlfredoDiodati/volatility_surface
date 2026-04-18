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

def _solve_weights(eta, alpha, K):
    indices = np.arange(K + 1)

    lambdas = eta * np.power(alpha, -indices)
    rhos = np.exp(-lambdas)

    def _rec_step(c_stack, k):
        i_range = np.arange(K + 1)
        mask = i_range < k

        diff = k - i_range
        terms = c_stack * np.power(alpha, -eta * diff) * np.exp(eta * (1.0 - np.power(alpha, -diff)))

        c_next = 1.0 - np.sum(np.where(mask, terms, 0.0))
        return c_stack.at[k].set(c_next), None

    c_init = np.zeros(K + 1)
    c_init = c_init.at[0].set(1.0)

    c_final, _ = lax.scan(
        _rec_step,
        c_init,
        np.arange(1, K + 1)
    )

    c_ordered = np.flip(c_final)
    w_tilde = c_ordered * np.power(alpha, -indices * eta)
    w = w_tilde / np.sum(w_tilde)

    return w, rhos


def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, score_power, state0=None):
    B = params["B"]
    A = params["A"]
    sigma2 = params["sigma2"]
    sigma_0 = params["sigma_0"]
    omega_load = params["omega_load"]
    eta = params["eta"]
    alpha = params["alpha"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]

    p_tilde = beta_bar.shape[0]
    p_full = p_tilde + 1
    K1 = K + 1
    h_inv = 1.0 / sigma2
    IminusB = np.eye(p_tilde) - B
    L_C = np.linalg.cholesky(C)

    w_tilde_norm, rho = _solve_weights(eta, alpha, K)

    def _step(state, inputs):
        b_t, beta_tilde_t = state
        y_t, base_t, bidx_t, mask_t = inputs

        sigma_t = sigma_0 + np.sum(b_t)
        beta_full_t = np.concatenate([beta_tilde_t, np.array([sigma_t])])

        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)
        eps_t = (y_t - Z_t @ beta_full_t) * mask_t

        ZtZ = Z_mask.T @ Z_mask
        ZtHinvZ = h_inv * ZtZ
        WLC = ZtHinvZ @ L_C
        Inner_mat = np.eye(p_full) + L_C.T @ WLC
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
        scaling_matrix = (eigvecs * (eigvals ** (-score_power))) @ eigvecs.T
        scaled_score = scaling_matrix @ nabla_t

        xi_sigma = scaled_score[-1]
        xi_tilde = A @ scaled_score[:-1]

        b_next = rho * b_t + w_tilde_norm * xi_sigma
        beta_tilde_next = IminusB @ beta_bar + B @ beta_tilde_t + xi_tilde

        return (b_next, beta_tilde_next), (ll_t, beta_full_t)

    init = (np.zeros(K1), beta_bar) if state0 is None else state0

    (b_T, beta_tilde_T), (lls, betas_prev) = lax.scan(
        _step, init,
        (y_masked, base_covariates, bucket_indices, mask_f),
    )
    sigma_T = sigma_0 + np.sum(b_T)
    beta_T = np.concatenate([beta_tilde_T, np.array([sigma_T])])
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, lls, (b_T, beta_tilde_T)


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
    p_tilde = initial_guess["beta_bar"].shape[0]
    p_full = p_tilde + 1
    n_buckets = initial_guess["omega_load"].shape[0]
    maxiter = int(maxiter)
    opt_options = opt_options or {}
    lr = opt_options.get("learning_rate", 1e-2)
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
        beta_bar = theta[idx:idx + p_tilde]
        idx += p_tilde
        B_diag = np.tanh(theta[idx:idx + p_tilde])
        idx += p_tilde
        B = np.diag(B_diag)
        A_diag = theta[idx:idx + p_tilde]
        idx += p_tilde
        A = np.diag(A_diag)
        sigma2 = np.exp(theta[idx])
        idx += 1
        sigma_0 = theta[idx]
        idx += 1
        omega_load = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]])
        idx += n_buckets - 1
        eta = np.exp(theta[idx])
        idx += 1
        alpha = 1.0 + jax.nn.softplus(theta[idx])
        idx += 1
        C_diag = np.exp(theta[idx:idx + p_full])
        idx += p_full
        C = np.diag(C_diag)
        nu = np.exp(theta[idx]) + 2.0
        return {
            "beta_bar": beta_bar,
            "B": B,
            "A": A,
            "sigma2": sigma2,
            "sigma_0": sigma_0,
            "omega_load": omega_load,
            "eta": eta,
            "alpha": alpha,
            "C": C,
            "nu": nu,
        }

    def _invlink(params):
        B_diag = np.diag(params["B"])
        unc_B = np.arctanh(B_diag)
        A_diag = np.diag(params["A"])
        unc_A = A_diag
        unc_s2 = np.log(params["sigma2"])
        unc_omega_load = params["omega_load"][1:]
        unc_eta = np.log(params["eta"])
        unc_alpha = np.log(np.exp(params["alpha"] - 1.0) - 1.0)
        C_diag = np.diag(params["C"])
        unc_C = np.log(C_diag)
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([
            params["beta_bar"],
            unc_B,
            unc_A,
            np.array([unc_s2]),
            np.array([params["sigma_0"]]),
            unc_omega_load,
            np.array([unc_eta]),
            np.array([unc_alpha]),
            unc_C,
            np.array([unc_nu]),
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
    state0 = (
        theta0,
        np.zeros_like(theta0),
        np.zeros_like(theta0),
        np.asarray(0, dtype=np.int32),
        np.asarray(np.inf),
        np.asarray(False),
    )
    theta_opt, _, _, niter, final_loss, is_converged = lax.while_loop(
        _not_converged, _adam_step, state0
    )
    params_opt = _link(theta_opt)
    betas, _, (b_T, beta_tilde_T) = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt, K, score_power)
    return params_opt | {
        "betas": betas,
        "b_T": b_T,
        "beta_tilde_T": beta_tilde_T,
        "log_likelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
        "score_power": score_power,
    }


def forecast(fit_result, covariates, y_test, K, score_power, alpha):
    state0 = (fit_result["b_T"], fit_result["beta_tilde_T"])

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

    n_obs_h = mask_bool.sum(axis=1)
    P = predictions.sum(axis=1) / n_obs_h
    z_sum = Z.sum(axis=1)
    F_sum = n_obs_h * fit_result["sigma2"] + np.einsum("hi,ij,hj->h", z_sum, fit_result["C"], z_sum)
    q = _t_unit_var_ppf(alpha, fit_result["nu"])
    VaR = P + q / n_obs_h * np.sqrt(F_sum)

    return predictions, P, VaR, log_liks
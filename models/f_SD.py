import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular

import jax.numpy as np
from jax import lax

def _solve_weights(eta, beta, K):
    indices = np.arange(K + 1)
    
    lambdas = eta * np.power(beta, -indices)
    rhos = np.exp(-lambdas)

    def _rec_step(c_stack, k):
        i_range = np.arange(K + 1)
        mask = i_range < k
        
        diff = k - i_range
        terms = c_stack * np.power(beta, -eta * diff) * np.exp(eta * (1.0 - np.power(beta, -diff)))
        
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
    w_tilde = c_ordered * np.power(beta, -indices * eta)
    w = w_tilde / np.sum(w_tilde)
    
    return w, rhos

def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, state0=None):
    B = params["B"]
    A = params["A"]
    sigma2 = params["sigma2"]
    sigma_0 = params["sigma_0"]
    omega_load = params["omega_load"]
    eta = params["eta"]
    rho_K = params["rho_K"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]

    p_tilde = beta_bar.shape[0]
    p_full = p_tilde + 1
    K1 = K + 1
    h_inv = 1.0 / sigma2
    IminusB = np.eye(p_tilde) - B
    C_reg = C + 1e-8 * np.eye(p_full)
    L_C = np.linalg.cholesky(C_reg)

    w_tilde_norm, rho = _solve_weights(eta, rho_K, K)

    def _step(state, inputs):
        b_t, beta_tilde_t = state
        y_t, base_t, bidx_t, mask_t = inputs

        sigma_t = sigma_0 + np.sum(b_t)
        beta_full_t = np.concatenate([beta_tilde_t, np.array([sigma_t])])

        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        eps_t = (y_t - Z_t @ beta_full_t) * mask_t
        N_t = np.sum(mask_t)

        ZtZ = Z_mask.T @ Z_mask
        Zte = Z_mask.T @ eps_t
        ZtHinvZ = h_inv * ZtZ
        gls_step = np.linalg.solve(ZtHinvZ + 1e-8 * np.eye(p_full), h_inv * Zte)
        mahal_H = h_inv * (eps_t @ eps_t)
        weight = (1.0 + (N_t + 2.0) / nu) / (1.0 + mahal_H / (nu - 2.0))

        xi_sigma = weight * gls_step[-1]
        xi_tilde = A @ (weight * gls_step[:-1])

        M = np.eye(p_full) + h_inv * (L_C.T @ ZtZ @ L_C)
        L_M = np.linalg.cholesky(M)
        u = solve_triangular(L_M, h_inv * (L_C.T @ Zte), lower=True)
        log_det_F = N_t * np.log(sigma2) + 2.0 * np.sum(np.log(np.diag(L_M)))
        quad = h_inv * (eps_t @ eps_t) - (u @ u)
        ll_t = (
            gammaln((nu + N_t) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
            - 0.5 * log_det_F
            - 0.5 * (nu + N_t) * np.log(1.0 + quad / (nu - 2.0))
        )
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
        rho_K = jax.nn.sigmoid(theta[idx])
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
            "rho_K": rho_K,
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
        unc_rho_K = np.log(params["rho_K"] / (1.0 - params["rho_K"]))
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
            np.array([unc_rho_K]),
            unc_C,
            np.array([unc_nu]),
        ])

    def _criterion(theta):
        params = _link(theta)
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K)
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
    betas, _, (b_T, beta_tilde_T) = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt, K)
    return params_opt | {
        "betas": betas,
        "b_T": b_T,
        "beta_tilde_T": beta_tilde_T,
        "log_likelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
    }
    
fit = jax.jit(fit, static_argnames=("maxiter", "K"))

def forecast(fit_result, covariates, K, q_alpha):
    state0 = (fit_result["b_T"], fit_result["beta_tilde_T"])

    covariates = np.asarray(covariates, dtype=float)
    H = covariates.shape[0]
    n_obs = covariates.shape[1]
    base_covariates = covariates[:, :, :-1]
    bucket_indices = covariates[:, :, -1].astype(np.int32)

    y_zeros = np.zeros((H, n_obs))
    mask_zeros = np.zeros((H, n_obs))

    betas, _, _ = _filter(y_zeros, base_covariates, bucket_indices, mask_zeros, fit_result, K, state0=state0)

    omega_cols = fit_result["omega_load"][bucket_indices]
    Z = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    predictions = np.einsum("hni,hi->hn", Z, betas[:H])

    P = predictions.mean(axis=1)

    z_sum = Z.sum(axis=1)
    F_sum = n_obs * fit_result["sigma2"] + np.einsum("hi,ij,hj->h", z_sum, fit_result["C"], z_sum)
    VaR = P + q_alpha / n_obs * np.sqrt(F_sum)

    return predictions, betas, P, VaR

forecast = jax.jit(forecast, static_argnames=("K",))
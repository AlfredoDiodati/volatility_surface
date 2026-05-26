import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln
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

def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, state0):
    B = params["B"]
    A = params["A"]
    sigma2 = params["sigma2"]
    omega = params["omega"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    p = beta_bar.shape[0]
    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C

    def _step(beta_t, inputs):
        y_t, base_t, bidx_t, mask_t = inputs
        omega_col = omega[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        eps_t = (y_t - Z_t @ beta_t) * mask_t
        N_t = np.sum(mask_t)
        
        V_t = h_inv * (Z_mask.T @ Z_mask)
        G_t = h_inv * (Z_mask.T @ eps_t)
        S = np.diag(C_inv) + V_t
        
        mahal_H = h_inv * np.sum(eps_t**2)
        S_inv = np.linalg.solve(S, np.concatenate([G_t[:, None], V_t], axis=1))
        S_inv_G = S_inv[:, 0]
        S_inv_V = S_inv[:, 1:]
        mahal_F = mahal_H - G_t @ S_inv_G

        weight = (1.0 + (N_t + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
        g_tilde = G_t - V_t @ S_inv_G
        V_tilde = V_t - V_t @ S_inv_V
        xi = A @ (weight * np.linalg.solve(V_tilde, g_tilde))
        
        beta_next = beta_bar + B @ (beta_t - beta_bar) + xi
        
        log_det_F = N_t * np.log(sigma2) + np.sum(np.log(C)) + np.linalg.slogdet(S)[1]
        log_constants = (gammaln((nu + N_t) / 2.0) - gammaln(nu / 2.0) - 
                         0.5 * N_t * np.log((nu - 2.0) * np.pi))
        log_lik = log_constants - 0.5 * log_det_F - 0.5 * (nu + N_t) * np.log(1.0 + mahal_F / (nu - 2.0))
        
        return beta_next, (beta_t, log_lik, beta_next)

    _, (betas, log_liks, beta_Ts) = lax.scan(
        _step, state0, (y_masked, base_covariates, bucket_indices, mask_f)
    )
    return betas, log_liks, beta_Ts[-1]

def fit(
    data: np.ndarray,
    M: np.ndarray,
    initial_guess: dict,
    opt_options: dict | None = None,
    maxiter: int = 5000,
    solver: str = "adam",
):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
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
        B_diag = np.tanh(theta[idx:idx + p])
        B = np.diag(B_diag)
        idx += p
        A = np.diag(theta[idx:idx + p])
        idx += p
        sigma2 = np.exp(theta[idx])
        idx += 1
        omega = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]])
        idx += n_buckets - 1
        C_diag = np.exp(theta[idx:idx + p])
        idx += p
        nu = np.exp(theta[idx]) + 2.0
        return {"beta_bar": beta_bar, "B": B, "A": A, "sigma2": sigma2,
                "omega": omega, "C": C_diag, "nu": nu}

    def _invlink(params):
        B_diag = np.diag(params["B"])
        unc_B = np.arctanh(B_diag)
        unc_A = np.diag(params["A"])
        unc_s2 = np.log(params["sigma2"])
        unc_omega = params["omega"][1:]
        unc_C = np.log(params["C"])
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([
            params["beta_bar"], unc_B, unc_A,
            np.array([unc_s2]), unc_omega,
            unc_C, np.array([unc_nu]),
        ])

    def _criterion(theta):
        params = _link(theta)
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, params["beta_bar"])
        return -np.sum(lls)

    theta0 = np.asarray(_invlink(initial_guess))
    theta_opt, niter, final_loss, is_converged = _SOLVERS[solver](_criterion, theta0, opt_options, maxiter)
    params_opt = _link(theta_opt)
    betas, _, beta_T = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt, params_opt["beta_bar"])
    return params_opt | {
        "betas": betas,
        "beta_T": beta_T,
        "log_likelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
    }


def forecast(fit_result, M, y_test, alpha):
    state0 = fit_result["beta_T"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = M.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, fit_result, state0)

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

def forecast_rolling_h(fit_result, M, y_test, eval_horizons):
    B = fit_result["B"]
    beta_bar = fit_result["beta_bar"]
    omega = fit_result["omega"]
    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas_origins, _, _ = _filter(
        y_masked, base_covariates[:T_half], bucket_indices[:T_half],
        mask_f, fit_result, fit_result["beta_T"],
    )

    B_h_list = []
    for h in eval_horizons:
        Bh = np.eye(B.shape[0], dtype=float)
        for _ in range(h - 1):
            Bh = Bh @ B
        B_h_list.append(Bh)
    B_h_stack = np.stack(B_h_list)
    deviations = betas_origins - beta_bar
    beta_h = beta_bar + np.einsum("hpq,tq->htp", B_h_stack, deviations)

    omega_cols = omega[bucket_indices]
    Z_all = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    return np.einsum("htnp,htp->htn", Z_h_stack, beta_h)


def forecast_h(fit_result, M):
    B = fit_result["B"]
    beta_bar = fit_result["beta_bar"]
    omega = fit_result["omega"]

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    n_h = base_covariates.shape[1]

    def _step(beta_h, inputs):
        base_t, bidx_t = inputs
        beta_next = beta_bar + B @ (beta_h - beta_bar)
        omega_col = omega[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        predictions_t = Z_t @ beta_next
        P_t = predictions_t.sum() / n_h
        return beta_next, (predictions_t, P_t)

    _, (predictions, P) = lax.scan(_step, fit_result["beta_T"], (base_covariates, bucket_indices))
    return predictions, P

def simulate_panel(params, M, n, key, beta_0=None):
    B = params["B"]
    A = params["A"]
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

    if beta_0 is None: beta_0 = beta_bar

    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C
    sqrt_C = np.sqrt(C)
    sqrt_sigma = np.sqrt(sigma2)

    keys = jax.random.split(key, n)

    def _one_path(k):
        k1, k2, k3 = jax.random.split(k, 3)
        g_samp = jax.random.chisquare(k1, nu, shape=(T,))
        w_samp = jax.random.normal(k2, shape=(T, p))
        z_samp = jax.random.normal(k3, shape=(T, N_obs))

        def step(beta_t, inputs):
            g, w, z, design_t = inputs

            V_t = h_inv * (design_t.T @ design_t)
            S_t = np.diag(C_inv) + V_t

            scale = np.sqrt((nu - 2.0) / g)
            eps_t = scale * (design_t @ (sqrt_C * w) + sqrt_sigma * z)
            y_t = design_t @ beta_t + eps_t

            G_t = h_inv * (design_t.T @ eps_t)
            S_inv = np.linalg.solve(S_t, np.concatenate([G_t[:, None], V_t], axis=1))
            S_inv_G = S_inv[:, 0]
            S_inv_V_t = S_inv[:, 1:]
            V_tilde_t = V_t - V_t @ S_inv_V_t
            mahal_H = h_inv * np.sum(eps_t ** 2)
            mahal_F = mahal_H - G_t @ S_inv_G
            wt = (1.0 + (N_obs + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
            g_tilde = G_t - V_t @ S_inv_G
            s_t = wt * np.linalg.solve(V_tilde_t, g_tilde)

            beta_next = beta_bar + B @ (beta_t - beta_bar) + A @ s_t
            return beta_next, (y_t, beta_t)

        _, (y_path, beta_path) = lax.scan(step, beta_0, (g_samp, w_samp, z_samp, design))
        return y_path, beta_path

    return jax.vmap(_one_path)(keys)
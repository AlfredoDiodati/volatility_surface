import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular
from models._solver import lbfgs


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


def _solve_weights_ff(eta, phi, K):
    def _single(ep):
        eta_k = ep[0]
        phi_k = ep[1]
        indices = np.arange(K + 1)
        lambdas_k = eta_k * np.power(phi_k, -indices)

        def _rec_step(c_stack, k):
            i_range = np.arange(K + 1)
            mask = i_range < k
            diff = k - i_range
            terms = (
                c_stack
                * np.power(phi_k, -eta_k * diff)
                * np.exp(eta_k * (1.0 - np.power(phi_k, -diff)))
            )
            c_next = 1.0 - np.sum(np.where(mask, terms, 0.0))
            return c_stack.at[k].set(c_next), None

        c_init = np.zeros(K + 1).at[0].set(1.0)
        c_final, _ = lax.scan(_rec_step, c_init, np.arange(1, K + 1))
        c_ordered = np.flip(c_final)
        w = np.exp(eta_k) * c_ordered * np.power(phi_k, -indices * eta_k)
        return w, lambdas_k

    ws, lambdas = jax.vmap(_single)(np.stack([eta, phi], axis=-1))
    return ws.T, lambdas.T


_KAPPA_TOL = 1e-8


def _compute_L_matrix(ws, lambdas):
    K1 = ws.shape[0]
    rhos = np.exp(-lambdas)
    one_minus_rho2 = 1.0 - rhos ** 2
    ws_ratio = np.sqrt(ws[None, :, :] / ws[:, None, :])
    sqrt_prod = np.sqrt(one_minus_rho2[:, None, :] * one_minus_rho2[None, :, :])
    rho_cross = 1.0 - rhos[:, None, :] * rhos[None, :, :]
    return ws_ratio * sqrt_prod / rho_cross * (1.0 - np.eye(K1)[:, :, None])


@jax.custom_vjp
def _solve_kappas_ff(ws, lambdas):
    L = _compute_L_matrix(ws, lambdas)
    L_lower = np.tril(L, k=-1)
    K1 = ws.shape[0]

    def _init_step(mu, i):
        z_i = np.sum(L_lower[i] * mu, axis=0)
        mu_i = (-z_i + np.sqrt(z_i ** 2 + 4.0)) / 2.0
        return mu.at[i].set(mu_i), None

    mu0 = np.zeros_like(ws).at[0].set(1.0)
    mu_init, _ = lax.scan(_init_step, mu0, np.arange(1, K1))

    def _cond(state):
        mu_curr, mu_prev = state
        return np.max(np.abs(mu_curr - mu_prev)) > _KAPPA_TOL

    def _body(state):
        mu_curr, _ = state
        z = np.einsum('ijk,jk->ik', L, mu_curr)
        return (-z + np.sqrt(z ** 2 + 4.0)) / 2.0, mu_curr

    mu_final, _ = lax.while_loop(_cond, _body, (mu_init, np.zeros_like(mu_init)))
    return mu_final


def _solve_kappas_ff_fwd(ws, lambdas):
    mu_star = _solve_kappas_ff(ws, lambdas)
    return mu_star, (mu_star, ws, lambdas)


def _solve_kappas_ff_bwd(res, g):
    mu_star, ws, lambdas = res
    L = _compute_L_matrix(ws, lambdas)

    z = np.einsum('ijk,jk->ik', L, mu_star)
    dFdz = -mu_star / (2.0 * mu_star + z)

    def _lin_cond(state):
        v, v_prev = state
        return np.max(np.abs(v - v_prev)) > _KAPPA_TOL

    def _lin_body(state):
        v, _ = state
        JFT_v = np.einsum('ijk,ik->jk', L * dFdz[:, None, :], v)
        return g + JFT_v, v

    v_final, _ = lax.while_loop(_lin_cond, _lin_body, (g, np.zeros_like(g)))

    def _F_params(ws_, lambdas_):
        L_ = _compute_L_matrix(ws_, lambdas_)
        z_ = np.einsum('ijk,jk->ik', L_, mu_star)
        return (-z_ + np.sqrt(z_ ** 2 + 4.0)) / 2.0

    _, vjp_fn = jax.vjp(_F_params, ws, lambdas)
    return vjp_fn(v_final)


_solve_kappas_ff.defvjp(_solve_kappas_ff_fwd, _solve_kappas_ff_bwd)


def _score_step(Z_mask, eps_t, N_t, h_inv, L_C, nu, A):
    p = L_C.shape[0]
    ZtHinvZ = h_inv * (Z_mask.T @ Z_mask)
    WLC = ZtHinvZ @ L_C
    L_Inner = np.linalg.cholesky(np.eye(p) + L_C.T @ WLC)
    log_det = -N_t * np.log(h_inv) + 2.0 * np.sum(np.log(np.diag(L_Inner)))
    V = solve_triangular(L_Inner, WLC.T, lower=True)
    fisher = (nu / (nu - 2.0)) * ((nu + N_t) / (nu + N_t + 2.0)) * (ZtHinvZ - V.T @ V)
    ZtHinv_e = h_inv * (Z_mask.T @ eps_t)
    we = solve_triangular(L_Inner, L_C.T @ ZtHinv_e, lower=True)
    mahal = h_inv * (eps_t @ eps_t) - we @ we
    nabla = ((nu + N_t) / ((nu - 2.0) + mahal)) * (ZtHinv_e - V.T @ we)
    eigvals, eigvecs = np.linalg.eigh(fisher)
    return A @ (eigvecs * (1.0 / eigvals)) @ eigvecs.T @ nabla, mahal, log_det


def _link(theta, p, n_buckets):
    idx = 0
    beta_bar = theta[idx:idx + p]; idx += p
    A = np.diag(theta[idx:idx + p]); idx += p
    sigma2 = np.exp(theta[idx]); idx += 1
    omega_load = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]]); idx += n_buckets - 1
    eta = 2.0 * jax.nn.sigmoid(theta[idx:idx + p]); idx += p
    phi = np.full(p, jax.nn.softplus(theta[idx]) + 7.5); idx += 1
    C = np.diag(np.exp(theta[idx:idx + p])); idx += p
    nu = np.exp(theta[idx]) + 2.0
    return {"beta_bar": beta_bar, "A": A, "sigma2": sigma2,
            "omega_load": omega_load, "eta": eta, "phi": phi, "C": C, "nu": nu}


def _invlink(params):
    unc_s2 = np.log(params["sigma2"])
    unc_omega_load = params["omega_load"][1:]
    unc_eta = np.log(params["eta"] / (2.0 - params["eta"]))
    unc_phi = np.log(np.exp(params["phi"][0] - 7.5) - 1.0)
    unc_C = np.log(np.diag(params["C"]))
    unc_nu = np.log(params["nu"] - 2.0)
    return np.concatenate([
        params["beta_bar"],
        np.diag(params["A"]),
        np.array([unc_s2]),
        unc_omega_load,
        unc_eta,
        np.array([unc_phi]),
        unc_C,
        np.array([unc_nu]),
    ])


def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, state0):
    A = params["A"]
    sigma2 = params["sigma2"]
    omega_load = params["omega_load"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]

    h_inv = 1.0 / sigma2
    L_C = np.linalg.cholesky(C)
    ws, lambdas = _solve_weights_ff(params["eta"], params["phi"], K)
    rhos = np.exp(-lambdas)
    mus = _solve_kappas_ff(ws, lambdas)
    innovation_scale = mus * np.sqrt((1.0 - rhos ** 2) / ws)

    def _step(b_t, inputs):
        y_t, base_t, bidx_t, mask_t = inputs

        beta_t = beta_bar + np.sum(ws * b_t, axis=0)
        omega_col = omega_load[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        N_t = np.sum(mask_t)
        eps_t = (y_t - Z_t @ beta_t) * mask_t

        scaled_score, mahal, log_det = _score_step(Z_mask, eps_t, N_t, h_inv, L_C, nu, A)
        ll_t = (
            gammaln((nu + N_t) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
            - 0.5 * log_det
            - 0.5 * (nu + N_t) * np.log(1.0 + mahal / (nu - 2.0))
        )
        b_next = b_t * rhos + innovation_scale * scaled_score[None, :]
        return b_next, (ll_t, beta_t)

    b_T, (lls, betas_prev) = lax.scan(
        _step, state0,
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

    def _criterion(theta):
        params = _link(theta, p, n_buckets)
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, np.zeros((K + 1, p)))
        return -np.sum(lls)

    theta0 = np.asarray(_invlink(initial_guess))
    theta_opt, niter, final_loss, is_converged = lbfgs(_criterion, theta0, opt_options, maxiter)
    params_opt = _link(theta_opt, p, n_buckets)
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

    def _link_flat(theta):
        d = _link(theta, p, n_buckets)
        return np.concatenate([
            d["beta_bar"], np.diag(d["A"]), np.array([d["sigma2"]]),
            d["omega_load"][1:], d["eta"], np.array([d["phi"][0]]),
            np.diag(d["C"]), np.array([d["nu"]]),
        ])

    def _criterion(theta):
        params = _link(theta, p, n_buckets)
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params, K, np.zeros((K + 1, p)))
        return -np.sum(lls)

    theta_opt = _invlink(fit_result)
    H = jax.hessian(_criterion)(theta_opt)
    eigvals, eigvecs = np.linalg.eigh(H)
    H_pd = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    cov_theta = np.linalg.inv(H_pd)
    J = jax.jacobian(_link_flat)(theta_opt)
    cov_natural = J @ cov_theta @ J.T
    se_flat = np.sqrt(np.abs(np.diag(cov_natural)))

    idx = 0
    se = {}
    se["beta_bar"] = se_flat[idx:idx + p]; idx += p
    se["A_diag"] = se_flat[idx:idx + p]; idx += p
    se["sigma2"] = se_flat[idx]; idx += 1
    se["omega_load"] = se_flat[idx:idx + n_buckets - 1]; idx += n_buckets - 1
    eta_idx = idx
    se["eta"] = se_flat[idx:idx + p]; idx += p
    se["phi"] = se_flat[idx:idx + 1]; idx += 1
    se["C_diag"] = se_flat[idx:idx + p]; idx += p
    se["nu"] = se_flat[idx]

    cov_eta = cov_natural[eta_idx:eta_idx + p, eta_idx:eta_idx + p]
    return se, cov_eta


def forecast(fit_result, M, y_test, K, alpha):
    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = M.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, fit_result, K, state0=fit_result["b_T"])

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
    A = fit_result["A"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cols = omega_load[bucket_indices]
    M_base = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)

    ws, lambdas = _solve_weights_ff(fit_result["eta"], fit_result["phi"], K)
    rhos = np.exp(-lambdas)
    mus = _solve_kappas_ff(ws, lambdas)
    innovation_scale = mus * np.sqrt((1.0 - rhos ** 2) / ws)

    eval_h_arr = np.array(eval_horizons, dtype=float) - 1.0
    exp_neg_h_stack = np.exp(-eval_h_arr[:, None, None] * lambdas[None])

    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H_ext, N, -1)
    d_all = M_base @ beta_bar

    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    d_h_stack = np.stack([d_all[h - 1:T_half + h - 1] for h in eval_horizons])
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)
    d_h_scan = d_h_stack.transpose(1, 0, 2)

    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

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

        scaled_score, _, _ = _score_step(Z_mask, eps_t, N_t, h_inv, L_C, nu, A)
        b_next = b_t * rhos + innovation_scale * scaled_score[None, :]
        return b_next, preds_h

    _, preds_all = lax.scan(
        _step, fit_result["b_T"],
        (Z_h_scan, d_h_scan, y_masked, base_covariates[:T_half], bucket_indices[:T_half], mask_f),
    )
    return preds_all.transpose(1, 0, 2)


def forecast_h(fit_result, M, K):
    beta_bar = fit_result["beta_bar"]
    omega_load = fit_result["omega_load"]

    M = np.asarray(M, dtype=float)
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    n_h = base_covariates.shape[1]

    ws, lambdas = _solve_weights_ff(fit_result["eta"], fit_result["phi"], K)
    rhos = np.exp(-lambdas)

    def _step(b_t, inputs):
        base_t, bidx_t = inputs
        beta_t = beta_bar + np.sum(ws * b_t, axis=0)
        b_next = b_t * rhos
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
    C = params["C"]
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
    L_C = np.linalg.cholesky(C)
    ws, lambdas = _solve_weights_ff(params["eta"], params["phi"], K)
    rhos = np.exp(-lambdas)
    mus = _solve_kappas_ff(ws, lambdas)
    innovation_scale = mus * np.sqrt((1.0 - rhos ** 2) / ws)

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

            scaled_score, _, _ = _score_step(design_t, eps_t, N_obs, h_inv, L_C, nu, A)
            b_next = b_t * rhos + innovation_scale * scaled_score[None, :]
            return b_next, (y_t, beta_t)

        _, (y_path, beta_path) = lax.scan(step, b_0, (g_samp, w_samp, z_samp, design))
        return y_path, beta_path

    return jax.vmap(_one_path)(keys)

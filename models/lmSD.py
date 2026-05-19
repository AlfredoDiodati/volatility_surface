import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln

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

def _compute_weights(d, T):
    def _step(w_prev, tau):
        w_next = w_prev * (tau - 1.0 - d) / tau
        return w_next, w_next

    taus = np.arange(1, T, dtype=float)
    _, w_rest = lax.scan(_step, np.ones_like(d), taus)
    return np.concatenate([np.ones((1, d.shape[0])), w_rest], axis=0)

def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, state0):
    B = params["B"]
    A = params["A"]
    d = params["d"]
    sigma2 = params["sigma2"]
    omega = params["omega"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C

    beta0, score_buf0 = state0
    T_buf = score_buf0.shape[0]
    A_diag = np.diag(A)
    weights = _compute_weights(d, T_buf)

    def _step(carry, inputs):
        beta_t, score_buf = carry
        y_t, base_t, bidx_t, mask_t = inputs

        omega_col = omega[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        eps_t = (y_t - Z_t @ beta_t) * mask_t
        N_t = np.sum(mask_t)

        V_t = h_inv * (Z_mask.T @ Z_mask)
        G_t = h_inv * (Z_mask.T @ eps_t)
        S = np.diag(C_inv) + V_t

        mahal_H = h_inv * np.sum(eps_t ** 2)
        S_inv_G = np.linalg.solve(S, G_t)
        mahal_F = mahal_H - G_t @ S_inv_G

        weight = (1.0 + (N_t + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
        S_inv_V = np.linalg.solve(S, V_t)
        g_tilde = G_t - V_t @ S_inv_G
        V_tilde = V_t - V_t @ S_inv_V
        s_t = weight * np.linalg.solve(V_tilde, g_tilde)

        score_buf_new = np.roll(score_buf, 1, axis=0).at[0].set(s_t)
        conv = A_diag * (weights * score_buf_new).sum(axis=0)

        beta_next = beta_bar + B @ (beta_t - beta_bar) + conv

        log_det_F = N_t * np.log(sigma2) + np.sum(np.log(C)) + np.linalg.slogdet(S)[1]
        log_constants = (gammaln((nu + N_t) / 2.0) - gammaln(nu / 2.0)
                         - 0.5 * N_t * np.log((nu - 2.0) * np.pi))
        log_lik = (log_constants - 0.5 * log_det_F
                   - 0.5 * (nu + N_t) * np.log(1.0 + mahal_F / (nu - 2.0)))

        return (beta_next, score_buf_new), (beta_t, log_lik, beta_next)

    final_carry, (betas, log_liks, beta_Ts) = lax.scan(
        _step, (beta0, score_buf0), (y_masked, base_covariates, bucket_indices, mask_f)
    )
    return betas, log_liks, beta_Ts[-1], final_carry[1]

def fit(
    data: np.ndarray,
    M: np.ndarray,
    initial_guess: dict,
    opt_options: dict | None = None,
    maxiter: int = 5000,
):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    T = data.shape[0]
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
        d = jax.nn.sigmoid(theta[idx:idx + p])
        idx += p
        sigma2 = np.exp(theta[idx])
        idx += 1
        omega = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]])
        idx += n_buckets - 1
        C_diag = np.exp(theta[idx:idx + p])
        idx += p
        nu = np.exp(theta[idx]) + 2.0
        return {"beta_bar": beta_bar, "B": B, "A": A, "d": d, "sigma2": sigma2,
                "omega": omega, "C": C_diag, "nu": nu}

    def _invlink(params):
        B_diag = np.diag(params["B"])
        unc_B = np.arctanh(B_diag)
        unc_A = np.diag(params["A"])
        d = params["d"]
        unc_d = np.log(d / (1.0 - d))
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
        score_buf0 = np.zeros((T, p))
        _, lls, _, _ = _filter(
            y_masked, base_covariates, bucket_indices, mask_f,
            params, (params["beta_bar"], score_buf0)
        )
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
    score_buf0 = np.zeros((T, p))
    betas, _, beta_T, score_buf_T = _filter(
        y_masked, base_covariates, bucket_indices, mask_f,
        params_opt, (params_opt["beta_bar"], score_buf0)
    )
    return params_opt | {
        "betas": betas,
        "beta_T": beta_T,
        "score_buf_T": score_buf_T,
        "log_likelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
    }

def simulate(params, M, horizon, key, beta_0=None, score_buf_size=None, score_buf_0=None):
    B = params["B"]
    A = params["A"]
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
    omega_col = omega[bidx_fixed]
    design = np.concatenate([base_fixed, omega_col[:, None]], axis=-1)
    N = design.shape[0]

    if score_buf_size is None: score_buf_size = horizon
    if beta_0 is None: beta_0 = beta_bar
    if score_buf_0 is None: score_buf_0 = np.zeros((score_buf_size, p))

    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C
    V = h_inv * (design.T @ design)
    S_base = np.diag(C_inv) + V
    S_inv_V = np.linalg.solve(S_base, V)
    V_tilde = V - V @ S_inv_V

    A_diag = np.diag(A)
    weights = _compute_weights(d, score_buf_size)
    sqrt_C = np.sqrt(C)
    sqrt_sigma = np.sqrt(sigma2)

    key, k1, k2, k3 = jax.random.split(key, 4)
    g_samp = jax.random.chisquare(k1, nu, shape=(horizon,))
    w_samp = jax.random.normal(k2, shape=(horizon, p))
    z_samp = jax.random.normal(k3, shape=(horizon, N))

    def step(carry, inputs):
        beta_t, score_buf = carry
        g, w, z = inputs

        scale = np.sqrt((nu - 2.0) / g)
        eps_t = scale * (design @ (sqrt_C * w) + sqrt_sigma * z)
        y_t = design @ beta_t + eps_t

        G_t = h_inv * (design.T @ eps_t)
        S_inv_G = np.linalg.solve(S_base, G_t)
        mahal_H = h_inv * np.sum(eps_t ** 2)
        mahal_F = mahal_H - G_t @ S_inv_G
        wt = (1.0 + (N + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
        g_tilde = G_t - V @ S_inv_G
        s_t = wt * np.linalg.solve(V_tilde, g_tilde)

        score_buf_new = np.roll(score_buf, 1, axis=0).at[0].set(s_t)
        conv = A_diag * (weights * score_buf_new).sum(axis=0)
        beta_next = beta_bar + B @ (beta_t - beta_bar) + conv

        return (beta_next, score_buf_new), (y_t, beta_t)

    _, (y_sim, beta_sim) = lax.scan(
        step, (beta_0, score_buf_0), (g_samp, w_samp, z_samp)
    )
    return y_sim, beta_sim

def simulate_panel(params, M, n, key, beta_0=None, score_buf_size=None, score_buf_0=None):
    B = params["B"]
    A = params["A"]
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
    omega_cols = omega[bidx]
    design = np.concatenate([base, omega_cols[:, :, None]], axis=-1)

    if score_buf_size is None: score_buf_size = T
    if beta_0 is None: beta_0 = beta_bar
    if score_buf_0 is None: score_buf_0 = np.zeros((score_buf_size, p))

    h_inv = 1.0 / sigma2
    C_inv = 1.0 / C
    A_diag = np.diag(A)
    weights = _compute_weights(d, score_buf_size)
    sqrt_C = np.sqrt(C)
    sqrt_sigma = np.sqrt(sigma2)

    keys = jax.random.split(key, n)

    def _one_path(k):
        k1, k2, k3 = jax.random.split(k, 3)
        g_samp = jax.random.chisquare(k1, nu, shape=(T,))
        w_samp = jax.random.normal(k2, shape=(T, p))
        z_samp = jax.random.normal(k3, shape=(T, N_obs))

        def step(carry, inputs):
            beta_t, score_buf = carry
            g, w, z, design_t = inputs

            V_t = h_inv * (design_t.T @ design_t)
            S_t = np.diag(C_inv) + V_t
            S_inv_V_t = np.linalg.solve(S_t, V_t)
            V_tilde_t = V_t - V_t @ S_inv_V_t

            scale = np.sqrt((nu - 2.0) / g)
            eps_t = scale * (design_t @ (sqrt_C * w) + sqrt_sigma * z)
            y_t = design_t @ beta_t + eps_t

            G_t = h_inv * (design_t.T @ eps_t)
            S_inv_G = np.linalg.solve(S_t, G_t)
            mahal_H = h_inv * np.sum(eps_t ** 2)
            mahal_F = mahal_H - G_t @ S_inv_G
            wt = (1.0 + (N_obs + 2.0) / nu) / (1.0 + mahal_F / (nu - 2.0))
            g_tilde = G_t - V_t @ S_inv_G
            s_t = wt * np.linalg.solve(V_tilde_t, g_tilde)

            score_buf_new = np.roll(score_buf, 1, axis=0).at[0].set(s_t)
            conv = A_diag * (weights * score_buf_new).sum(axis=0)
            beta_next = beta_bar + B @ (beta_t - beta_bar) + conv

            return (beta_next, score_buf_new), (y_t, beta_t)

        _, (y_path, beta_path) = lax.scan(
            step, (beta_0, score_buf_0), (g_samp, w_samp, z_samp, design)
        )
        return y_path, beta_path

    return jax.vmap(_one_path)(keys)

def forecast(fit_result, M, y_test, alpha):
    p = fit_result["beta_bar"].shape[0]
    H = M.shape[0]
    score_buf_init = np.concatenate([
        fit_result["score_buf_T"],
        np.zeros((H, p)),
    ], axis=0)
    state0 = (fit_result["beta_T"], score_buf_init)

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H = M.shape[0]
    base_covariates = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    mask_bool = ~np.isnan(y_test)
    y_masked = np.where(mask_bool, y_test, 0.0)
    mask_f = mask_bool.astype(float)

    betas, log_liks, _, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, fit_result, state0)

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
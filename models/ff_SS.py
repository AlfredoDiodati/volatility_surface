import jax
import jax.numpy as np
from jax import lax
from models._kalman import _filter_light_univariate, _fit
from models.ff_SD import _solve_weights_ff


def _dynamics(y, _a, _P, params, _Z, _T, _H, _R, _Q, idx):
    raw = jax.lax.dynamic_index_in_dim(params["covariates"], idx, axis=0, keepdims=False)
    base_cols = raw[:, :-1]
    bucket_indices = raw[:, -1].astype(int)
    omega_col = params["omega"][bucket_indices]
    M_t = np.concatenate([base_cols, omega_col[:, None]], axis=-1) # (max_n, p)
    ws = params["ws"] # (K+1, p)
    Z_t = (M_t[:, None, :] * ws[None, :, :]).reshape(M_t.shape[0], -1) # (max_n, (K+1)*p)
    d_t = M_t @ params["beta_bar"] # (max_n,)
    return Z_t, params["T_aug"], params["H_obs"], _R, params["Q_aug"], d_t, 0.0


def _build_M_base(M, omega):
    base_cols = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cube = omega[bucket_indices]
    return np.concatenate([base_cols, omega_cube[:, :, None]], axis=-1) # (H, N, p)


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
    _, max_n, _ = M.shape
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    state_dim = (K + 1) * p
    maxiter = int(maxiter)
    opt_options = opt_options or {}

    def _link(theta):
        i = 0
        beta_bar = theta[i:i + p]; i += p
        sigma2 = np.exp(theta[i]); i += 1
        q_diag = np.exp(theta[i:i + p]); i += p
        omega = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        eta = 2.0 * jax.nn.sigmoid(theta[i:i + p]); i += p
        alpha = 1.0 + jax.nn.softplus(theta[i])

        ws, lambdas = _solve_weights_ff(eta, np.full(p, alpha), K)
        T_aug = np.diag(np.exp(-lambdas).ravel())
        Q_aug = np.diag(np.tile(q_diag, K + 1))
        H_obs = sigma2 * np.eye(max_n, dtype=float)

        return {
            "beta_bar": beta_bar,
            "sigma2": sigma2,
            "Q_param": np.diag(q_diag),
            "omega": omega,
            "eta": eta,
            "alpha": alpha,
            "ws": ws,
            "lambdas": lambdas,
            "T_aug": T_aug,
            "Q_aug": Q_aug,
            "H_obs": H_obs,
        }

    def _invlink(params):
        return np.concatenate([
            params["beta_bar"],
            np.array([np.log(params["sigma2"])]),
            np.log(np.diag(params["Q_param"])),
            params["omega"][1:],
            np.log(params["eta"] / (2.0 - params["eta"])),
            np.array([np.log(np.exp(params["alpha"] - 1.0) - 1.0)]),
        ])

    init = _link(_invlink(initial_guess))
    carry_initial = (
        np.zeros(state_dim, dtype=float),
        10.0 * np.eye(state_dim, dtype=float),
        np.zeros((max_n, state_dim), dtype=float),
        init["T_aug"],
        init["H_obs"],
        np.eye(state_dim, dtype=float),
        init["Q_aug"],
        np.asarray(0, dtype=np.int32),
    )

    result = _fit(
        data, initial_guess, M, carry_initial,
        _dynamics, _link, _invlink, opt_options,
        maxiter=maxiter,
        _filter_fn=_filter_light_univariate,
    )

    param_keys = ["beta_bar", "sigma2", "Q_param", "omega", "eta", "alpha", "ws", "lambdas", "T_aug", "Q_aug"]
    kf_keys = ["logdetF", "quad", "a", "P", "att", "Ptt", "v", "F", "K"]
    return (
        {k: result[k] for k in param_keys}
        | {k: result[k] for k in kf_keys}
        | {"loglikelihood": result["loglikelihood"], "niter": result["niter"], "is_converged": result["is_converged"]}
    )


def forecast(fit_result, M, y_test, q_alpha):
    a0 = fit_result["att"][-1]
    P0 = fit_result["Ptt"][-1]
    T_aug = fit_result["T_aug"]
    Q_aug = fit_result["Q_aug"]
    sigma2 = fit_result["sigma2"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H, N, _ = M.shape

    M_base = _build_M_base(M, omega) # (H, N, p)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H, N, -1) # (H, N, (K+1)*p)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar) # (H, N)

    def _step(carry, inputs):
        Z_h, d_h, y_h = inputs
        a_t, P_t = carry
        a_pred = T_aug @ a_t
        P_pred = T_aug @ P_t @ T_aug.T + Q_aug
        mask_h = ~np.isnan(y_h)
        n_h = np.sum(mask_h)
        y_hat = Z_h @ a_pred + d_h
        P_mean = y_hat.sum() / n_h
        z_sum = Z_h.sum(axis=0)
        F_sum = n_h * sigma2 + z_sum @ P_pred @ z_sum
        VaR_h = P_mean + q_alpha / n_h * np.sqrt(F_sum)

        v = (y_h - y_hat) * mask_h
        Z_masked = np.where(mask_h[:, None], Z_h, 0.0)
        h_inv = 1.0 / sigma2
        ZtZ = np.einsum("ip,iq->pq", Z_masked, Z_masked)
        ZtV = Z_masked.T @ v
        Inner = np.eye(a_pred.shape[0]) + h_inv * P_pred @ ZtZ
        _, log_det_corr = np.linalg.slogdet(Inner)
        log_det_F = n_h * np.log(sigma2) + log_det_corr
        c = np.linalg.solve(Inner, P_pred @ ZtV)
        quad = h_inv * np.sum(v ** 2) - h_inv ** 2 * ZtV @ c
        oos_ll = -0.5 * (n_h * np.log(2.0 * np.pi) + log_det_F + quad)

        D = np.linalg.solve(Inner, P_pred @ ZtZ)
        a_filtered = a_pred + h_inv * c
        P_filtered = P_pred - h_inv * D @ P_pred

        return (a_filtered, P_filtered), (y_hat, P_mean, VaR_h, oos_ll)

    _, (predictions, P_means, VaR, log_liks) = lax.scan(
        _step, (a0, P0), (Z_all, d_all, y_test)
    )
    return predictions, P_means, VaR, log_liks


def forecast_rolling_h(fit_result, M, y_test, eval_horizons):
    a0 = fit_result["att"][-1]
    P0 = fit_result["Ptt"][-1]
    T_aug = fit_result["T_aug"]
    Q_aug = fit_result["Q_aug"]
    sigma2 = fit_result["sigma2"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H_ext, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    T_h_list = []
    for h in eval_horizons:
        Th = np.eye(T_aug.shape[0], dtype=float)
        for _ in range(h):
            Th = Th @ T_aug
        T_h_list.append(Th)
    T_h_stack = np.stack(T_h_list)

    Z_h_stack = np.stack([Z_all[h - 1:T_half + h - 1] for h in eval_horizons])
    d_h_stack = np.stack([d_all[h - 1:T_half + h - 1] for h in eval_horizons])
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)
    d_h_scan = d_h_stack.transpose(1, 0, 2)
    Z_update = Z_all[:T_half]
    d_update = d_all[:T_half]
    h_inv = 1.0 / sigma2

    def _step(carry, inputs):
        a_t, P_t = carry
        Z_h_t, d_h_t, Z_upd_t, d_upd_t, y_h = inputs

        a_h_stack = np.einsum("hpq,q->hp", T_h_stack, a_t)
        preds_h = np.einsum("hnp,hp->hn", Z_h_t, a_h_stack) + d_h_t

        a_pred = T_aug @ a_t
        P_pred = T_aug @ P_t @ T_aug.T + Q_aug
        mask_h = ~np.isnan(y_h)
        y_hat = Z_upd_t @ a_pred + d_upd_t
        v = (y_h - y_hat) * mask_h
        Z_masked = np.where(mask_h[:, None], Z_upd_t, 0.0)
        ZtZ = np.einsum("ip,iq->pq", Z_masked, Z_masked)
        ZtV = Z_masked.T @ v
        Inner = np.eye(a_pred.shape[0]) + h_inv * P_pred @ ZtZ
        c = np.linalg.solve(Inner, P_pred @ ZtV)
        D = np.linalg.solve(Inner, P_pred @ ZtZ)
        a_filtered = a_pred + h_inv * c
        P_filtered = P_pred - h_inv * D @ P_pred
        return (a_filtered, P_filtered), preds_h

    _, preds_all = lax.scan(_step, (a0, P0), (Z_h_scan, d_h_scan, Z_update, d_update, y_test))
    return preds_all.transpose(1, 0, 2)


def forecast_h(fit_result, M):
    a0 = fit_result["att"][-1]
    T_aug = fit_result["T_aug"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    H, N, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    def _step(a_t, inputs):
        Z_h, d_h = inputs
        a_pred = T_aug @ a_t
        y_hat = Z_h @ a_pred + d_h
        P_mean = y_hat.sum() / N
        return a_pred, (y_hat, P_mean)

    _, (predictions, P_means) = lax.scan(_step, a0, (Z_all, d_all))
    return predictions, P_means


def simulate_panel(params, M, n, key, b_0=None):
    T_aug = params["T_aug"]
    Q_aug = params["Q_aug"]
    sigma2 = params["sigma2"]
    omega = params["omega"]
    beta_bar = params["beta_bar"]
    ws = params["ws"]
    state_dim = T_aug.shape[0]

    M = np.asarray(M, dtype=float)
    T_obs, N_obs, _ = M.shape

    M_base = _build_M_base(M, omega)                                                     # (T, N, p)
    Z = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(T_obs, N_obs, -1)        # (T, N, (K+1)*p)
    d = np.einsum("tnp,p->tn", M_base, beta_bar)                                        # (T, N)

    if b_0 is None:
        b_0 = np.zeros(state_dim, dtype=float)

    sqrt_sigma = np.sqrt(sigma2)
    L_Q = np.linalg.cholesky(Q_aug)
    keys = jax.random.split(key, n)

    def _one_path(k):
        k1, k2 = jax.random.split(k, 2)
        eta_draws = jax.random.normal(k1, shape=(T_obs, state_dim))
        eps_draws = jax.random.normal(k2, shape=(T_obs, N_obs))

        def step(b_t, inputs):
            Z_t, d_t, eta_t, eps_t = inputs
            y_t = Z_t @ b_t + d_t + sqrt_sigma * eps_t
            b_next = T_aug @ b_t + L_Q @ eta_t
            return b_next, y_t

        _, y_path = lax.scan(step, b_0, (Z, d, eta_draws, eps_draws))
        return y_path

    return jax.vmap(_one_path)(keys)

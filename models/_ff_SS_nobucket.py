import jax
import jax.numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular
from models._kalman import _filter_light_chol, _filter_chol, _fit
from models.ff_SD import _solve_weights_ff


def _dynamics(y, _a, _P, params, _Z, _T, _H, _R, _Q, idx):
    M_t = jax.lax.dynamic_index_in_dim(params["M_cube"], idx, axis=0, keepdims=False)
    ws = params["ws"]
    Z_t = (M_t[:, None, :] * ws[None, :, :]).reshape(M_t.shape[0], -1)
    d_t = M_t @ params["beta_bar"]
    return Z_t, params["T_aug"], params["H_obs"], _R, params["Q_aug"], d_t, 0.0


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
    state_dim = (K + 1) * p
    maxiter = int(maxiter)
    opt_options = opt_options or {}
    base_covariates = M[:, :, :-1]

    def _link(theta):
        i = 0
        beta_bar = theta[i:i + p]; i += p
        sigma2 = np.exp(theta[i]); i += 1
        q_diag = np.exp(theta[i:i + p]); i += p
        eta = 2.0 * jax.nn.sigmoid(theta[i:i + p]); i += p
        alpha = 1.0 + jax.nn.softplus(theta[i])

        ws, lambdas = _solve_weights_ff(eta, np.full(p, alpha), K)
        T_aug = np.diag(np.exp(-lambdas).ravel())
        Q_aug = np.diag(np.tile(q_diag, K + 1))
        result = {
            "beta_bar": beta_bar,
            "sigma2": sigma2,
            "Q_param": np.diag(q_diag),
            "eta": eta,
            "alpha": alpha,
            "ws": ws,
            "lambdas": lambdas,
            "T_aug": T_aug,
            "Q_aug": Q_aug,
            "H_obs": np.array([[sigma2]]),
            "M_cube": base_covariates,
        }
        return result

    def _invlink(params):
        return np.concatenate([
            params["beta_bar"],
            np.array([np.log(params["sigma2"])]),
            np.log(np.diag(params["Q_param"])),
            np.log(params["eta"] / (2.0 - params["eta"])),
            np.array([np.log(np.exp(params["alpha"] - 1.0) - 1.0)]),
        ])

    init = _link(_invlink(initial_guess))
    carry_initial = (
        np.zeros(state_dim, dtype=float),
        10.0 * np.eye(state_dim, dtype=float),
        np.zeros((max_n, state_dim), dtype=float),
        init["T_aug"],
        np.ones((1, 1), dtype=float),
        np.eye(state_dim, dtype=float),
        init["Q_aug"],
        np.asarray(0, dtype=np.int32),
    )

    result = _fit(
        data, initial_guess, M, carry_initial,
        _dynamics, _link, _invlink, opt_options,
        maxiter=maxiter,
        _filter_fn=_filter_light_chol,
        _filter_final_fn=_filter_chol,
    )

    param_keys = ["beta_bar", "sigma2", "Q_param", "eta", "alpha", "ws", "lambdas", "T_aug", "Q_aug"]
    kf_keys = ["logdetF", "quad", "a", "P", "att", "Ptt", "v"]
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
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H, N, _ = M.shape

    M_base = M[:, :, :-1]
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    def _step(carry, inputs):
        Z_h, d_h, y_h = inputs
        a_t, P_t = carry
        a_pred = T_aug @ a_t
        P_pred = T_aug @ P_t @ T_aug.T + Q_aug
        mask_h = ~np.isnan(y_h)
        n_h = np.sum(mask_h).astype(float)
        y_hat = Z_h @ a_pred + d_h
        P_mean = y_hat.sum() / n_h
        z_sum = Z_h.sum(axis=0)
        F_sum = n_h * sigma2 + z_sum @ P_pred @ z_sum
        VaR_h = P_mean + q_alpha / n_h * np.sqrt(F_sum)

        v = (y_h - y_hat) * mask_h
        Z_masked = np.where(mask_h[:, None], Z_h, 0.0)
        ZP = Z_masked @ P_pred
        F = ZP @ Z_masked.T + sigma2 * np.eye(N)
        L_F = np.linalg.cholesky(F)
        Linv_v = solve_triangular(L_F, v, lower=True)
        quad = np.sum(Linv_v ** 2)
        logdet_F = 2.0 * np.sum(np.log(np.diag(L_F))) - (N - n_h) * np.log(sigma2)
        oos_ll = -0.5 * (n_h * np.log(2.0 * np.pi) + logdet_F + quad)

        tmp = solve_triangular(L_F, ZP, lower=True)
        KtT = solve_triangular(L_F.T, tmp, lower=False)
        a_filtered = a_pred + KtT.T @ v
        P_filtered = P_pred - KtT.T @ ZP

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
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape

    M_base = M[:, :, :-1]
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H_ext, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    h_shifts = np.array(eval_horizons, dtype=np.int32) - 1
    t_idx = np.arange(T_half)
    Z_h_stack = Z_all[h_shifts[:, None] + t_idx[None, :]]
    d_h_stack = d_all[h_shifts[:, None] + t_idx[None, :]]
    Z_h_scan = Z_h_stack.transpose(1, 0, 2, 3)
    d_h_scan = d_h_stack.transpose(1, 0, 2)
    Z_update = Z_all[:T_half]
    d_update = d_all[:T_half]

    def _T_power(Tk, _):
        return Tk @ T_aug, Tk @ T_aug
    _, T_powers = lax.scan(_T_power, np.eye(T_aug.shape[0]), None, length=max(eval_horizons))
    T_h_stack = T_powers[h_shifts]

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
        ZP = Z_masked @ P_pred
        F = ZP @ Z_masked.T + sigma2 * np.eye(N)
        L_F = np.linalg.cholesky(F)
        tmp = solve_triangular(L_F, ZP, lower=True)
        KtT = solve_triangular(L_F.T, tmp, lower=False)
        a_filtered = a_pred + KtT.T @ v
        P_filtered = P_pred - KtT.T @ ZP
        return (a_filtered, P_filtered), preds_h

    _, preds_all = lax.scan(_step, (a0, P0), (Z_h_scan, d_h_scan, Z_update, d_update, y_test))
    return preds_all.transpose(1, 0, 2)


def forecast_h(fit_result, M):
    a0 = fit_result["att"][-1]
    T_aug = fit_result["T_aug"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    H, N, _ = M.shape

    M_base = M[:, :, :-1]
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
    beta_bar = params["beta_bar"]
    ws = params["ws"]
    state_dim = T_aug.shape[0]

    M = np.asarray(M, dtype=float)
    T_obs, N_obs, _ = M.shape

    M_base = M[:, :, :-1]
    Z = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(T_obs, N_obs, -1)
    d = np.einsum("tnp,p->tn", M_base, beta_bar)

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

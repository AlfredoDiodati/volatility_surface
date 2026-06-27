import jax
import jax.numpy as np
from jax import lax
from models._kalman import _filter_light_vec, _filter_vec
from models._solver import lbfgs

def _solve_weights_ff(eta, alpha, K):
    def _single(ea):
        eta_k = ea[0]
        alpha_k = ea[1]
        indices = np.arange(K + 1)
        lambdas_k = eta_k * np.power(alpha_k, -indices)

        def _rec_step(c_stack, k):
            i_range = np.arange(K + 1)
            mask = i_range < k
            diff = k - i_range
            terms = (
                c_stack
                * np.power(alpha_k, -eta_k * diff)
                * np.exp(eta_k * (1.0 - np.power(alpha_k, -diff)))
            )
            c_next = 1.0 - np.sum(np.where(mask, terms, 0.0))
            return c_stack.at[k].set(c_next), None

        c_init = np.zeros(K + 1).at[0].set(1.0)
        c_final, _ = lax.scan(_rec_step, c_init, np.arange(1, K + 1))
        c_ordered = np.flip(c_final)
        w = np.exp(eta_k) * c_ordered * np.power(alpha_k, -indices * eta_k)
        return w, lambdas_k

    ws, lambdas = jax.vmap(_single)(np.stack([eta, alpha], axis=-1))
    return ws.T, lambdas.T

def _dynamics(y, _a, _P, params, _Z, _T, _H, _R, _Q, idx):
    raw = jax.lax.dynamic_index_in_dim(params["covariates"], idx, axis=0, keepdims=False)
    base_cols = raw[:, :-1]
    bucket_indices = raw[:, -1].astype(int)
    omega_col = params["omega"][bucket_indices]
    M_t = np.concatenate([base_cols, omega_col[:, None]], axis=-1)
    ws = params["ws"]
    Z_t = (M_t[:, None, :] * ws[None, :, :]).reshape(M_t.shape[0], -1)
    d_t = M_t @ params["beta_bar"]
    return Z_t, params["T_aug"], params["H_obs"], _R, params["Q_aug"], d_t, 0.0


def _build_M_base(M, omega):
    base_cols = M[:, :, :-1]
    bucket_indices = M[:, :, -1].astype(np.int32)
    omega_cube = omega[bucket_indices]
    return np.concatenate([base_cols, omega_cube[:, :, None]], axis=-1)


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
        r_diag = np.exp(theta[i:i + p]); i += p
        omega = np.concatenate([np.zeros(1), theta[i:i + n_buckets - 1]]); i += n_buckets - 1
        eta = 2.0 * jax.nn.sigmoid(theta[i:i + p]); i += p
        alpha = 1.0 + jax.nn.softplus(theta[i])

        ws, lambdas = _solve_weights_ff(eta, np.full(p, alpha), K)
        T_aug = np.diag(np.exp(-lambdas).ravel())
        Q_aug = np.diag(np.tile(r_diag, K + 1))
        return {
            "beta_bar": beta_bar,
            "Q_param": np.diag(r_diag),
            "omega": omega,
            "eta": eta,
            "alpha": alpha,
            "ws": ws,
            "lambdas": lambdas,
            "T_aug": T_aug,
            "Q_aug": Q_aug,
            "H_obs": np.array([[1.0]]),
        }

    def _invlink(params):
        return np.concatenate([
            params["beta_bar"],
            np.log(np.diag(params["Q_param"]) / params["sigma2"]),
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
        np.ones((1, 1), dtype=float),
        np.eye(state_dim, dtype=float),
        init["Q_aug"],
        np.asarray(0, dtype=np.int32),
    )

    N_total = np.sum(~np.isnan(data)).astype(float)

    def _criterion(theta):
        p_ = _link(theta)
        kf = _filter_light_vec(data, _dynamics, p_ | {"covariates": M}, carry_initial)
        return N_total / 2 * np.log(np.sum(kf["quad"])) + 0.5 * np.sum(kf["logdetF"])

    unc_params0 = np.asarray(_invlink(initial_guess))
    unc_params, niter, final_loss, is_converged = lbfgs(_criterion, unc_params0, opt_options, maxiter)

    params = _link(unc_params)
    kf = _filter_vec(data, _dynamics, params | {"covariates": M}, carry_initial)
    sigma2 = np.sum(kf["quad"]) / N_total

    return {
        "beta_bar": params["beta_bar"],
        "sigma2": sigma2,
        "Q_param": params["Q_param"] * sigma2,
        "omega": params["omega"],
        "eta": params["eta"],
        "alpha": params["alpha"],
        "ws": params["ws"],
        "lambdas": params["lambdas"],
        "T_aug": params["T_aug"],
        "Q_aug": params["Q_aug"] * sigma2,
        "logdetF": kf["logdetF"],
        "quad": kf["quad"],
        "a": kf["a"],
        "P": kf["P"],
        "att": kf["att"],
        "Ptt": kf["Ptt"],
        "v": kf["v"],
        "loglikelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
    }


def eval_and_refit_k(fit_result, data, M, K):
    data = np.asarray(data, dtype=float)
    M = np.asarray(M, dtype=float)
    _, max_n, _ = M.shape
    p = fit_result["beta_bar"].shape[0]
    state_dim = (K + 1) * p
    N_total = np.sum(~np.isnan(data)).astype(float)

    r_diag = np.diag(fit_result["Q_param"]) / fit_result["sigma2"]
    ws, lambdas = _solve_weights_ff(fit_result["eta"], np.full(p, fit_result["alpha"]), K)
    T_aug = np.diag(np.exp(-lambdas).ravel())
    Q_aug = np.diag(np.tile(r_diag, K + 1))

    params_k = {
        "beta_bar": fit_result["beta_bar"],
        "Q_param": np.diag(r_diag),
        "omega": fit_result["omega"],
        "eta": fit_result["eta"],
        "alpha": fit_result["alpha"],
        "ws": ws,
        "lambdas": lambdas,
        "T_aug": T_aug,
        "Q_aug": Q_aug,
        "H_obs": np.array([[1.0]]),
    }
    carry = (
        np.zeros(state_dim, dtype=float),
        10.0 * np.eye(state_dim, dtype=float),
        np.zeros((max_n, state_dim), dtype=float),
        T_aug,
        np.ones((1, 1), dtype=float),
        np.eye(state_dim, dtype=float),
        Q_aug,
        np.asarray(0, dtype=np.int32),
    )

    kf_light = _filter_light_vec(data, _dynamics, params_k | {"covariates": M}, carry)
    ll = float(-(N_total / 2 * np.log(np.sum(kf_light["quad"])) + 0.5 * np.sum(kf_light["logdetF"])))

    kf = _filter_vec(data, _dynamics, params_k | {"covariates": M}, carry)
    sigma2 = float(np.sum(kf["quad"]) / N_total)

    return ll, {
        **fit_result,
        "ws": ws,
        "lambdas": lambdas,
        "T_aug": T_aug,
        "Q_aug": Q_aug * sigma2,
        "sigma2": sigma2,
        "Q_param": np.diag(r_diag) * sigma2,
        "a": kf["a"],
        "P": kf["P"],
        "att": kf["att"],
        "Ptt": kf["Ptt"],
        "v": kf["v"],
        "logdetF": kf["logdetF"],
        "quad": kf["quad"],
        "loglikelihood": ll,
    }


def forecast(fit_result, M, y_test, q_alpha):
    a0 = fit_result["att"][-1]
    P0 = fit_result["sigma2"] * fit_result["Ptt"][-1]
    T_aug = fit_result["T_aug"]
    Q_aug = fit_result["Q_aug"]
    sigma2 = fit_result["sigma2"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    H, N, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

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
        rhs = np.concatenate([(P_pred @ ZtV)[:, None], P_pred @ ZtZ], axis=1)
        sol = np.linalg.solve(Inner, rhs)
        c, D = sol[:, 0], sol[:, 1:]
        log_det_F = n_h * np.log(sigma2) + np.linalg.slogdet(Inner)[1]
        quad = h_inv * np.sum(v ** 2) - h_inv ** 2 * ZtV @ c
        oos_ll = -0.5 * (n_h * np.log(2.0 * np.pi) + log_det_F + quad)
        a_filtered = a_pred + h_inv * c
        P_filtered = P_pred - h_inv * D @ P_pred
        return (a_filtered, P_filtered), (y_hat, P_mean, VaR_h, oos_ll, a_filtered)

    _, (predictions, P_means, VaR, log_liks, a_hist) = lax.scan(
        _step, (a0, P0), (Z_all, d_all, y_test)
    )
    return predictions, P_means, VaR, log_liks, a_hist


def forecast_rolling_h(fit_result, M, y_test, eval_horizons):
    a0 = fit_result["att"][-1]
    P0 = fit_result["sigma2"] * fit_result["Ptt"][-1]
    T_aug = fit_result["T_aug"]
    Q_aug = fit_result["Q_aug"]
    sigma2 = fit_result["sigma2"]
    omega = fit_result["omega"]
    beta_bar = fit_result["beta_bar"]
    ws = fit_result["ws"]
    lambdas = fit_result["lambdas"].ravel()

    M = np.asarray(M, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    T_half = y_test.shape[0]
    H_ext, N, _ = M.shape

    M_base = _build_M_base(M, omega)
    Z_all = (M_base[:, :, None, :] * ws[None, None, :, :]).reshape(H_ext, N, -1)
    d_all = np.einsum("hnp,p->hn", M_base, beta_bar)

    T_h_stack = np.stack([np.diag(np.exp(-lambdas * h)) for h in eval_horizons])

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
        rhs = np.concatenate([(P_pred @ ZtV)[:, None], P_pred @ ZtZ], axis=1)
        sol = np.linalg.solve(Inner, rhs)
        c, D = sol[:, 0], sol[:, 1:]
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

    M_base = _build_M_base(M, omega)
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

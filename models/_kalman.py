"""Implementation of the Kalman filter.
Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

import jax
import jax.numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular
from models._solver import lbfgs

def _filter(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Kalman Filter implementation

    Args:
        data (ndarray): data in compatible ndarray
        dynamics (callable): function that specifies Zt, Tt, Rt and Qt
        params (dict): parameters of the model
        carry0 (tuple[float, float]): initial state prediction and variance
    """

    def _step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry

        missing = np.any(np.isnan(yt))
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)
        Zt = np.where(missing, np.zeros_like(Zt), Zt)
        yt = np.where(missing, 0.0, yt)
        vt = yt - Zt @ at - dt
        ZP = Zt @ Pt
        Ft = ZP @ Zt.T + Ht
        L = np.linalg.cholesky(Ft)
        PtZtT = Pt @ Zt.T
        x = solve_triangular(L, PtZtT.T, lower=True)
        Kt = solve_triangular(L.T, x, lower=False).T
        att = at + Kt @ vt
        atp1 = Tt @ att + ct
        Ptt = Pt - Kt @ Zt @ Pt
        Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T
        Linv_v = solve_triangular(L, vt, lower=True)
        quad_t = np.where(missing, 0.0, Linv_v.T @ Linv_v)
        logdet_t = np.where(missing, 0.0, 2.0 * np.sum(np.log(np.diag(L))))
        idx = idx + 1
        new_carry = (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx)
        return new_carry, (logdet_t, quad_t, at, Pt, att, Ptt, vt, Ft, Kt)
    
    _, (logdetF, quad, at, Pt, att, Ptt, vt, Ft, Kt) = lax.scan(_step, carry0, data)
    return {
        "logdetF": logdetF,"quad": quad,"a": at,
        "P": Pt, "att": att, "Ptt": Ptt, "v": vt,
        "F": Ft, "K": Kt,
    }

def _filter_light(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Version of filter that does not store kalman components, thus speeding up XLA"""
    def _step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry

        missing = np.any(np.isnan(yt))
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)
        Zt = np.where(missing, np.zeros_like(Zt), Zt)
        yt = np.where(missing, 0.0, yt)
        vt = yt - Zt @ at - dt
        ZP = Zt @ Pt
        Ft = ZP @ Zt.T + Ht
        L = np.linalg.cholesky(Ft)
        PtZtT = Pt @ Zt.T
        x = solve_triangular(L, PtZtT.T, lower=True)
        Kt = solve_triangular(L.T, x, lower=False).T
        att = at + Kt @ vt
        atp1 = Tt @ att + ct
        Ptt = Pt - Kt @ Zt @ Pt
        Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T
        Linv_v = solve_triangular(L, vt, lower=True)
        quad_t = np.where(missing, 0.0, Linv_v.T @ Linv_v)
        logdet_t = np.where(missing, 0.0, 2.0 * np.sum(np.log(np.diag(L))))
        idx = idx + 1
        new_carry = (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx)
        return new_carry, (logdet_t, quad_t)

    _, (logdetF, quad) = lax.scan(_step, carry0, data)
    return {"logdetF": logdetF, "quad": quad}

def _filter_light_univariate(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Univariate treatment of the Kalman filter. Requires diagonal Ht.
    Replaces the O(p^3) Cholesky at each t with p scalar divisions."""
    def _outer_step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)
        H_diag = np.diag(Ht)
        dt_vec = np.zeros(Zt.shape[0], dtype=float) + np.asarray(dt, dtype=float)

        def _inner_step(inner_carry, obs_slice):
            at_i, Pt_i, logdet_acc, quad_acc = inner_carry
            y_it, Z_it, H_it, dt_it = obs_slice
            missing = np.isnan(y_it)
            Z_it = np.where(missing, np.zeros_like(Z_it), Z_it)
            y_it = np.where(missing, 0.0, y_it)
            F_it = Z_it @ Pt_i @ Z_it + H_it
            K_it = Pt_i @ Z_it / F_it
            v_it = y_it - Z_it @ at_i - dt_it
            at_next = at_i + K_it * v_it
            Pt_next = Pt_i - np.outer(K_it, K_it) * F_it
            logdet_acc = logdet_acc + np.where(missing, 0.0, np.log(F_it))
            quad_acc = quad_acc + np.where(missing, 0.0, v_it * v_it / F_it)
            return (at_next, Pt_next, logdet_acc, quad_acc), None

        (at_f, Pt_f, logdet_t, quad_t), _ = lax.scan(
            _inner_step, (at, Pt, 0.0, 0.0), (yt, Zt, H_diag, dt_vec)
        )
        at_tp1 = Tt @ at_f + ct
        Pt_tp1 = Tt @ Pt_f @ Tt.T + Rt @ Qt @ Rt.T
        return (at_tp1, Pt_tp1, Zt, Tt, Ht, Rt, Qt, idx + 1), (logdet_t, quad_t)

    _, (logdetF, quad) = lax.scan(_outer_step, carry0, data)
    return {"logdetF": logdetF, "quad": quad}

def _filter_light_vec(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Vectorized Kalman filter using Woodbury identity for H_obs = sigma2 * I_N.
    Efficient when state_dim << N. Replaces the inner scan over N observations with
    state_dim x state_dim linear solves."""
    def _step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)

        sigma2 = Ht[0, 0]
        h_inv = 1.0 / sigma2
        dt_vec = np.zeros(Zt.shape[0], dtype=float) + np.asarray(dt, dtype=float)
        state_dim = at.shape[0]

        mask = ~np.isnan(yt)
        n_t = np.sum(mask).astype(float)
        Z_masked = np.where(mask[:, None], Zt, 0.0)
        v = (yt - Zt @ at - dt_vec) * mask

        ZtZ = np.einsum("ip,iq->pq", Z_masked, Z_masked)
        ZtV = Z_masked.T @ v
        Inner = np.eye(state_dim) + h_inv * Pt @ ZtZ
        _, log_det_corr = np.linalg.slogdet(Inner)
        logdet_t = n_t * np.log(sigma2) + log_det_corr

        c = np.linalg.solve(Inner, Pt @ ZtV)
        quad_t = h_inv * np.sum(v ** 2) - h_inv ** 2 * ZtV @ c

        D = np.linalg.solve(Inner, Pt @ ZtZ)
        att = at + h_inv * c
        Ptt = Pt - h_inv * D @ Pt

        atp1 = Tt @ att + ct
        Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T

        return (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx + 1), (logdet_t, quad_t)

    _, (logdetF, quad) = lax.scan(_step, carry0, data)
    return {"logdetF": logdetF, "quad": quad}

def _filter_vec(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Full Kalman filter with Woodbury identity for H_obs = sigma2 * I_N.
    Returns att, Ptt, v etc. without forming N×N matrices or storing N×N F/K."""
    def _step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)

        sigma2 = Ht[0, 0]
        h_inv = 1.0 / sigma2
        dt_vec = np.zeros(Zt.shape[0], dtype=float) + np.asarray(dt, dtype=float)
        state_dim = at.shape[0]

        mask = ~np.isnan(yt)
        n_t = np.sum(mask).astype(float)
        Z_masked = np.where(mask[:, None], Zt, 0.0)
        v = (yt - Zt @ at - dt_vec) * mask

        ZtZ = np.einsum("ip,iq->pq", Z_masked, Z_masked)
        ZtV = Z_masked.T @ v
        Inner = np.eye(state_dim) + h_inv * Pt @ ZtZ
        _, log_det_corr = np.linalg.slogdet(Inner)
        logdet_t = n_t * np.log(sigma2) + log_det_corr

        c = np.linalg.solve(Inner, Pt @ ZtV)
        quad_t = h_inv * np.sum(v ** 2) - h_inv ** 2 * ZtV @ c

        D = np.linalg.solve(Inner, Pt @ ZtZ)
        att = at + h_inv * c
        Ptt = Pt - h_inv * D @ Pt

        atp1 = Tt @ att + ct
        Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T

        return (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx + 1), (logdet_t, quad_t, at, Pt, att, Ptt, v)

    _, (logdetF, quad, a, P, att, Ptt, v) = lax.scan(_step, carry0, data)
    return {"logdetF": logdetF, "quad": quad, "a": a, "P": P, "att": att, "Ptt": Ptt, "v": v}

def _filter_light_chol(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Kalman filter using direct N×N Cholesky for H_obs = sigma2 * I_N.
    Numerically stable for any parameter values; efficient when state_dim >= N."""
    def _step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)

        sigma2 = Ht[0, 0]
        N_obs = Zt.shape[0]
        dt_vec = np.zeros(N_obs, dtype=float) + np.asarray(dt, dtype=float)

        mask = ~np.isnan(yt)
        n_t = np.sum(mask).astype(float)
        Z_masked = np.where(mask[:, None], Zt, 0.0)
        v = (yt - Zt @ at - dt_vec) * mask

        ZP = Z_masked @ Pt
        F = ZP @ Z_masked.T + sigma2 * np.eye(N_obs)
        L_F = np.linalg.cholesky(F)
        Linv_v = solve_triangular(L_F, v, lower=True)
        quad_t = np.sum(Linv_v ** 2)
        logdet_F = 2.0 * np.sum(np.log(np.diag(L_F)))
        logdet_t = logdet_F - (N_obs - n_t) * np.log(sigma2)

        tmp = solve_triangular(L_F, ZP, lower=True)
        KtT = solve_triangular(L_F.T, tmp, lower=False)
        att = at + KtT.T @ v
        Ptt = Pt - KtT.T @ ZP

        atp1 = Tt @ att + ct
        Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T

        return (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx + 1), (logdet_t, quad_t)

    _, (logdetF, quad) = lax.scan(_step, carry0, data)
    return {"logdetF": logdetF, "quad": quad}

def _filter_chol(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Full Kalman filter with direct N×N Cholesky for H_obs = sigma2 * I_N.
    Numerically stable for any parameter values; efficient when state_dim >= N."""
    def _step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)

        sigma2 = Ht[0, 0]
        N_obs = Zt.shape[0]
        dt_vec = np.zeros(N_obs, dtype=float) + np.asarray(dt, dtype=float)

        mask = ~np.isnan(yt)
        n_t = np.sum(mask).astype(float)
        Z_masked = np.where(mask[:, None], Zt, 0.0)
        v = (yt - Zt @ at - dt_vec) * mask

        ZP = Z_masked @ Pt
        F = ZP @ Z_masked.T + sigma2 * np.eye(N_obs)
        L_F = np.linalg.cholesky(F)
        Linv_v = solve_triangular(L_F, v, lower=True)
        quad_t = np.sum(Linv_v ** 2)
        logdet_F = 2.0 * np.sum(np.log(np.diag(L_F)))
        logdet_t = logdet_F - (N_obs - n_t) * np.log(sigma2)

        tmp = solve_triangular(L_F, ZP, lower=True)
        KtT = solve_triangular(L_F.T, tmp, lower=False)
        att = at + KtT.T @ v
        Ptt = Pt - KtT.T @ ZP

        atp1 = Tt @ att + ct
        Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T

        return (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx + 1), (logdet_t, quad_t, at, Pt, att, Ptt, v)

    _, (logdetF, quad, a, P, att, Ptt, v) = lax.scan(_step, carry0, data)
    return {"logdetF": logdetF, "quad": quad, "a": a, "P": P, "att": att, "Ptt": Ptt, "v": v}

def _loglikelihood(filter_output: dict):
    """Without constant term"""
    return -0.5 * np.sum(filter_output["logdetF"] + filter_output["quad"])

def _fit(
    data: np.ndarray,
    initial_guess: dict,
    covariates: np.ndarray | None,
    carry_initial: tuple,
    _dynamics: callable,
    _link: callable = lambda x: x,
    _invlink: callable = lambda x: x,
    opt_options: dict = {}, maxiter: int = 5000,
    extra_loglikelihood_fn: callable = lambda _, __: 0.0,
    extra_ll_data: np.ndarray = None,
    _filter_fn: callable = _filter_light,
    _filter_final_fn: callable = None) -> dict:
    """
    Args:
        data (np.ndarray)
        params (dict)
        covariates (np.ndarray): _description_
        _dynamics (callable): _description_
        _link (callable | None, optional): function that maps uncostrained, ndarray parameters to
        constrained space and returns them in a dictionary. Defaults to None.
        _invlink (callable | None, optional): inverse of _link. Defaults to None.
    """
    maxiter = int(maxiter)
    initial_guess = dict(initial_guess)
    initial_guess["covariates"] = covariates

    def _criterion(params):
        constr_params = _link(params)
        kf = _filter_fn(data, _dynamics, constr_params | {"covariates": covariates}, carry_initial)
        return -(_loglikelihood({"logdetF": kf["logdetF"], "quad": kf["quad"]}) + extra_loglikelihood_fn(constr_params, extra_ll_data))

    unc_params0 = np.asarray(_invlink(initial_guess))
    unc_params, niter, final_loss, is_converged = lbfgs(_criterion, unc_params0, opt_options, maxiter)

    final_filter = _filter_final_fn if _filter_final_fn is not None else _filter
    params = _link(unc_params)
    params = dict(params)
    params["covariates"] = covariates
    kf = final_filter(data, _dynamics, params, carry_initial)

    out = {
        "loglikelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
    }
    return params | kf | out


def _collapse(data: np.ndarray, Z: np.ndarray, H: np.ndarray):
    Hinv = np.linalg.inv(H)
    ZtHinvZ = Z.T @ Hinv @ Z
    Hstar = np.linalg.inv(ZtHinvZ)
    Astar = Hstar @ Z.T @ Hinv
    ystar = data @ Astar.T
    return ystar, Hstar

def _loglikelihood_correction(data: np.ndarray, ystar: np.ndarray, Z: np.ndarray, H: np.ndarray, Hstar: np.ndarray) -> float:
    n = data.shape[0]
    Hinv = np.linalg.inv(H)
    _, logdet_H = np.linalg.slogdet(H)
    _, logdet_Hstar = np.linalg.slogdet(Hstar)
    et = data - ystar @ Z.T
    quad = np.sum(np.einsum("ti,ij,tj->t", et, Hinv, et))
    return -n / 2 * (logdet_H - logdet_Hstar) - 0.5 * quad

def _fit_collapsed(
    data: np.ndarray,
    initial_guess: dict,
    carry_initial: tuple,
    _dynamics: callable,
    _link: callable = lambda x: x,
    _invlink: callable = lambda x: x,
    opt_options: dict = {},
    maxiter: int = 5000) -> dict:
    maxiter = int(maxiter)
    opt_options = opt_options or {}

    def _criterion(unc_params):
        constr = _link(unc_params)
        Z = constr["Lambda"]
        H = constr["Sigma_eps"]
        ystar, Hstar = _collapse(data, Z, H)
        kf = _filter_light(ystar, _dynamics, constr | {"Hstar": Hstar}, carry_initial)
        ll_star = _loglikelihood(kf)
        correction = _loglikelihood_correction(data, ystar, Z, H, Hstar)
        return -(ll_star + correction)

    unc_params0 = np.asarray(_invlink(initial_guess))
    unc_params, niter, final_loss, is_converged = lbfgs(_criterion, unc_params0, opt_options, maxiter)

    constr = _link(unc_params)
    Z = constr["Lambda"]
    H = constr["Sigma_eps"]
    ystar, Hstar = _collapse(data, Z, H)
    kf = _filter(ystar, _dynamics, constr | {"Hstar": Hstar}, carry_initial)

    out = {
        "loglikelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
        "ystar": ystar,
    }
    return constr | kf | out


def _simulation(fit_output: dict, nsim: int, dynamics: callable, npaths: int, key: jax.Array):
    Qt, Ht = fit_output["Q"][-1], fit_output["H"][-1]
    at, Pt = fit_output["a"][-1], fit_output["P"][-1]
    Zt, Tt = fit_output["Z"][-1], fit_output["T"][-1]
    Rt = fit_output["R"][-1]
    q = Qt.shape[0]
    p = Ht.shape[0]
    k1, k2 = jax.random.split(key, 2)
    eta_draws = jax.random.multivariate_normal(k1, np.zeros(q), Qt, shape=(nsim, npaths))
    eps_draws = jax.random.multivariate_normal(k2, np.zeros(p), Ht, shape=(nsim, npaths))
    at = np.broadcast_to(at, (npaths,) + at.shape)
    carry0 = (at, Pt, Zt, Tt, Ht, Rt, Qt, 0)
    def _sim_step(carry, noise_t):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry
        eta_t, eps_t = noise_t

        y_dummy = np.zeros((p,), dtype=float)
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(y_dummy, at[0], Pt, fit_output, Zt, Tt, Ht, Rt, Qt, idx)

        dt = np.asarray(dt, dtype=float)
        ct = np.asarray(ct, dtype=float)

        y_t = np.einsum("ij,pj->pi", Zt, at) + dt + eps_t
        atp1 = np.einsum("ij,pj->pi", Tt, at) + ct + np.einsum("ij,pj->pi", Rt, eta_t)

        Ptp1 = Tt @ Pt @ Tt.T + Rt @ Qt @ Rt.T
        idx = idx + 1

        return (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx), y_t

    _, y_sim = lax.scan(_sim_step, carry0, (eta_draws, eps_draws), length=nsim)

    return {"y": y_sim, "eta": eta_draws, "eps": eps_draws}

_simulation = jax.jit(_simulation, static_argnames=("nsim", "npaths", "dynamics"))
"""Implementation of the Kalman filter.
Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _filter(data, dynamics, params, carry0):
    at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry0
    T = len(data)
    p, m = Zt.shape

    logdetF = np.zeros(T)
    quad = np.zeros(T)
    a_out = np.empty((T, m))
    P_out = np.empty((T, m, m))
    att_out = np.empty((T, m))
    Ptt_out = np.empty((T, m, m))
    v_out = np.zeros((T, p))
    F_out = np.empty((T, p, p))
    K_out = np.zeros((T, m, p))

    for t, yt in enumerate(data):
        missing = np.any(np.isnan(yt))
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)

        a_out[t] = at
        P_out[t] = Pt

        if missing:
            att_out[t] = at
            Ptt_out[t] = Pt
            F_out[t] = Ht
            atp1 = Tt @ at + ct
            Ptp1 = Tt @ Pt @ Tt.T + Rt @ Qt @ Rt.T
        else:
            ZP = Zt @ Pt
            Ft = ZP @ Zt.T + Ht
            cho = cho_factor(Ft, lower=True)
            Kt = cho_solve(cho, ZP).T
            vt = yt - Zt @ at - dt
            att = at + Kt @ vt
            Ptt = Pt - Kt @ ZP
            atp1 = Tt @ att + ct
            Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T
            logdetF[t] = 2.0 * np.sum(np.log(np.diag(cho[0])))
            quad[t] = vt @ cho_solve(cho, vt)
            att_out[t] = att
            Ptt_out[t] = Ptt
            v_out[t] = vt
            F_out[t] = Ft
            K_out[t] = Kt

        at, Pt = atp1, Ptp1
        idx += 1

    return {
        "logdetF": logdetF, "quad": quad,
        "a": a_out, "P": P_out,
        "att": att_out, "Ptt": Ptt_out,
        "v": v_out, "F": F_out, "K": K_out,
    }


def _filter_light(data, dynamics, params, carry0):
    at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry0
    T = len(data)

    logdetF = np.zeros(T)
    quad = np.zeros(T)

    for t, yt in enumerate(data):
        missing = np.any(np.isnan(yt))
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)

        if missing:
            atp1 = Tt @ at + ct
            Ptp1 = Tt @ Pt @ Tt.T + Rt @ Qt @ Rt.T
        else:
            ZP = Zt @ Pt
            Ft = ZP @ Zt.T + Ht
            cho = cho_factor(Ft, lower=True)
            Kt = cho_solve(cho, ZP).T
            vt = yt - Zt @ at - dt
            att = at + Kt @ vt
            Ptt = Pt - Kt @ ZP
            atp1 = Tt @ att + ct
            Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T
            logdetF[t] = 2.0 * np.sum(np.log(np.diag(cho[0])))
            quad[t] = vt @ cho_solve(cho, vt)

        at, Pt = atp1, Ptp1
        idx += 1

    return {"logdetF": logdetF, "quad": quad}


def _filter_light_univariate(data, dynamics, params, carry0):
    at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry0
    T = len(data)

    logdetF = np.zeros(T)
    quad = np.zeros(T)

    for t, yt in enumerate(data):
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)
        H_diag = np.diag(Ht)
        dt_arr = np.zeros(Zt.shape[0], dtype=float) + np.asarray(dt, dtype=float)

        logdet_acc = 0.0
        quad_acc = 0.0
        for j in range(len(yt)):
            if np.isnan(yt[j]):
                continue
            Z_j = Zt[j]
            PZ = Pt @ Z_j
            F_j = Z_j @ PZ + H_diag[j]
            v_j = yt[j] - Z_j @ at - dt_arr[j]
            at = at + PZ * (v_j / F_j)
            Pt = Pt - np.outer(PZ, PZ) / F_j
            logdet_acc += np.log(F_j)
            quad_acc += v_j * v_j / F_j

        logdetF[t] = logdet_acc
        quad[t] = quad_acc
        at = Tt @ at + ct
        Pt = Tt @ Pt @ Tt.T + Rt @ Qt @ Rt.T
        idx += 1

    return {"logdetF": logdetF, "quad": quad}


def _loglikelihood(filter_output):
    return -0.5 * np.sum(filter_output["logdetF"] + filter_output["quad"])


def _fit(
    data,
    initial_guess,
    covariates,
    carry_initial,
    _dynamics,
    _link=lambda x: x,
    _invlink=lambda x: x,
    opt_options=None,
    maxiter=5000,
    extra_loglikelihood_fn=lambda _, __: 0.0,
    extra_ll_data=None,
    _filter_fn=_filter_light,
):
    maxiter = int(maxiter)
    opt_options = opt_options or {}

    def _criterion(params):
        constr_params = _link(params)
        kf = _filter_fn(data, _dynamics, constr_params | {"covariates": covariates}, carry_initial)
        return -(_loglikelihood(kf) + extra_loglikelihood_fn(constr_params, extra_ll_data))

    unc_params0 = np.asarray(_invlink(dict(initial_guess) | {"covariates": covariates}))
    result = minimize(_criterion, unc_params0, method="L-BFGS-B", options={"maxiter": maxiter, **opt_options})
    unc_params, niter, final_loss, is_converged = result.x, result.nit, result.fun, result.success

    params = dict(_link(unc_params))
    params["covariates"] = covariates
    kf = _filter(data, _dynamics, params, carry_initial)

    return params | kf | {
        "loglikelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
    }


def _collapse(data, Z, H):
    Hinv = np.linalg.inv(H)
    ZtHinvZ = Z.T @ Hinv @ Z
    Hstar = np.linalg.inv(ZtHinvZ)
    ystar = data @ (Hstar @ Z.T @ Hinv).T
    return ystar, Hstar, Hinv


def _loglikelihood_correction(data, ystar, Z, H, Hstar, Hinv):
    n = data.shape[0]
    _, logdet_H = np.linalg.slogdet(H)
    _, logdet_Hstar = np.linalg.slogdet(Hstar)
    et = data - ystar @ Z.T
    quad = np.sum(np.einsum("ti,ij,tj->t", et, Hinv, et))
    return -n / 2 * (logdet_H - logdet_Hstar) - 0.5 * quad


def _fit_collapsed(
    data,
    initial_guess,
    carry_initial,
    _dynamics,
    _link=lambda x: x,
    _invlink=lambda x: x,
    maxiter=5000,
):
    maxiter = int(maxiter)

    def _criterion(unc_params):
        constr = _link(unc_params)
        Z, H = constr["Lambda"], constr["Sigma_eps"]
        ystar, Hstar, Hinv = _collapse(data, Z, H)
        kf = _filter_light(ystar, _dynamics, constr | {"Hstar": Hstar}, carry_initial)
        return -(_loglikelihood(kf) + _loglikelihood_correction(data, ystar, Z, H, Hstar, Hinv))

    unc_params0 = np.asarray(_invlink(initial_guess))
    result = minimize(_criterion, unc_params0, method="L-BFGS-B", options={"maxiter": maxiter})
    unc_params, niter, final_loss, is_converged = result.x, result.nit, result.fun, result.success

    constr = _link(unc_params)
    Z, H = constr["Lambda"], constr["Sigma_eps"]
    ystar, Hstar, _ = _collapse(data, Z, H)
    kf = _filter(ystar, _dynamics, constr | {"Hstar": Hstar}, carry_initial)

    return constr | kf | {
        "loglikelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
        "ystar": ystar,
    }


def _simulation(fit_output, nsim, dynamics, npaths, rng):
    Qt, Ht = fit_output["Q"][-1], fit_output["H"][-1]
    at, Pt = fit_output["a"][-1], fit_output["P"][-1]
    Zt, Tt = fit_output["Z"][-1], fit_output["T"][-1]
    Rt = fit_output["R"][-1]
    q, p = Qt.shape[0], Ht.shape[0]
    eta_draws = rng.multivariate_normal(np.zeros(q), Qt, size=(nsim, npaths))
    eps_draws = rng.multivariate_normal(np.zeros(p), Ht, size=(nsim, npaths))
    at = np.broadcast_to(at, (npaths,) + at.shape).copy()

    y_sim = np.empty((nsim, npaths, p))
    carry = (at, Pt, Zt, Tt, Ht, Rt, Qt, 0)

    for t in range(nsim):
        at_c, Pt_c, Zt_c, Tt_c, Ht_c, Rt_c, Qt_c, idx = carry
        Zt_c, Tt_c, Ht_c, Rt_c, Qt_c, dt, ct = dynamics(
            np.zeros(p), at_c[0], Pt_c, fit_output, Zt_c, Tt_c, Ht_c, Rt_c, Qt_c, idx
        )
        dt = np.asarray(dt, dtype=float)
        ct = np.asarray(ct, dtype=float)
        y_sim[t] = at_c @ Zt_c.T + dt + eps_draws[t]
        atp1 = at_c @ Tt_c.T + ct + eta_draws[t] @ Rt_c.T
        Ptp1 = Tt_c @ Pt_c @ Tt_c.T + Rt_c @ Qt_c @ Rt_c.T
        carry = (atp1, Ptp1, Zt_c, Tt_c, Ht_c, Rt_c, Qt_c, idx + 1)

    return {"y": y_sim, "eta": eta_draws, "eps": eps_draws}

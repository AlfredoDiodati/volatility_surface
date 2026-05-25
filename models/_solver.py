import jax
import jax.numpy as jnp
from jax import lax

def adam(criterion, theta0, opt_options=None, maxiter=5000):
    opt_options = opt_options or {}
    lr = opt_options.get("learning_rate", 1e-2)
    tol = opt_options.get("tol", 1e-6)
    b1 = opt_options.get("beta1", 0.9)
    b2 = opt_options.get("beta2", 0.999)
    eps = opt_options.get("eps", 1e-8)
    maxiter = int(maxiter)
    theta0 = jnp.asarray(theta0, dtype=jnp.float32)
    value_and_grad = jax.value_and_grad(criterion)

    def _step(state):
        theta, m, v, b1t, b2t, i, curr_loss, best_theta, best_loss, converged = state
        loss, g = value_and_grad(theta)
        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * g * g
        b1t_new = b1t * b1
        b2t_new = b2t * b2
        mhat = m_new / (1.0 - b1t_new)
        vhat = v_new / (1.0 - b2t_new)
        step = mhat / (jnp.sqrt(vhat) + eps)
        theta_new = theta - lr * step
        best_theta_new = jnp.where(loss < best_loss, theta, best_theta)
        best_loss_new = jnp.minimum(loss, best_loss)
        converged_new = jnp.linalg.norm(step) < tol
        return (theta_new, m_new, v_new, b1t_new, b2t_new, i + 1, loss, best_theta_new, best_loss_new, converged_new)

    def _not_converged(state):
        _, _, _, _, _, i, curr_loss, _, _, converged = state
        return (i < maxiter) & ~converged & jnp.isfinite(curr_loss)

    state0 = (
        theta0,
        jnp.zeros_like(theta0),
        jnp.zeros_like(theta0),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32),
        theta0,
        jnp.asarray(jnp.finfo(jnp.float32).max, dtype=jnp.float32),
        jnp.asarray(False),
    )

    _, _, _, _, _, niter, _, best_theta, best_loss, is_converged = lax.while_loop(
        _not_converged, _step, state0
    )
    return best_theta, niter, best_loss, is_converged

def adam_adj(criterion, theta0, opt_options=None, maxiter=5000):
    opt_options = opt_options or {}
    lr = opt_options.get("learning_rate", 1e-2)
    tol = opt_options.get("tol", 1e-6)
    b1 = opt_options.get("beta1", 0.9)
    b2 = opt_options.get("beta2", 0.999)
    eps = opt_options.get("eps", 1e-8)
    maxiter = int(maxiter)

    theta0 = jnp.asarray(theta0)
    theta0 = theta0.astype(jnp.result_type(theta0.dtype, jnp.float32))
    float_info = jnp.finfo(theta0.dtype)
    value_and_grad = jax.value_and_grad(criterion)

    def _step(state):
        theta, m, v, b1t, b2t, i, loss_finite, best_theta, best_loss, converged = state
        loss, g = value_and_grad(theta)

        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * g * g
        b1t_new = b1t * b1
        b2t_new = b2t * b2

        mhat = m_new / (1.0 - b1t_new)
        vhat = v_new / (1.0 - b2t_new)

        i1 = i + 1
        lr_t = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * i1 / maxiter))
        theta_new = theta - lr_t * mhat / (jnp.sqrt(vhat) + eps)

        best_theta_new = jnp.where(loss < best_loss, theta, best_theta)
        best_loss_new = jnp.minimum(loss, best_loss)
        converged_new = jnp.linalg.norm(g) / jnp.sqrt(g.size) < tol

        return (theta_new, m_new, v_new, b1t_new, b2t_new, i1, jnp.isfinite(loss), best_theta_new, best_loss_new, converged_new)

    def _not_converged(state):
        _, _, _, _, _, i, loss_finite, _, _, converged = state
        return (i < maxiter) & ~converged & loss_finite

    state0 = (
        theta0,
        jnp.zeros_like(theta0),
        jnp.zeros_like(theta0),
        jnp.asarray(1.0, dtype=theta0.dtype),
        jnp.asarray(1.0, dtype=theta0.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
        theta0,
        jnp.asarray(float_info.max, dtype=theta0.dtype),
        jnp.asarray(False),
    )

    _, _, _, _, _, niter, _, best_theta, best_loss, is_converged = lax.while_loop(
        _not_converged, _step, state0
    )

    return best_theta, niter, best_loss, is_converged


def sgn(criterion, theta0, opt_options=None, maxiter=5000):
    opt_options = opt_options or {}
    gamma = opt_options.get("learning_rate", 0.1)
    eps = opt_options.get("eps", 0.1)
    alpha_mom = opt_options.get("momentum", 0.47)
    tol = opt_options.get("tol", 1e-6)
    lm_scale = opt_options.get("lm_scale", 1e-3)
    seed = int(opt_options.get("seed", 0))
    maxiter = int(maxiter)

    theta0 = jnp.asarray(theta0)
    theta0 = theta0.astype(jnp.result_type(theta0.dtype, jnp.float32))

    p = theta0.shape[0]
    float_info = jnp.finfo(theta0.dtype)
    L = int(opt_options.get("L", max(25, int(1.5 * p))))

    value_and_grad_fn = jax.value_and_grad(criterion)

    key = jax.random.PRNGKey(seed)
    all_Z = jax.random.normal(key, shape=(maxiter + L, p), dtype=theta0.dtype)

    loss0, g0 = value_and_grad_fn(theta0)
    Z_init = all_Z[:L]

    def _perturb_init(z):
        _, gp = value_and_grad_fn(theta0 + eps * z)
        return (gp - g0) / eps

    Y_init = jax.vmap(_perturb_init)(Z_init)

    _I = jnp.eye(p, dtype=theta0.dtype)

    def _fit_G(Zb, Yb):
        Zm = Zb - Zb.mean(axis=0)
        return jnp.linalg.solve(Zm.T @ Zm + 1e-10 * _I, Zm.T @ Yb).T

    Gb_init = _fit_G(Z_init, Y_init)

    state0 = (
        theta0,
        theta0,
        Z_init,
        Y_init,
        Gb_init,
        g0,
        jnp.asarray(0, dtype=jnp.int32),
        theta0,
        jnp.asarray(float_info.max, dtype=theta0.dtype),
        jnp.asarray(True),
        jnp.asarray(False),
    )

    def _step(state):
        theta, theta_prev, Zb, Yb, Gb, g, i, best_theta, best_loss, loss_finite, converged = state

        Hb = Gb.T @ Gb
        Zm = Zb - Zb.mean(axis=0)
        rmse = jnp.sqrt(jnp.mean((Yb - Zm @ Gb.T) ** 2))
        pen = lm_scale * rmse * jnp.sqrt(jnp.sum(Gb ** 2)) * jnp.sqrt(jnp.sum(Hb ** 2))

        step_dir = jnp.linalg.solve(Hb + pen * _I, Gb.T @ g)
        theta_new = theta - gamma * step_dir + alpha_mom * (theta - theta_prev)

        loss_new, g_new = value_and_grad_fn(theta_new)
        loss_finite_new = jnp.isfinite(loss_new) & jnp.all(jnp.isfinite(g_new))

        z_next = all_Z[L + i]
        _, g_pert = value_and_grad_fn(theta_new + eps * z_next)
        y_next = (g_pert - g_new) / eps

        Zb_new = jnp.concatenate([Zb[1:], z_next[None]], axis=0)
        Yb_new = jnp.concatenate([Yb[1:], y_next[None]], axis=0)
        Gb_new = _fit_G(Zb_new, Yb_new)

        update_best = loss_finite_new & (loss_new < best_loss)
        best_theta_new = jnp.where(update_best, theta_new, best_theta)
        best_loss_new = jnp.where(update_best, loss_new, best_loss)
        converged_new = jnp.linalg.norm(g_new) / jnp.sqrt(p) < tol

        return (
            theta_new, theta, Zb_new, Yb_new, Gb_new, g_new, i + 1,
            best_theta_new, best_loss_new, loss_finite_new, converged_new,
        )

    def _not_converged(state):
        _, _, _, _, _, _, i, _, _, loss_finite, converged = state
        return (i < maxiter) & ~converged & loss_finite

    _, _, _, _, _, _, niter, best_theta, best_loss, _, is_converged = lax.while_loop(
        _not_converged, _step, state0
    )

    return best_theta, niter, best_loss, is_converged

def lbfgs(criterion, theta0, opt_options=None, maxiter=5000):
    opt_options = opt_options or {}
    tol = opt_options.get("tol", 1e-6)
    memory = int(opt_options.get("memory", 10))
    lr = opt_options.get("learning_rate", 1.0)
    c1 = opt_options.get("c1", 1e-4)
    maxls = int(opt_options.get("maxls", 20))
    maxiter = int(maxiter)

    theta0 = jnp.asarray(theta0)
    theta0 = theta0.astype(jnp.result_type(theta0.dtype, jnp.float32))

    p = theta0.shape[0]
    float_info = jnp.finfo(theta0.dtype)
    value_and_grad_fn = jax.value_and_grad(criterion)

    def two_loop_recursion(g, s_hist, y_hist, rho_hist, write_idx):
        indices = (write_idx + jnp.arange(memory)) % memory

        def backward_body(q, k):
            idx = indices[memory - 1 - k]
            alpha_i = rho_hist[idx] * jnp.dot(s_hist[idx], q)
            return q - alpha_i * y_hist[idx], alpha_i

        q, alphas = lax.scan(backward_body, g, jnp.arange(memory))

        newest = indices[memory - 1]
        sy = jnp.dot(s_hist[newest], y_hist[newest])
        yy = jnp.dot(y_hist[newest], y_hist[newest])
        gamma = jnp.where(yy > 1e-10, sy / yy, jnp.ones((), dtype=theta0.dtype))
        r = gamma * q

        def forward_body(r, k):
            idx = indices[k]
            beta_i = rho_hist[idx] * jnp.dot(y_hist[idx], r)
            return r + s_hist[idx] * (alphas[memory - 1 - k] - beta_i), None

        r, _ = lax.scan(forward_body, r, jnp.arange(memory))
        return r

    def backtracking_line_search(theta, f0, g, direction):
        slope = jnp.dot(g, direction)
        lr_typed = jnp.asarray(lr, theta0.dtype)
        f_init = criterion(theta + lr_typed * direction)

        def ls_cond(state):
            alpha, f_trial, i = state
            return (i < maxls) & (~jnp.isfinite(f_trial) | (f_trial > f0 + c1 * alpha * slope))

        def ls_body(state):
            alpha, _, i = state
            alpha_new = alpha * jnp.asarray(0.5, theta0.dtype)
            return alpha_new, criterion(theta + alpha_new * direction), i + 1

        alpha_final, _, _ = lax.while_loop(
            ls_cond, ls_body,
            (lr_typed, f_init, jnp.asarray(0, jnp.int32))
        )
        return alpha_final

    def _step(state):
        theta, g, f, s_hist, y_hist, rho_hist, write_idx, i, best_theta, best_loss, loss_finite, converged = state

        direction = -two_loop_recursion(g, s_hist, y_hist, rho_hist, write_idx)
        alpha = backtracking_line_search(theta, f, g, direction)

        theta_new = theta + alpha * direction
        f_new, g_new = value_and_grad_fn(theta_new)

        s_new = theta_new - theta
        y_new = g_new - g
        ys = jnp.dot(y_new, s_new)
        rho_new = jnp.where(ys > 1e-10, 1.0 / ys, jnp.zeros((), dtype=theta0.dtype))

        s_hist_new = s_hist.at[write_idx].set(s_new)
        y_hist_new = y_hist.at[write_idx].set(y_new)
        rho_hist_new = rho_hist.at[write_idx].set(rho_new)
        write_idx_new = (write_idx + 1) % memory

        loss_finite_new = jnp.isfinite(f_new) & jnp.all(jnp.isfinite(g_new))
        update_best = loss_finite_new & (f_new < best_loss)
        best_theta_new = jnp.where(update_best, theta_new, best_theta)
        best_loss_new = jnp.where(update_best, f_new, best_loss)
        converged_new = jnp.linalg.norm(g_new) / jnp.sqrt(p) < tol

        return (
            theta_new, g_new, f_new, s_hist_new, y_hist_new, rho_hist_new, write_idx_new, i + 1,
            best_theta_new, best_loss_new, loss_finite_new, converged_new,
        )

    def _not_converged(state):
        _, _, _, _, _, _, _, i, _, _, loss_finite, converged = state
        return (i < maxiter) & ~converged & loss_finite

    f0, g0 = value_and_grad_fn(theta0)

    state0 = (
        theta0,
        g0,
        f0,
        jnp.zeros((memory, p), dtype=theta0.dtype),
        jnp.zeros((memory, p), dtype=theta0.dtype),
        jnp.zeros(memory, dtype=theta0.dtype),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        theta0,
        jnp.asarray(float_info.max, dtype=theta0.dtype),
        jnp.isfinite(f0),
        jnp.asarray(False),
    )

    _, _, _, _, _, _, _, niter, best_theta, best_loss, _, is_converged = lax.while_loop(
        _not_converged, _step, state0
    )

    return best_theta, niter, best_loss, is_converged
import jax
import jax.numpy as jnp
from jax import lax

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
    tol_sq = jnp.asarray(tol * tol * p, dtype=theta0.dtype)
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
        gamma = jnp.where(yy > 0, sy / yy, jnp.ones((), dtype=theta0.dtype))
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
        yy = jnp.dot(y_new, y_new)
        ss = jnp.dot(s_new, s_new)
        ys_min = float_info.eps * jnp.sqrt(yy * ss)
        rho_new = jnp.where(ys > ys_min, 1.0 / ys, jnp.zeros((), dtype=theta0.dtype))

        s_hist_new = s_hist.at[write_idx].set(s_new)
        y_hist_new = y_hist.at[write_idx].set(y_new)
        rho_hist_new = rho_hist.at[write_idx].set(rho_new)
        write_idx_new = (write_idx + 1) % memory

        g_sq = jnp.dot(g_new, g_new)
        loss_finite_new = jnp.isfinite(f_new) & jnp.isfinite(g_sq)
        update_best = loss_finite_new & (f_new < best_loss)
        best_theta_new = jnp.where(update_best, theta_new, best_theta)
        best_loss_new = jnp.where(update_best, f_new, best_loss)
        converged_new = g_sq < tol_sq

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
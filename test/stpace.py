import jax
import numpy as onp
import jax.numpy as np
import models.ss as ss

key = jax.random.PRNGKey(20260227)

horizon = 3500

moneyness_grid = np.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5], dtype=float)
time_to_maturity_grid = np.array([10.0, 50.0, 100.0, 180.0], dtype=float) / 255.0
moneyness_mesh, time_to_maturity_mesh = np.meshgrid(moneyness_grid, time_to_maturity_grid, indexing="xy")
moneyness_vector = moneyness_mesh.reshape(-1)
time_to_maturity_vector = time_to_maturity_mesh.reshape(-1)
Mt = np.column_stack([np.ones(moneyness_vector.shape[0], dtype=float), moneyness_vector, time_to_maturity_vector])

pH = Mt.shape[0]
p = Mt.shape[1]

B_true = np.diag(np.array([0.98, 0.93, 0.90], dtype=float))
Q_true = np.diag(np.array([0.001, 0.005, 0.004], dtype=float))
sigma2_eps_true = 0.05
H_true = sigma2_eps_true * np.eye(pH, dtype=float)

key, key_bar = jax.random.split(key, 2)
bar_beta_true = jax.random.uniform(key_bar, shape=(p,), minval=0.0, maxval=1.0)
ct_true = (np.eye(p, dtype=float) - B_true) @ bar_beta_true

covariates = np.broadcast_to(Mt[None, :, :], (horizon, pH, p))

a1 = bar_beta_true
P1 = 1e2 * np.eye(p, dtype=float)
Z1 = np.asarray(Mt, dtype=float)
T1 = np.eye(p, dtype=float)
H1 = H_true
R1 = np.eye(p, dtype=float)
Q1 = Q_true
initialization = (a1, P1, Z1, T1, H1, R1, Q1, 0)

fit_output_true = {
    "Q_param": Q_true,
    "H_param": H_true,
    "B": B_true,
    "bar_beta": bar_beta_true,
    "ct": ct_true,
    "covariates": covariates,
    "a": np.asarray([a1]),
    "P": np.asarray([P1]),
    "Z": np.asarray([Z1]),
    "T": np.asarray([T1]),
    "H": np.asarray([H1]),
    "R": np.asarray([R1]),
    "Q": np.asarray([Q1]),
}

key, key_sim = jax.random.split(key, 2)
sim = ss.simulation(fit_output_true, nsim=horizon, npaths=1, key=key_sim)
y = sim["y"][:, 0, :]

initial_guess = {
    "Q_param": 1e-3 * np.eye(p, dtype=float),
    "H_param": 5e-2 * np.eye(pH, dtype=float),
    "B": 0.9 * np.eye(p, dtype=float),
    "bar_beta": 0.5 * np.ones((p,), dtype=float),
}

opt_options = {"maxiter": 400, "learning_rate": 1e-2, "tol": 1e-6}

fitted = ss.fit_collapsed(y, covariates, initial_guess, initialization, opt_options=opt_options)

bar_beta_hat = onp.asarray(jax.device_get(fitted["bar_beta"]))
B_hat = onp.asarray(jax.device_get(fitted["B"]))
Q_hat = onp.asarray(jax.device_get(fitted["Q_param"]))
H_hat = onp.asarray(jax.device_get(fitted["H_param"]))

bar_beta_true_ = onp.asarray(jax.device_get(bar_beta_true))
B_true_ = onp.asarray(jax.device_get(B_true))
Q_true_ = onp.asarray(jax.device_get(Q_true))

print("param        true                              estimated                         abs_err      rel_err")
print("-" * 110)

abs_err = float(onp.linalg.norm(bar_beta_hat - bar_beta_true_))
rel_err = abs_err / (float(onp.linalg.norm(bar_beta_true_)) + 1e-18)
print(f"{'bar_beta':<12} {bar_beta_true_} {bar_beta_hat} {abs_err:>10.6g} {rel_err:>10.6g}")

abs_err = float(onp.linalg.norm(onp.diag(B_hat) - onp.diag(B_true_)))
rel_err = abs_err / (float(onp.linalg.norm(onp.diag(B_true_))) + 1e-18)
print(f"{'diag(B)':<12} {onp.diag(B_true_)} {onp.diag(B_hat)} {abs_err:>10.6g} {rel_err:>10.6g}")

abs_err = float(onp.linalg.norm(onp.diag(Q_hat) - onp.diag(Q_true_)))
rel_err = abs_err / (float(onp.linalg.norm(onp.diag(Q_true_))) + 1e-18)
print(f"{'diag(Q)':<12} {onp.diag(Q_true_)} {onp.diag(Q_hat)} {abs_err:>10.6g} {rel_err:>10.6g}")

abs_err = float(abs(H_hat[0, 0] - sigma2_eps_true))
rel_err = abs_err / (sigma2_eps_true + 1e-18)
print(f"{'sigma2_eps':<12} {sigma2_eps_true} {H_hat[0, 0]} {abs_err:>10.6g} {rel_err:>10.6g}")
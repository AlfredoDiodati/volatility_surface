import os
import math
import numpy as np
import matplotlib.pyplot as plt

from holder_est.scaling_reg import moment_scaling


def gaussian_abs_moment(q: float) -> float:
    return (2.0 ** (q / 2.0)) * math.gamma((q + 1.0) / 2.0) / math.sqrt(math.pi)


def theoretical_log_S(phi: float, sigma: float, dt: np.ndarray, q: float, path_length: int) -> np.ndarray:
    stationary_variance = (sigma * sigma) / (1.0 - phi * phi)
    increment_variance = 2.0 * stationary_variance * (1.0 - (phi ** dt.astype(float)))
    n_blocks = np.floor((float(path_length) - 1.0) / dt.astype(float))
    n_blocks = np.maximum(1.0, n_blocks)
    return np.log(n_blocks) + np.log(gaussian_abs_moment(q)) + (q / 2.0) * np.log(increment_variance)


def simulate_ou_path(phi: float, path_length: int, x0: float = 0.0, sigma: float = 1.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(path_length, dtype=float)
    x[0] = x0
    eps = rng.standard_normal(path_length - 1)
    for t in range(1, path_length):
        x[t] = x0 + phi * (x[t - 1] - x0) + sigma * eps[t - 1]
    return x


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = x.astype(float)
    y = y.astype(float)
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    slope = float(np.sum(x_centered * y_centered) / np.sum(x_centered ** 2))
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def main() -> None:
    path_length = 3500
    moments = np.arange(1, 9) / 2
    minf = 1.0
    maxf = float(path_length)
    factor = 1.1

    tick_days = np.array([1, 5, 21, 63, 116])
    tick_labels = ["", "1 week", "1 month", "3 months", "6 months"]

    phi_grid = np.array([0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99], dtype=float)
    sigma = 1.0

    npaths = 500
    base_seed = 12345

    plots_dir = os.path.join("plot", "test")
    os.makedirs(plots_dir, exist_ok=True)

    global_y_min = np.inf
    global_y_max = -np.inf
    panel_payload = []

    tau_payload = []

    for phi in phi_grid:
        x0 = simulate_ou_path(phi=phi, path_length=path_length, x0=0.0, sigma=sigma, seed=base_seed)
        scaling_0 = moment_scaling(x0, minf, maxf, moments, factor=factor)

        dt_full = scaling_0["delta_ts"].astype(int)
        log_dt_full = np.log(dt_full.astype(float))

        in_view = (dt_full >= 1) & (dt_full <= 126)
        dt = dt_full[in_view]

        sum_power_var = np.zeros((moments.shape[0], dt_full.shape[0]), dtype=float)

        for k in range(npaths):
            x = simulate_ou_path(phi=phi, path_length=path_length, x0=0.0, sigma=sigma, seed=base_seed + k)
            scaling_k = moment_scaling(x, minf, maxf, moments, factor=factor)
            for qi, q in enumerate(moments):
                sum_power_var[qi] += np.exp(scaling_k[q]["log_power_var"].astype(float))

        mean_power_var = sum_power_var / float(npaths)
        avg_log_power_var = np.log(np.maximum(mean_power_var, 1e-300))

        curves = []
        theoretical_curves = []
        tau_emp = np.empty(moments.shape[0], dtype=float)
        tau_theory = np.empty(moments.shape[0], dtype=float)

        for qi, q in enumerate(moments):
            holder_emp, intercept_emp = _ols_slope_intercept(log_dt_full, avg_log_power_var[qi])
            tau_emp[qi] = holder_emp
            y = (avg_log_power_var[qi] - intercept_emp)[in_view].astype(float)
            y = y - float(y[0])

            y_theory_full = theoretical_log_S(phi=phi, sigma=sigma, dt=dt_full, q=float(q), path_length=path_length).astype(float)
            holder_theory, intercept_theory = _ols_slope_intercept(log_dt_full, y_theory_full)
            tau_theory[qi] = holder_theory
            y_theory = (y_theory_full - intercept_theory)[in_view].astype(float)
            y_theory = y_theory - float(y_theory[0])

            global_y_min = min(global_y_min, float(np.min(y)))
            global_y_max = max(global_y_max, float(np.max(y)))

            curves.append((float(q), y))
            theoretical_curves.append((float(q), y_theory))

        tau_payload.append((float(phi), tau_emp, tau_theory))
        panel_payload.append((float(phi), dt.astype(int), curves, theoretical_curves))

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, sharey=True)
    axes = axes.ravel()

    for panel_idx, (phi, dt, curves, theoretical_curves) in enumerate(panel_payload):
        ax = axes[panel_idx]

        for x_tick in tick_days:
            ax.axvline(x_tick, linestyle="--", linewidth=0.8)

        ax.set_xticks(tick_days)
        ax.set_xticklabels(tick_labels, ha="center")

        for q, y in curves:
            line, = ax.plot(dt, y)
            for q_theory, y_theory in theoretical_curves:
                if q_theory == q:
                    ax.plot(dt, y_theory, linestyle=":", linewidth=1.0)
                    break
            ax.text(
                float(dt[-1]) * 1.01,
                float(y[-1]),
                f"q={q:g}",
                va="center",
                fontsize=8,
                clip_on=False,
            )

        ax.set_xlim((1, 126))
        ax.set_ylim((global_y_min, global_y_max))
        ax.set_title(f"phi={phi:.2f}")
        ax.tick_params(axis="x", labelrotation=0, pad=8)

    for ax in axes[::3]:
        ax.set_ylabel(r"$S(q, \Delta t)$")
    for ax in axes[-3:]:
        ax.set_xlabel(r"$\Delta t$")

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.10, top=0.93, wspace=0.28, hspace=0.38)
    fig.savefig(os.path.join(plots_dir, "scaling_ou_3x3_panel_mc.pdf"))
    plt.close(fig)

    holder_bm = moments / 2.0 - 1.0

    fig2, ax2 = plt.subplots()
    ax2.plot(moments, holder_bm, linestyle="--", color="black")

    for phi, tau_emp, tau_theory in tau_payload:
        line, = ax2.plot(moments, tau_emp)
        ax2.scatter(moments, tau_theory, s=18, color=line.get_color())

    ax2.set_xlabel(r"$q$")
    ax2.set_ylabel(r"$\tau(q)$")
    ax2.set_xlim((float(moments[0]), float(moments[-1])))

    fig2.tight_layout()
    fig2.savefig(os.path.join(plots_dir, "OU_tau_mc.pdf"))
    plt.close(fig2)


if __name__ == "__main__":
    main()

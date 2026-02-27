import os
import numpy as np
import matplotlib.pyplot as plt

from holder_est.scaling_reg import moment_scaling


def simulate_ou_path(phi: float, path_length: int, x0: float = 0.0, sigma: float = 1.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(path_length, dtype=float)
    x[0] = x0
    eps = rng.standard_normal(path_length - 1)
    for t in range(1, path_length):
        x[t] = x0 + phi * (x[t - 1] - x0) + sigma * eps[t - 1]
    return x


def main() -> None:
    path_length = 3500
    moments = np.arange(1, 9) / 2
    minf = 1.0
    maxf = float(path_length)
    factor = 1.1

    tick_days = np.array([1, 5, 21, 63, 116])
    tick_labels = ["", "1 week", "1 month", "3 months", "6 months"]

    phi_grid = np.array([0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99], dtype=float)
    sigma_grid = np.array([1.0], dtype=float)

    plots_dir = os.path.join("plot", "test")
    os.makedirs(plots_dir, exist_ok=True)

    for sigma in sigma_grid:
        scalings = {}
        global_y_min = np.inf
        global_y_max = -np.inf

        panel_payload = []

        for phi in phi_grid:
            x = simulate_ou_path(phi=phi, path_length=path_length, x0=0.0, sigma=sigma, seed=12345)

            scaling_dict = moment_scaling(x, minf, maxf, moments, factor=factor)
            dt_full = scaling_dict["delta_ts"].astype(int)
            in_view = (dt_full >= 1) & (dt_full <= 126)
            dt = dt_full[in_view]

            holder = []
            curves = []
            for q in moments:
                holder.append(scaling_dict[q]["holder"])
                y_full = scaling_dict[q]["shifted_power_var"]
                y = y_full[in_view]

                global_y_min = min(global_y_min, float(np.min(y)))
                global_y_max = max(global_y_max, float(np.max(y)))

                curves.append((float(q), y.astype(float)))

            scalings[str(phi)] = np.array(holder, dtype=float)
            panel_payload.append((float(phi), dt.astype(int), curves))

        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, sharey=True)
        axes = axes.ravel()

        for panel_idx, (phi, dt, curves) in enumerate(panel_payload):
            ax = axes[panel_idx]

            for x_tick in tick_days:
                ax.axvline(x_tick, linestyle="--", linewidth=0.8, color="black")

            ax.set_xticks(tick_days)
            ax.set_xticklabels(tick_labels, ha="center")

            for q, y in curves:
                line, = ax.plot(dt, y)
                ax.text(
                    float(dt[-1]) * 1.01,
                    float(y[-1]),
                    f"q={q:g}",
                    color=line.get_color(),
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
        fig.savefig(os.path.join(plots_dir, "scaling_ou_3x3_panel.pdf"))
        plt.close(fig)

        holder_bm = moments / 2.0 - 1.0
        plt.figure()
        plt.plot(moments, holder_bm, color="black", linestyle="--")
        for label in scalings.keys():
            plt.plot(moments, scalings[label])
        plt.ylabel(r"$\tau(q)$")
        plt.xlabel(r"$q$")
        plt.xlim((moments[0], moments[-1]))
        plt.ylim((None, 1.0))
        plt.savefig(os.path.join(plots_dir, "OU_moments_scaling.pdf"))
        plt.close()


if __name__ == "__main__":
    main()
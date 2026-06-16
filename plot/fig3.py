import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

eta = 0.7
phi = 10.0
K_list = [1, 3, 5, 10, 20, 50]
tau = np.logspace(0, 6, 800)
target = tau ** (-eta)


def bc_parameters(K):
    rho = np.array([np.exp(-eta * phi ** (-i)) for i in range(K + 1)])
    c = np.zeros(K + 1)
    c[K] = 1.0
    for k in range(K - 1, -1, -1):
        i = K - k
        leakage = 0.0
        for j in range(i):
            gap = i - j
            leakage += (
                c[K - j]
                * phi ** (-eta * gap)
                * np.exp(eta * (1 - phi ** (-gap)))
            )
        c[k] = 1.0 - leakage
    w = np.array([c[i] * phi ** (-i * eta) * np.exp(eta) for i in range(K + 1)])
    return rho, w


def bc_autocovariance(tau, K):
    rho, w = bc_parameters(K)
    return np.sum(w[:, None] * rho[:, None] ** tau[None, :], axis=0)


def coupling_matrix(rho, w):
    one_minus = 1 - np.outer(rho, rho)
    valid = one_minus > 1e-12
    correlation = np.zeros_like(one_minus)
    numerator = np.sqrt(np.outer(1 - rho ** 2, 1 - rho ** 2))
    correlation[valid] = numerator[valid] / one_minus[valid]
    L = np.sqrt(np.outer(1.0 / w, w)) * correlation
    np.fill_diagonal(L, 0.0)
    return L


def solve_mu(rho, w, J):
    L = coupling_matrix(rho, w)
    mu = np.ones(len(w))
    for _ in range(J):
        s = L @ mu
        mu = (-s + np.sqrt(s ** 2 + 4.0)) / 2.0
    return mu


def sd_autocovariance(tau, K, J):
    rho, w = bc_parameters(K)
    chain_rule_loading = np.sqrt(w * (1 - rho ** 2))
    if J == 0:
        kappa = np.ones(K + 1)
    else:
        mu = solve_mu(rho, w, J)
        kappa = mu / w
    A = w * kappa * chain_rule_loading
    one_minus = 1 - np.outer(rho, rho)
    valid = one_minus > 1e-12
    resolvent = np.zeros_like(one_minus)
    resolvent[valid] = 1.0 / one_minus[valid]
    gamma = np.zeros(len(tau))
    for i in range(K + 1):
        for j in range(K + 1):
            gamma += A[i] * A[j] * resolvent[i, j] * rho[i] ** tau
    return gamma


def errors(gamma):
    eps = (gamma - target) / target
    eps_cumulative = np.cumsum(np.abs(eps)) / (np.arange(len(gamma)) + 1)
    return eps, eps_cumulative


def render_panels(K_values, J_values, output_path):
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    figure, axes = plt.subplots(2, 2, figsize=(8, 5.5))

    entries = [(K, J) for K in K_values for J in J_values]
    for idx, (K, J) in enumerate(entries):
        color = palette[idx % len(palette)]
        gamma = sd_autocovariance(tau, K, J)
        eps, eps_cumulative = errors(gamma)
        in_band = (tau >= 1.0) & (tau <= phi ** K)
        label = f"$K={K}$" if len(J_values) == 1 else f"$K={K},\\ J={J}$"
        axes[0, 0].plot(tau, np.where(in_band, eps, np.nan), color=color, lw=0.9, label=label)
        axes[0, 1].plot(tau, np.where(in_band, eps_cumulative, np.nan), color=color, lw=0.9, label=label)
        axes[1, 0].plot(tau, eps, color=color, lw=0.9, label=label)
        axes[1, 1].plot(tau, eps_cumulative, color=color, lw=0.9, label=label)

    K_max = max(K_values)
    eps_ss, eps_ss_cumulative = errors(bc_autocovariance(tau, K_max))
    in_band = (tau >= 1.0) & (tau <= phi ** K_max)
    axes[0, 0].plot(tau, np.where(in_band, eps_ss, np.nan), color="black", lw=1.1, ls="--", label="SS", zorder=10)
    axes[0, 1].plot(tau, np.where(in_band, eps_ss_cumulative, np.nan), color="black", lw=1.1, ls="--", label="_nolegend_", zorder=10)
    axes[1, 0].plot(tau, eps_ss, color="black", lw=1.1, ls="--", label="_nolegend_", zorder=10)
    axes[1, 1].plot(tau, eps_ss_cumulative, color="black", lw=1.1, ls="--", label="_nolegend_", zorder=10)

    K_present = list(set(K for K, _ in entries))
    for axis in axes.flat:
        axis.axvline(1.0, color="gray", lw=0.5, ls="--")
        for K in K_present:
            axis.axvline(phi ** K, color="gray", lw=0.5, ls="--")
        axis.set_xscale("log")
        axis.set_xlim(1e0, 1e6)
        axis.set_xlabel(r"$\tau$", fontsize=9)
        axis.xaxis.set_major_formatter(mticker.LogFormatterMathtext())

    axes[0, 0].set_ylabel(r"$\epsilon$", fontsize=9)
    axes[0, 1].set_ylabel(r"Cumulative $\epsilon$", fontsize=9)
    axes[1, 0].set_ylabel(r"$\epsilon$", fontsize=9)
    axes[1, 1].set_ylabel(r"Cumulative $\epsilon$", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = min(len(entries) + 1, 6)
    figure.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8,
                  frameon=True, bbox_to_anchor=(0.5, -0.04), handlelength=2.0, columnspacing=1.0)
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)


def render_K50_vary_J(J_values, output_path):
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    K = 50

    for idx, J in enumerate(J_values):
        color = palette[idx % len(palette)]
        gamma = sd_autocovariance(tau, K, J)
        eps, eps_cumulative = errors(gamma)
        in_band = (tau >= 1.0) & (tau <= phi ** K)
        label = f"$J = {J}$"
        axes[0].plot(tau, np.where(in_band, eps, np.nan), color=color, lw=0.9, label=label)
        axes[1].plot(tau, np.where(in_band, eps_cumulative, np.nan), color=color, lw=0.9, label=label)

    eps_ss, eps_ss_cumulative = errors(bc_autocovariance(tau, K))
    in_band = (tau >= 1.0) & (tau <= phi ** K)
    axes[0].plot(tau, np.where(in_band, eps_ss, np.nan), color="black", lw=1.1, ls="--", label="SS", zorder=10)
    axes[1].plot(tau, np.where(in_band, eps_ss_cumulative, np.nan), color="black", lw=1.1, ls="--", label="_nolegend_", zorder=10)

    for axis in axes.flat:
        axis.axvline(1.0, color="gray", lw=0.5, ls="--")
        axis.axvline(phi ** K, color="gray", lw=0.5, ls="--")
        axis.set_xscale("log")
        axis.set_xlim(1e0, 1e6)
        axis.set_xlabel(r"$\tau$", fontsize=9)
        axis.xaxis.set_major_formatter(mticker.LogFormatterMathtext())

    axes[0].set_ylabel(r"$\epsilon$", fontsize=9)
    axes[1].set_ylabel(r"Cumulative $\epsilon$", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(J_values) + 1,
                  fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.08), handlelength=2.0, columnspacing=1.0)
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)


render_panels(K_list, [0], "out/test/sd_panels_unadjusted_corrected.pdf")
render_panels(K_list, [300], "out/test/sd_panels_adjusted_corrected.pdf")
render_K50_vary_J([0, 1, 2, 3, 5, 10], "out/test/sd_panels_K50_vary_J_corrected.pdf")
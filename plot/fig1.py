import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

eta = 0.7
phi = 10.0
K_values = [1, 3, 5, 10, 20, 50]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

def recursive_coefficients(K, eta, phi):
    c = np.zeros(K + 1)
    c[K] = 1.0
    for k in range(1, K + 1):
        s = 0.0
        for i in range(k):
            s += c[K-i] * phi**(-eta*(k-i)) * np.exp(eta*(1.0 - phi**(-(k-i))))
        c[K-k] = 1.0 - s
    return c

def approximation(tau, K, eta, phi):
    c = recursive_coefficients(K, eta, phi)
    i_arr = np.arange(K + 1)
    lambdas = eta * phi**(-i_arr)
    w = np.exp(eta) * c * phi**(-i_arr * eta)
    return np.array([np.sum(w * np.exp(-lambdas * t)) for t in tau])

tau = np.logspace(0, 6, 3000)
target = tau**(-eta)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

titles = [
    ("With truncation", "With truncation"),
    ("Without truncation", "Without truncation")
]

for idx, (K, color) in enumerate(zip(K_values, colors)):
    g = approximation(tau, K, eta, phi)
    rel_err    = g / target - 1.0
    cumulative = np.cumsum(np.abs(rel_err)) / np.arange(1, len(tau) + 1)

    tau_max = phi**K
    mask = (tau >= 1.0) & (tau <= tau_max)
    rel_err_trunc = np.where(mask, rel_err,    np.nan)
    cumul_trunc   = np.where(mask, cumulative, np.nan)

    label = f"$K = {K}$"

    axes[0, 0].semilogx(tau, rel_err_trunc, color=color, linewidth=1.5, label=label)
    axes[0, 1].semilogx(tau, cumul_trunc,   color=color, linewidth=1.5)
    axes[1, 0].semilogx(tau, rel_err,       color=color, linewidth=1.5)
    axes[1, 1].semilogx(tau, cumulative,    color=color, linewidth=1.5)

boundary_taus = [phi**k for k in range(1, 51) if phi**k <= 1e6]
for ax in axes.flat:
    for t_b in boundary_taus:
        ax.axvline(x=t_b, color="gray", linewidth=0.7, linestyle="-")
    ax.set_xlabel(r"$\tau$")
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

axes[0, 0].set_title("With truncation")
axes[0, 1].set_title("With truncation")
axes[1, 0].set_title("Without truncation")
axes[1, 1].set_title("Without truncation")

axes[0, 0].set_ylabel(r"$\varepsilon$")
axes[0, 1].set_ylabel(r"Cumulative $\varepsilon$")
axes[1, 0].set_ylabel(r"$\varepsilon$")
axes[1, 1].set_ylabel(r"Cumulative $\varepsilon$")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(K_values),
           frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    r"Relative Approximation Error against true power law ($\eta=0.7$, $\phi=10$)",
    fontsize=12
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("out/test/fig1.pdf", bbox_inches="tight")
print("saved")
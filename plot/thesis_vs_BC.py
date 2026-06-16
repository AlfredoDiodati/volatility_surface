import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

alpha = 0.7
beta = 10.0
N = 10

def recursive_coefficients(N, alpha, beta):
    c = np.zeros(N + 1)
    c[N] = 1.0
    for k in range(1, N + 1):
        s = 0.0
        for i in range(k):
            s += c[N-i] * beta**(-alpha*(k-i)) * np.exp(alpha*(1.0 - beta**(-(k-i))))
        c[N-k] = 1.0 - s
    return c

c = recursive_coefficients(N, alpha, beta)
i_arr = np.arange(N + 1)
lambdas = alpha * beta**(-i_arr)

# BC Eq.(4): g(x) = sum c_i * beta^{-i*alpha} * exp(alpha) * exp(-alpha/beta^i * x)
w_bc     = c * beta**(-i_arr * alpha) * np.exp(alpha)
# thesis Eq.(25): w_i = c_i * phi^{-i*eta}  (exp(alpha) absent)
w_thesis = c * beta**(-i_arr * alpha)

x = np.logspace(0, N, 2000)
f = x**(-alpha)
g_bc     = np.array([np.sum(w_bc     * np.exp(-lambdas * xi)) for xi in x])
g_thesis = np.array([np.sum(w_thesis * np.exp(-lambdas * xi)) for xi in x])

figure, axis = plt.subplots(figsize=(8, 4.5))
axis.semilogx(x, g_bc/f - 1.0,     color="navy",    label=r"BC Eq.(4): $w_i = c_i\,\beta^{-i\alpha}\,e^{\alpha}$")
axis.semilogx(x, g_thesis/f - 1.0, color="crimson", label=r"thesis Eq.(25): $w_i = c_i\,\phi^{-i\eta}$")
axis.axhline(0.0,              color="navy",    linestyle=":", linewidth=1)
axis.axhline(np.exp(-alpha)-1, color="crimson", linestyle=":", linewidth=1)
axis.set_xlabel(r"$x$")
axis.set_ylabel(r"relative error $g(x)/f(x) - 1$")
axis.set_title(rf"$\alpha={alpha}$, $\beta={beta:.0f}$, $N={N}$")
axis.legend(loc="lower left", fontsize=9)
axis.grid(True, which="both", alpha=0.25)
figure.tight_layout()
figure.savefig("out/test/thesis_vs_BC.pdf")
print("saved")

"""
Gas vs Markov switching multifractal.
"""
import numpy as np
from fractrics import MSM
import matplotlib.pyplot as plt

from old_v.v4 import simulation

rng = np.random.default_rng(123)
nsim = 10000
m0 = 1.5
psi_bar = 1.0
max_k = 10
gamma_1 = 0.05
gamma_k = 1.0 - gamma_1
b = 3.0
alpha = 0.1
temperature = 5.0
k = 5
f_in = np.zeros(k)

model_info = MSM.metadata(data=None,
    parameters= {
    'unconditional_term': psi_bar,
    'arrival_gdistance': b,
    'hf_arrival': gamma_k,
    'marginal_value': m0 
    }, num_latent=k)

msm_ret, msm_vol = MSM.simulation(nsim, model_info, seed=123)
gas_ret, gas_vol = simulation(
    nsim, rng, m0, psi_bar, k, f_in, gamma_1, b, alpha, temperature
)

print(f"""
      m0: {m0}, psi_bar: {psi_bar}, k {k}, gamma_1 {gamma_1},
""")

m_high = m0
m_low = 2.0 - m0
j = np.arange(k+1)

grid_sigma2_direct = np.exp(psi_bar) * (m_low**(k-j)) * (m_high**j)
grid_log_sigma2_direct = np.log(grid_sigma2_direct)

grid_sigma_direct = np.exp(psi_bar) * (m_low**(k-j)) * (m_high**j)
grid_log_sigma2_if_sigma = 2.0*np.log(grid_sigma_direct)

print("grid log(sigma2) if multipliers on sigma2:", grid_log_sigma2_direct)
print("grid log(sigma2) if multipliers on sigma:", grid_log_sigma2_if_sigma)


plt.figure()
plt.plot(msm_ret, label="msm")
plt.plot(gas_ret, label="gas")
plt.legend()
plt.title("MSM vs gas returns")
plt.savefig("_returns.png", dpi=150)
plt.close()

# --- sanity checks -----------------------------------------------------------

def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    var = np.dot(x, x)
    if var == 0.0:
        return np.full(max_lag + 1, np.nan, dtype=float)
    out = np.empty(max_lag + 1, dtype=float)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(x[:-lag], x[lag:]) / var
    return out

def _kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    m2 = np.mean(x * x)
    if m2 == 0.0:
        return np.nan
    m4 = np.mean(x**4)
    return m4 / (m2 * m2)  # Pearson kurtosis (normal = 3)

max_lag = 200
msm_logv = np.log(msm_vol)
gas_logv = np.log(gas_vol)

msm_acf = _acf(msm_logv, max_lag)
gas_acf = _acf(gas_logv, max_lag)

print("\n--- Sanity checks ---")
print(f"kurtosis(msm returns): {_kurtosis(msm_ret):.4f}")
print(f"kurtosis(gas returns): {_kurtosis(gas_ret):.4f}")
print(f"kurtosis(msm log sigma2): {_kurtosis(msm_logv):.4f}")
print(f"kurtosis(gas log sigma2): {_kurtosis(gas_logv):.4f}")

qs = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
print("\nlog(sigma2) quantiles at", qs.tolist())
print("msm:", np.quantile(msm_logv[np.isfinite(msm_logv)], qs))
print("gas:", np.quantile(gas_logv[np.isfinite(gas_logv)], qs))

plt.figure()
plt.plot(msm_acf, label="msm")
plt.plot(gas_acf, label="gas")
plt.title("ACF of log(sigma2)")
plt.xlabel("lag")
plt.ylabel("acf")
plt.legend()
plt.tight_layout()
plt.savefig("_acf_log_sigma2.png", dpi=150)
plt.close()

plt.figure()
plt.hist(msm_logv[np.isfinite(msm_logv)], bins=80, alpha=0.5, density=True, label="msm")
plt.hist(gas_logv[np.isfinite(gas_logv)], bins=80, alpha=0.5, density=True, label="gas")
plt.title("Distribution of log(sigma2)")
plt.xlabel("log(sigma2)")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.savefig("_dist_log_sigma2.png", dpi=150)
plt.close()


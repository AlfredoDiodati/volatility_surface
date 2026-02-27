import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma

q_grid = np.linspace(0.5, 6, 50)

df = pd.read_parquet("data/SPY/put/bucket.parquet")

bool_cols = df.select_dtypes(include=["boolean", "bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)

Mt_cols = df.columns.drop(["DATE", "logIV"])
M = df[Mt_cols].values

k = M.shape[1]

P_pred = np.eye(k)
C = np.eye(k) * 0.02
H_t = np.array([[0.05]])

state_vals = []
sd_vals = []
sd_adj_vals = []

for q in q_grid:
    s_ss = []
    s_sd = []
    s_adj = []
    coeff = 2**(q/2) * gamma((q+1)/2) / np.sqrt(np.pi)

    for i in range(len(df)):
        M_row = M[i].reshape(1, -1)

        F_t = M_row @ P_pred @ M_row.T + H_t
        var_ss = F_t[0, 0]

        var_sd = H_t[0, 0]

        var_adj = (H_t + M_row @ C @ M_row.T)[0, 0]

        s_ss.append(coeff * var_ss**(q/2))
        s_sd.append(coeff * var_sd**(q/2))
        s_adj.append(coeff * var_adj**(q/2))

    state_vals.append(np.log(np.mean(s_ss)))
    sd_vals.append(np.log(np.mean(s_sd)))
    sd_adj_vals.append(np.log(np.mean(s_adj)))

os.makedirs("plot/test", exist_ok=True)

plt.figure()
plt.plot(q_grid, state_vals, label="State Space")
plt.plot(q_grid, sd_vals, label="Score Driven")
plt.plot(q_grid, sd_adj_vals, label="Score Driven Adjusted")
plt.xlabel("q")
plt.ylabel("log S_q")
plt.legend()
plt.savefig("plot/test/scaling_gaussianmodels.pdf")
plt.close()
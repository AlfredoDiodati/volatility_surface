import numpy as np
import pandas as pd

DATA_PATH = "data/SPX/otm/full.parquet"
W = 10


def build_stilde(k, tau, w, kernel="gaussian"):
    coords = np.column_stack([w * k, tau])
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1))
    if kernel == "gaussian":
        S = np.exp(-d**2)
    else:
        S = np.exp(-d)
    return S


df = pd.read_parquet(DATA_PATH, columns=["DATE", "logIV", "moneyness", "maturity"])

date = df["DATE"].iloc[0]
cross = df[df["DATE"] == date].copy()

cross["log_moneyness"] = np.log(cross["moneyness"])

k = cross["log_moneyness"].to_numpy()
tau = cross["maturity"].to_numpy()

S_gaussian = build_stilde(k, tau, w=W, kernel="gaussian")
S_laplacian = build_stilde(k, tau, w=W, kernel="laplacian")

np.save("scaling/S_gaussian.npy", S_gaussian)
np.save("scaling/S_laplacian.npy", S_laplacian)

print(f"date: {date}, n={len(k)}")
print(f"S_gaussian shape: {S_gaussian.shape}")
print(f"S_laplacian shape: {S_laplacian.shape}")

for name, S in [("gaussian", S_gaussian), ("laplacian", S_laplacian)]:
    off = S[~np.eye(len(S), dtype=bool)]
    print(f"{name} — below 1e-4: {(off < 1e-4).mean():.3f}, below 1e-2: {(off < 1e-2).mean():.3f}")

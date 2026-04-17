import os
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import chi2, norm
from scipy.special import xlogy

from fitting_models.mcs import mcs

PERF_DIR = "out/SPX/put/bucket_performance"
OUTPUT_DIR = "out/SPX/put/mcs"
PARQUET_PATH = "data/SPX/put/bucket.parquet"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
TRAIN_SIZE = 500
VAR_ALPHA = float(norm.ppf(0.05))  # must match Q_ALPHA in bucket_performance.py
VAR_LEVEL = 0.05

MCS_ALPHA = 0.01
MCS_B = 2000
MCS_BLOCK = 100
MCS_SEED = 42

MODEL_ORDER = ["ss", "adjSD", "fSD_K3", "fSD_K10", "fSD_K25", "fSD_K50", "fSD_K100", "fSD_K300"]


def load_y_test(parquet_path, train_size):
    raw = (
        pl.read_parquet(parquet_path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    raw = raw.with_columns(
        pl.max_horizontal(
            [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
        ).alias("bucket_idx")
    ).sort(["DATE", *FACTOR_LOADING_COLS])
    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())
    y_cube = np.full((T, max_n), np.nan)
    for t, date in enumerate(dates):
        s = raw.filter(pl.col("DATE") == date).sort(FACTOR_LOADING_COLS)
        n_t = len(s)
        y_cube[t, :n_t] = s["logIV"].to_numpy()
    return y_cube[train_size:], dates[train_size:]


def per_step_mse(y_actual, y_hat, mask):
    return np.where(mask, (y_actual - y_hat) ** 2, 0.0).sum(axis=1) / mask.sum(axis=1)


def per_step_mae(y_actual, y_hat, mask):
    return np.where(mask, np.abs(y_actual - y_hat), 0.0).sum(axis=1) / mask.sum(axis=1)


def kupiec_test(hits, alpha):
    T = len(hits)
    T1 = int(hits.sum())
    T0 = T - T1
    pi_hat = T1 / T
    LR_uc = -2.0 * (
        xlogy(T0, 1 - alpha) + xlogy(T1, alpha)
        - xlogy(T0, 1 - pi_hat) - xlogy(T1, pi_hat)
    )
    return {"hit_rate": pi_hat, "LR_uc": LR_uc, "pval_uc": float(1 - chi2.cdf(LR_uc, df=1))}


def christoffersen_test(hits, alpha):
    H = np.asarray(hits, dtype=int)
    T = len(H)
    T1 = int(H.sum())
    T0 = T - T1
    pi_hat = T1 / T

    n00 = int(((H[:-1] == 0) & (H[1:] == 0)).sum())
    n01 = int(((H[:-1] == 0) & (H[1:] == 1)).sum())
    n10 = int(((H[:-1] == 1) & (H[1:] == 0)).sum())
    n11 = int(((H[:-1] == 1) & (H[1:] == 1)).sum())

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    LR_uc = -2.0 * (
        xlogy(T0, 1 - alpha) + xlogy(T1, alpha)
        - xlogy(T0, 1 - pi_hat) - xlogy(T1, pi_hat)
    )
    LR_ind = -2.0 * (
        xlogy(n00 + n10, 1 - pi_hat) + xlogy(n01 + n11, pi_hat)
        - xlogy(n00, 1 - pi_01) - xlogy(n01, pi_01)
        - xlogy(n10, 1 - pi_11) - xlogy(n11, pi_11)
    )
    LR_cc = LR_uc + LR_ind
    return {
        "hit_rate": pi_hat, "n_hits": T1,
        "LR_uc": LR_uc, "pval_uc": float(1 - chi2.cdf(LR_uc, df=1)),
        "LR_ind": LR_ind, "pval_ind": float(1 - chi2.cdf(LR_ind, df=1)),
        "LR_cc": LR_cc, "pval_cc": float(1 - chi2.cdf(LR_cc, df=2)),
    }


def tick_loss_series(realized, var_forecast, alpha):
    e = realized - var_forecast
    return e * (alpha - (e < 0).astype(float))


def main():
    print("Loading bucket_performance results...")
    df_step = pd.read_parquet(os.path.join(PERF_DIR, "step_results.parquet"))
    df_agg = pd.read_csv(os.path.join(PERF_DIR, "aggregate_metrics.csv"))

    available = set(df_step["model"].unique())
    model_names = [m for m in MODEL_ORDER if m in available]
    n_test = df_step[df_step["model"] == model_names[0]].shape[0]

    print("Loading actuals...")
    y_test_all, test_dates = load_y_test(PARQUET_PATH, TRAIN_SIZE)
    y_test_all = y_test_all[:n_test]
    mask_all = ~np.isnan(y_test_all)
    index = test_dates[:n_test]

    realized_P = np.where(mask_all, y_test_all, np.nan).mean(axis=1)  # (n_test,) portfolio mean

    print("Building per-step loss series...")
    mse_data, mae_data, negll_data, tick_data = {}, {}, {}, {}
    var_forecasts = {}

    for name in model_names:
        df_m = df_step[df_step["model"] == name].sort_values("date").reset_index(drop=True)
        preds = np.load(os.path.join(PERF_DIR, f"predictions_{name}.npy"))[:n_test]

        mse_data[name] = per_step_mse(y_test_all, preds, mask_all)
        mae_data[name] = per_step_mae(y_test_all, preds, mask_all)
        negll_data[name] = -df_m["oos_loglik"].values[:n_test]

        var_fc = df_m["VaR"].values[:n_test]
        var_forecasts[name] = var_fc
        tick_data[name] = tick_loss_series(realized_P, var_fc, VAR_LEVEL)

    df_mse = pd.DataFrame(mse_data, index=index)
    df_mae = pd.DataFrame(mae_data, index=index)
    df_negll = pd.DataFrame(negll_data, index=index)
    df_tick = pd.DataFrame(tick_data, index=index)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- VaR tests ---
    print("\nRunning VaR coverage tests...")
    var_rows = []
    for name in model_names:
        hits = (realized_P < var_forecasts[name]).astype(int)
        res = christoffersen_test(hits, VAR_LEVEL)
        var_rows.append({"model": name, **res})

    df_var = pd.DataFrame(var_rows)
    df_var.to_csv(os.path.join(OUTPUT_DIR, "var_coverage_tests.csv"), index=False)

    print(f"  {'model':20s}  {'hit_rate':>9}  {'pval_uc':>8}  {'pval_ind':>9}  {'pval_cc':>8}")
    for _, row in df_var.iterrows():
        print(f"  {row['model']:20s}  {row['hit_rate']:9.4f}  {row['pval_uc']:8.4f}  "
              f"{row['pval_ind']:9.4f}  {row['pval_cc']:8.4f}")

    # --- MCS ---
    criteria = {
        "mse": df_mse,
        "mae": df_mae,
        "neg_oos_loglik": df_negll,
        "tick_loss": df_tick,
    }

    print(f"\nRunning MCS (alpha={MCS_ALPHA}, B={MCS_B}, block_length={MCS_BLOCK})...")
    detail_rows = []
    for crit, df_losses in criteria.items():
        print(f"  {crit}...", flush=True)
        res = mcs(df_losses, alpha=MCS_ALPHA, B=MCS_B, block_length=MCS_BLOCK,
                  rng=np.random.default_rng(MCS_SEED))
        elim_set = res["elimination_order"]
        for name in model_names:
            detail_rows.append({
                "model": name,
                "criterion": crit,
                "in_mcs": name in res["mcs_models"],
                "pvalue": float(res["pvalues"][name]),
                "elimination_step": elim_set.index(name) + 1 if name in elim_set else None,
            })
        print(f"    MCS: {res['mcs_models']}")

    df_detail = pd.DataFrame(detail_rows)
    df_detail.to_csv(os.path.join(OUTPUT_DIR, "mcs_detail.csv"), index=False)

    df_wide = df_detail.pivot(index="model", columns="criterion",
                               values=["in_mcs", "pvalue", "elimination_step"])
    df_wide.columns = [f"{metric}_{crit}" for metric, crit in df_wide.columns]
    df_wide = df_wide.reset_index()

    col_order = (
        ["model"]
        + [f"in_mcs_{c}" for c in criteria]
        + [f"pvalue_{c}" for c in criteria]
        + [f"elimination_step_{c}" for c in criteria]
    )
    df_wide = df_wide[[c for c in col_order if c in df_wide.columns]]

    df_out = df_wide.merge(
        df_agg[["model", "mse", "mae", "total_oos_loglik", "aic", "bic"]], on="model"
    ).merge(
        df_var[["model", "hit_rate", "n_hits", "pval_uc", "pval_ind", "pval_cc"]], on="model"
    )
    df_out["rank_loglik"] = df_out["total_oos_loglik"].rank(ascending=False).astype(int)
    df_out["rank_mse"] = df_out["mse"].rank(ascending=True).astype(int)
    df_out["rank_mae"] = df_out["mae"].rank(ascending=True).astype(int)
    df_out["rank_tick"] = df_out["hit_rate"].apply(lambda h: abs(h - VAR_LEVEL)).rank().astype(int)
    df_out = df_out.sort_values("rank_loglik").reset_index(drop=True)

    df_out.to_csv(os.path.join(OUTPUT_DIR, "mcs_results.csv"), index=False)

    for crit, df_losses in criteria.items():
        df_losses.to_csv(os.path.join(OUTPUT_DIR, f"losses_{crit}.csv"))

    print("\n=== MCS Summary ===")
    for crit in criteria:
        winners = df_detail[(df_detail["criterion"] == crit) & df_detail["in_mcs"]]["model"].tolist()
        print(f"  {crit:20s}: {winners}")

    print("\n=== VaR Coverage (nominal 5%) ===")
    print(df_var[["model", "hit_rate", "n_hits", "pval_uc", "pval_ind", "pval_cc"]].to_string(index=False))

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

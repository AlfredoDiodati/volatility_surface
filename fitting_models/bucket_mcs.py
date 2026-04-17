import os
import numpy as np
import pandas as pd
import polars as pl

from fitting_models.mcs import mcs

PERF_DIR = "out/SPX/put/bucket_performance"
OUTPUT_DIR = "out/SPX/put/mcs"
PARQUET_PATH = "data/SPX/put/bucket.parquet"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
TRAIN_SIZE = 500

MCS_ALPHA = 0.10
MCS_B = 2000
MCS_BLOCK = 10
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

    print("Loading predictions and building loss series...")
    mse_data, mae_data, negll_data = {}, {}, {}
    for name in model_names:
        preds = np.load(os.path.join(PERF_DIR, f"predictions_{name}.npy"))[:n_test]
        mse_data[name] = per_step_mse(y_test_all, preds, mask_all)
        mae_data[name] = per_step_mae(y_test_all, preds, mask_all)
        df_m = df_step[df_step["model"] == name].sort_values("date").reset_index(drop=True)
        negll_data[name] = -df_m["oos_loglik"].values[:n_test]

    df_mse = pd.DataFrame(mse_data, index=index)
    df_mae = pd.DataFrame(mae_data, index=index)
    df_negll = pd.DataFrame(negll_data, index=index)

    criteria = {"mse": df_mse, "mae": df_mae, "neg_oos_loglik": df_negll}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nRunning MCS (alpha={MCS_ALPHA}, B={MCS_B}, block_length={MCS_BLOCK})...")

    detail_rows = []
    for crit, df_losses in criteria.items():
        print(f"  {crit}...", flush=True)
        res = mcs(df_losses, alpha=MCS_ALPHA, B=MCS_B, block_length=MCS_BLOCK,
                  rng=np.random.default_rng(MCS_SEED))
        elim_set = res["elimination_order"]
        for name in model_names:
            elim_pos = elim_set.index(name) + 1 if name in elim_set else None
            detail_rows.append({
                "model": name,
                "criterion": crit,
                "in_mcs": name in res["mcs_models"],
                "pvalue": float(res["pvalues"][name]),
                "elimination_step": elim_pos,
            })
        print(f"    MCS: {res['mcs_models']}")

    df_detail = pd.DataFrame(detail_rows)

    df_wide = df_detail.pivot(index="model", columns="criterion", values=["in_mcs", "pvalue", "elimination_step"])
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
        df_agg[["model", "mse", "mae", "total_oos_loglik", "aic", "bic"]],
        on="model",
    )
    df_out["model_rank_oos_loglik"] = df_out["total_oos_loglik"].rank(ascending=False).astype(int)
    df_out["model_rank_mse"] = df_out["mse"].rank(ascending=True).astype(int)
    df_out["model_rank_mae"] = df_out["mae"].rank(ascending=True).astype(int)
    df_out = df_out.sort_values("model_rank_oos_loglik").reset_index(drop=True)

    df_out.to_csv(os.path.join(OUTPUT_DIR, "mcs_results.csv"), index=False)
    df_detail.to_csv(os.path.join(OUTPUT_DIR, "mcs_detail.csv"), index=False)

    for crit in criteria:
        df_losses = criteria[crit]
        df_losses.to_csv(os.path.join(OUTPUT_DIR, f"losses_{crit}.csv"))

    print("\n=== MCS Summary ===")
    for crit in criteria:
        winners = df_detail[(df_detail["criterion"] == crit) & df_detail["in_mcs"]]["model"].tolist()
        print(f"  {crit:20s}: {winners}")

    print("\n=== Full Results ===")
    display_cols = (
        ["model"]
        + [f"in_mcs_{c}" for c in criteria]
        + ["mse", "mae", "total_oos_loglik", "aic", "bic"]
        + ["model_rank_oos_loglik"]
    )
    print(df_out[[c for c in display_cols if c in df_out.columns]].to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

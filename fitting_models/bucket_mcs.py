import os
import jax
import jax.numpy as jnp
import polars as pl
from scipy.stats import chi2
from scipy.special import xlogy

from fitting_models.mcs import mcs

PERF_DIR = "out/SPX/put/bucket_performance"
OUTPUT_DIR = "out/SPX/put/mcs"
PARQUET_PATH = "data/SPX/put/bucket.parquet"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
TRAIN_SIZE = 500
VAR_LEVEL = 0.05
Q_ALPHA = -1.6448536269514722  # must match bucket_performance.py

MCS_ALPHA = 0.10
MCS_B = 10000
MCS_BLOCK = 10
MCS_SEED = 42

MODEL_ORDER = ["ss", "adjSD", "fSD_K3", "fSD_K10", "fSD_K25", "fSD_K50", "fSD_K100", "fSD_K300"]


def load_y_test(parquet_path, train_size):
    raw = (
        pl.read_parquet(parquet_path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    raw = (
        raw
        .with_columns(
            pl.max_horizontal(
                [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
            ).alias("bucket_idx")
        )
        .sort(["DATE", *FACTOR_LOADING_COLS])
        .with_columns(
            pl.int_range(pl.len()).over("DATE").cast(pl.Int32).alias("row_in_date")
        )
    )
    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())
    raw = raw.with_columns(
        pl.col("DATE").replace(
            {d: i for i, d in enumerate(dates)}, return_dtype=pl.Int32
        ).alias("date_idx")
    )
    t_idx = jnp.array(raw["date_idx"], dtype=jnp.int32)
    n_idx = jnp.array(raw["row_in_date"], dtype=jnp.int32)
    y_vals = jnp.array(raw["logIV"])
    y_cube = jnp.full((T, max_n), jnp.nan).at[t_idx, n_idx].set(y_vals)
    return y_cube[train_size:], dates[train_size:]


def per_step_mse(y_actual, y_hat, mask):
    return (jnp.where(mask, (y_actual - y_hat) ** 2, 0.0).sum(axis=1) / mask.sum(axis=1)).tolist()


def per_step_mae(y_actual, y_hat, mask):
    return (jnp.where(mask, jnp.abs(y_actual - y_hat), 0.0).sum(axis=1) / mask.sum(axis=1)).tolist()


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
    H = jnp.asarray(hits, dtype=int)
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
    return (e * (alpha - (e < 0).astype(float))).tolist()


def main():
    print("Loading bucket_performance results...")
    df_step = pl.read_parquet(os.path.join(PERF_DIR, "step_results.parquet"))
    df_agg = pl.read_csv(os.path.join(PERF_DIR, "aggregate_metrics.csv"))

    available = set(df_step["model"].unique().to_list())
    model_names = [m for m in MODEL_ORDER if m in available]
    n_test = df_step.filter(pl.col("model") == model_names[0]).height

    print("Loading actuals...")
    y_test_all, test_dates = load_y_test(PARQUET_PATH, TRAIN_SIZE)
    y_test_all = y_test_all[:n_test]
    mask_all = ~jnp.isnan(y_test_all)
    index = test_dates[:n_test]

    realized_P = jnp.nanmean(y_test_all, axis=1)

    print("Building per-step loss series...")
    mse_data, mae_data, negll_data, tick_data = {}, {}, {}, {}
    var_forecasts = {}

    for name in model_names:
        df_m = df_step.filter(pl.col("model") == name).sort("date")
        pred_df = pl.read_parquet(os.path.join(PERF_DIR, f"predictions_{name}.parquet")).sort("date")
        preds = jnp.array(pred_df["predictions"].to_list())[:n_test]

        mse_data[name] = per_step_mse(y_test_all, preds, mask_all)
        mae_data[name] = per_step_mae(y_test_all, preds, mask_all)
        negll_data[name] = (-jnp.array(df_m["oos_loglik"].to_list())[:n_test]).tolist()

        var_fc = jnp.array(df_m["VaR"].to_list())[:n_test]
        var_forecasts[name] = var_fc
        tick_data[name] = tick_loss_series(realized_P, var_fc, VAR_LEVEL)

    df_mse = pl.DataFrame(mse_data)
    df_mae = pl.DataFrame(mae_data)
    df_negll = pl.DataFrame(negll_data)
    df_tick = pl.DataFrame(tick_data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nRunning VaR coverage tests...")
    var_rows = []
    for name in model_names:
        hits = (realized_P < var_forecasts[name]).astype(int)
        res = christoffersen_test(hits, VAR_LEVEL)
        var_rows.append({"model": name, **res})

    df_var = pl.DataFrame(var_rows)
    df_var.write_csv(os.path.join(OUTPUT_DIR, "var_coverage_tests.csv"))

    print(f"  {'model':20s}  {'hit_rate':>9}  {'pval_uc':>8}  {'pval_ind':>9}  {'pval_cc':>8}")
    for row in df_var.iter_rows(named=True):
        print(f"  {row['model']:20s}  {row['hit_rate']:9.4f}  {row['pval_uc']:8.4f}  "
              f"  {row['pval_ind']:9.4f}  {row['pval_cc']:8.4f}")

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
                  key=jax.random.PRNGKey(MCS_SEED))
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

    df_detail = pl.DataFrame(detail_rows)
    df_detail.write_csv(os.path.join(OUTPUT_DIR, "mcs_detail.csv"))

    df_wide = (
        df_detail
        .pivot(index="model", on="criterion", values=["in_mcs", "pvalue", "elimination_step"])
    )

    df_out = (
        df_wide
        .join(df_agg.select(["model", "mse", "mae", "total_oos_loglik", "aic", "bic"]), on="model")
        .join(df_var.select(["model", "hit_rate", "n_hits", "pval_uc", "pval_ind", "pval_cc"]), on="model")
    )
    df_out = df_out.sort("total_oos_loglik", descending=True)
    df_out.write_csv(os.path.join(OUTPUT_DIR, "mcs_results.csv"))

    for crit, df_losses in criteria.items():
        df_losses.write_csv(os.path.join(OUTPUT_DIR, f"losses_{crit}.csv"))

    print("\n=== MCS Summary ===")
    for crit in criteria:
        winners = df_detail.filter((pl.col("criterion") == crit) & pl.col("in_mcs"))["model"].to_list()
        print(f"  {crit:20s}: {winners}")

    print("\n=== VaR Coverage (nominal 5%) ===")
    print(df_var.select(["model", "hit_rate", "n_hits", "pval_uc", "pval_ind", "pval_cc"]))

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

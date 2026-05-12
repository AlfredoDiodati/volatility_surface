import os
import jax
import jax.numpy as jnp
import polars as pl
from scipy.stats import chi2
from scipy.special import xlogy

from fit.mcs import mcs
from fit._forecast_metrics import per_step_mse, per_step_mae, compute_aic, compute_bic, per_step_aic, per_step_bic

PERF_DIR = "out/SPX/otm/full_performance"
STEP_CACHE_DIR = "out/SPX/otm/full_performance/step_cache"
OUTPUT_DIR = "out/SPX/otm/full_mcs"
PARQUET_PATH = "data/SPX/otm/full.parquet"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
TRAIN_SIZE = 500
VAR_LEVEL = 0.05
Q_ALPHA = -1.6448536269514722  # must match full_performance.py

RUN_MCS = True

MCS_ALPHAS = [0.05, 0.10, 0.25]
MCS_BLOCKS = [1, 5, 10, 15, 20]
MCS_B = 10000
MCS_SEED = 42
CANONICAL_ALPHA = 0.10
CANONICAL_BLOCK = 10
MAJORITY = 3

def load_y_test(parquet_path, train_size):
    raw = (
        pl.read_parquet(parquet_path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
        .sort(["DATE", *FACTOR_LOADING_COLS])
        .with_columns(
            pl.int_range(pl.len()).over("DATE").cast(pl.Int32).alias("row_in_date")
        )
    )
    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())
    raw = raw.with_columns(
        pl.col("DATE").replace_strict(
            {d: i for i, d in enumerate(dates)}, return_dtype=pl.Int32
        ).alias("date_idx")
    )
    t_idx = jnp.array(raw["date_idx"], dtype=jnp.int32)
    n_idx = jnp.array(raw["row_in_date"], dtype=jnp.int32)
    y_vals = jnp.array(raw["logIV"])
    y_cube = jnp.full((T, max_n), jnp.nan).at[t_idx, n_idx].set(y_vals)
    return y_cube[train_size:], dates[train_size:]

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

CRIT_LABELS = {
    "mse": "MSE",
    "mae": "MAE",
    "neg_oos_loglik": r"Neg.\ Log-Lik.",
    "tick_loss": "Tick Loss",
    "aic": "AIC",
    "bic": "BIC",
}

_STATIC_LABELS = {"ss": "SS", "adjSD": "adjSD", "lmSD": "lmSD"}

def _model_label(name: str) -> str:
    if name in _STATIC_LABELS:
        return _STATIC_LABELS[name]
    for family in ("ffSD", "fSD", "ifSD"):
        if name.startswith(family + "_K"):
            k = name[len(family) + 2:]
            return rf"{family} ($K{{=}}{k}$)"
    return name

def _stars(in_mcs_at_alpha: dict) -> str:
    if in_mcs_at_alpha[0.25]:
        return r"$^{***}$"
    if in_mcs_at_alpha[0.10]:
        return r"$^{**}$"
    if in_mcs_at_alpha[0.05]:
        return r"$^{*}$"
    return ""

def _make_var_latex_table(model_names, df_var) -> str:
    rows_by_name = {r["model"]: r for r in df_var.iter_rows(named=True)}
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Hit Rate & $N_{\text{hits}}$ & $p_{\text{uc}}$ & $p_{\text{ind}}$ & $p_{\text{cc}}$ \\",
        r"\midrule",
    ]
    for name in model_names:
        r = rows_by_name[name]
        label = _model_label(name)
        lines.append(
            f"{label} & {r['hit_rate']:.4f} & {r['n_hits']} & "
            f"{r['pval_uc']:.4f} & {r['pval_ind']:.4f} & {r['pval_cc']:.4f}" + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\footnotesize Nominal VaR level $\alpha=5\%$. "
        r"$p_{\text{uc}}$: Kupiec unconditional coverage; "
        r"$p_{\text{ind}}$: independence; "
        r"$p_{\text{cc}}$: conditional coverage.}\\",
        r"\end{tabular}",
    ]
    return "\n".join(lines)

def _make_latex_table(model_names, crits, metric_means, in_mcs_by_name_crit_alpha, block) -> str:
    n_crits = len(crits)
    col_spec = "l" + "r" * n_crits
    crit_header = " & ".join(["Model"] + [CRIT_LABELS[c] for c in crits])
    footnote_cols = n_crits + 1
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        crit_header + r" \\",
        r"\midrule",
    ]
    for name in model_names:
        cells = [_model_label(name)]
        for crit in crits:
            val = metric_means[crit][name]
            s = _stars(in_mcs_by_name_crit_alpha[name][crit])
            cells.append(f"{val:.4f}{s}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        rf"\multicolumn{{{footnote_cols}}}{{l}}{{\footnotesize "
        rf"Block length $l={block}$. "
        r"$^{*}$in MCS at $\alpha=0.05$; "
        r"$^{**}$at $\alpha=0.10$; "
        r"$^{***}$at $\alpha=0.25$.}}\\",
        r"\end{tabular}",
    ]
    return "\n".join(lines)

def main():
    print("Loading full_performance results...")
    os.makedirs(STEP_CACHE_DIR, exist_ok=True)

    # Ingest any new step data into per-model cache (never overwrites existing entries)
    step_files = sorted(f for f in os.listdir(PERF_DIR)
                        if f.startswith("step_results") and f.endswith(".parquet"))
    for sf in step_files:
        df_new = pl.read_parquet(os.path.join(PERF_DIR, sf))
        for name in df_new["model"].unique().to_list():
            cache_path = os.path.join(STEP_CACHE_DIR, f"{name}.parquet")
            if not os.path.exists(cache_path):
                df_new.filter(pl.col("model") == name).write_parquet(cache_path)

    cache_files = sorted(f for f in os.listdir(STEP_CACHE_DIR) if f.endswith(".parquet"))
    df_step = pl.concat([pl.read_parquet(os.path.join(STEP_CACHE_DIR, f)) for f in cache_files])

    pred_names = {f[len("predictions_"):-len(".parquet")]
                  for f in os.listdir(PERF_DIR)
                  if f.startswith("predictions_") and f.endswith(".parquet")}
    available = set(df_step["model"].unique().to_list()) & pred_names
    model_names = sorted(m for m in available if not m.startswith("ifSD"))
    n_test = df_step.filter(pl.col("model") == model_names[0]).height

    print("Loading actuals...")
    y_test_all, test_dates = load_y_test(PARQUET_PATH, TRAIN_SIZE)
    y_test_all = y_test_all[:n_test]
    mask_all = ~jnp.isnan(y_test_all)
    total_obs = int(jnp.sum(mask_all))

    realized_P = jnp.nanmean(y_test_all, axis=1)

    print("Building per-step loss series...")
    mse_data, mae_data, negll_data, tick_data, aic_data, bic_data = {}, {}, {}, {}, {}, {}
    var_forecasts = {}
    agg_rows = []

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

        n_params = int(df_m["n_params"][0])
        aic_data[name] = per_step_aic(negll_data[name], n_params, n_test)
        bic_data[name] = per_step_bic(negll_data[name], n_params, total_obs, n_test)

        total_oos_ll = float(jnp.sum(jnp.array(df_m["oos_loglik"].to_list())[:n_test]))
        agg_rows.append({
            "model": name,
            "mse": float(jnp.mean(jnp.array(mse_data[name]))),
            "mae": float(jnp.mean(jnp.array(mae_data[name]))),
            "total_oos_loglik": total_oos_ll,
            "aic": float(compute_aic(jnp.array(total_oos_ll), n_params)),
            "bic": float(compute_bic(jnp.array(total_oos_ll), n_params, total_obs)),
        })

    df_agg = pl.DataFrame(agg_rows)

    def _has_nan(series: list) -> bool:
        return bool(jnp.any(jnp.isnan(jnp.array(series, dtype=float))).item())

    valid_model_names = [
        name for name in model_names
        if not any(_has_nan(s) for s in [
            mse_data[name], mae_data[name], negll_data[name],
            tick_data[name], aic_data[name], bic_data[name],
        ])
    ]
    skipped_models = set(model_names) - set(valid_model_names)
    if skipped_models:
        print(f"  Skipping models with NaN loss series: {sorted(skipped_models)}")

    all_loss_data = {
        "mse": mse_data,
        "mae": mae_data,
        "neg_oos_loglik": negll_data,
        "tick_loss": tick_data,
        "aic": aic_data,
        "bic": bic_data,
    }
    criteria = {
        crit: pl.DataFrame({n: all_loss_data[crit][n] for n in valid_model_names})
        for crit in all_loss_data
    }

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

    var_tex = _make_var_latex_table(model_names, df_var)
    with open(os.path.join(OUTPUT_DIR, "var_table.tex"), "w") as f:
        f.write(var_tex)

    print("\n=== VaR Coverage (nominal 5%) ===")
    print(df_var.select(["model", "hit_rate", "n_hits", "pval_uc", "pval_ind", "pval_cc"]))

    if not RUN_MCS:
        print(f"\nSaved to {OUTPUT_DIR}")
        return

    metric_means = {
        crit: pl.DataFrame({n: all_loss_data[crit][n] for n in model_names}).mean().to_dicts()[0]
        for crit in all_loss_data
    }

    print(f"\nRunning MCS grid (alphas={MCS_ALPHAS}, blocks={MCS_BLOCKS}, B={MCS_B})...")
    all_rows = []
    in_mcs_grid = {
        block: {name: {crit: {a: False for a in MCS_ALPHAS} for crit in all_loss_data} for name in model_names}
        for block in MCS_BLOCKS
    }

    for block in MCS_BLOCKS:
        for alpha in MCS_ALPHAS:
            print(f"  block={block}, alpha={alpha}...", flush=True)
            for crit, df_losses in criteria.items():
                res = mcs(df_losses, alpha=alpha, B=MCS_B, block_length=block,
                          key=jax.random.PRNGKey(MCS_SEED))
                elim_set = res["elimination_order"]
                for name in model_names:
                    if name in skipped_models:
                        in_mcs, pval, elim_step = False, float("nan"), None
                    else:
                        in_mcs = name in res["mcs_models"]
                        pval = float(res["pvalues"][name])
                        elim_step = elim_set.index(name) + 1 if name in elim_set else None
                    in_mcs_grid[block][name][crit][alpha] = in_mcs
                    all_rows.append({
                        "model": name,
                        "criterion": crit,
                        "alpha": alpha,
                        "block": block,
                        "in_mcs": in_mcs,
                        "pvalue": pval,
                        "elimination_step": elim_step,
                    })

    pl.DataFrame(all_rows).write_csv(os.path.join(OUTPUT_DIR, "mcs_grid.csv"))

    for crit, df_losses in criteria.items():
        df_losses.write_csv(os.path.join(OUTPUT_DIR, f"losses_{crit}.csv"))

    crits = list(all_loss_data.keys())
    all_tables = []
    for block in MCS_BLOCKS:
        table = _make_latex_table(
            model_names, crits, metric_means,
            in_mcs_grid[block], block,
        )
        all_tables.append(table)

    with open(os.path.join(OUTPUT_DIR, "mcs_table.tex"), "w") as f:
        f.write("\n\n".join(all_tables))

    df_wide = (
        pl.DataFrame(all_rows)
        .filter((pl.col("alpha") == CANONICAL_ALPHA) & (pl.col("block") == CANONICAL_BLOCK))
        .pivot(index="model", on="criterion", values=["in_mcs", "pvalue", "elimination_step"])
        .join(df_agg.select(["model", "mse", "mae", "total_oos_loglik", "aic", "bic"]), on="model")
        .join(df_var.select(["model", "hit_rate", "n_hits", "pval_uc", "pval_ind", "pval_cc"]), on="model")
        .sort("total_oos_loglik", descending=True)
    )
    df_wide.write_csv(os.path.join(OUTPUT_DIR, "mcs_results.csv"))

    print(f"\n=== MCS Summary (canonical alpha={CANONICAL_ALPHA}, block={CANONICAL_BLOCK}) ===")
    for crit in criteria:
        winners = [
            n for n in model_names
            if in_mcs_grid[CANONICAL_BLOCK][n][crit][CANONICAL_ALPHA]
        ]
        print(f"  {crit:20s}: {winners}")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

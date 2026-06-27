import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from fit.mcs import mcs
from fit._forecast_metrics import per_step_mse, per_step_mae, per_step_aic, per_step_bic

PERF_DIR = "out/SPX/otm/full_performance_fixed"
OUTPUT_DIR = "out/SPX/otm/full_mcs_fixed"
PARQUET_PATH = "data/SPX/otm/full.parquet"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]

FCST_HORIZONS = [5, 22, 66, 260]

MCS_ALPHAS = [0.05, 0.10, 0.25]
MCS_BLOCKS = [1, 5, 10, 15, 20]
MCS_B = 10000
MCS_SEED = 42
CANONICAL_BLOCK = 10
CANONICAL_ALPHA = 0.10

METRICS_HSTEP = ["mse", "mae"]
METRICS_1STEP_EXTRA = ["neg_oos_loglik", "aic", "bic"]
METRIC_LABELS = {
    "mse": "MSE",
    "mae": "MAE",
    "neg_oos_loglik": r"Neg.\ Log-Lik.",
    "aic": "AIC",
    "bic": "BIC",
}

_STATIC_LABELS = {"ss": "SS", "adjSD": "adjSD", "lmSD": "lmSD"}


def _model_label(name):
    if name in _STATIC_LABELS:
        return _STATIC_LABELS[name]
    for family in ("ffSS", "ffSD", "msmSD"):
        if name.startswith(family + "_K"):
            k = name[len(family) + 2:]
            return rf"{family} ($K={k}$)"
    return name


def _stars(in_mcs_at_alpha):
    if in_mcs_at_alpha.get(0.25, False):
        return r"$^{***}$"
    if in_mcs_at_alpha.get(0.10, False):
        return r"$^{**}$"
    if in_mcs_at_alpha.get(0.05, False):
        return r"$^{*}$"
    return ""


def load_actuals(parquet_path):
    raw = (
        pl.read_parquet(parquet_path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
        .sort(["DATE", *FACTOR_LOADING_COLS])
        .with_columns(pl.int_range(pl.len()).over("DATE").cast(pl.Int32).alias("row_in_date"))
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
    train_size = T // 2
    return y_cube[train_size:], dates[train_size:]


def _preds_from_df(df):
    return jnp.array(np.stack(df["predictions"].to_list()))


def _load_1step(model_names):
    return {
        name: _preds_from_df(
            pl.read_parquet(os.path.join(PERF_DIR, f"predictions_{name}.parquet")).sort("date")
        )
        for name in model_names
    }


def _load_hstep(model_names, h):
    return {
        name: _preds_from_df(
            pl.read_parquet(os.path.join(PERF_DIR, f"predictions_h_{name}.parquet"))
            .filter(pl.col("horizon") == h)
            .sort("date")
        )
        for name in model_names
    }


def _load_step_data(model_names):
    result = {}
    for name in model_names:
        path = os.path.join(PERF_DIR, f"step_{name}.parquet")
        if os.path.exists(path):
            result[name] = pl.read_parquet(path).sort("date")
    return result


def _has_nan(arr):
    return bool(jnp.any(jnp.isnan(arr)).item())


def _make_latex_table_horizons(model_names, all_horizons, metric, mean_losses, in_mcs, block):
    horizon_labels = {
        1: r"$h=1$", 5: r"$h=5$", 22: r"$h=22$", 66: r"$h=66$", 260: r"$h=260$",
    }
    col_labels = [horizon_labels[h] for h in all_horizons]
    n_cols = len(all_horizons)
    metric_label = METRIC_LABELS.get(metric, metric)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule",
        "Model & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]
    for name in model_names:
        cells = [_model_label(name)]
        for h in all_horizons:
            val = mean_losses.get((name, h, metric))
            s = _stars(in_mcs.get((h, metric, block, name), {}))
            if val is None or math.isnan(val):
                cells.append("---")
            else:
                cells.append(f"{val:.4f}{s}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{{metric_label} across forecast horizons (SPX OTM options, block length $l={block}$)."
        r" $^{*}$in MCS at $\alpha=0.05$; $^{**}$at $\alpha=0.10$; $^{***}$at $\alpha=0.25$.}}",
        rf"\label{{tab:mcs_{metric}_block{block}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _make_latex_table_1step(model_names, metrics_1step, mean_losses, in_mcs, block):
    col_labels = [METRIC_LABELS[m] for m in metrics_1step]
    n_cols = len(metrics_1step)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule",
        "Model & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]
    for name in model_names:
        cells = [_model_label(name)]
        for metric in metrics_1step:
            val = mean_losses.get((name, 1, metric))
            s = _stars(in_mcs.get((1, metric, block, name), {}))
            if val is None or math.isnan(val):
                cells.append("---")
            else:
                cells.append(f"{val:.4f}{s}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{One-step-ahead predictive performance (SPX OTM options, block length $l={block}$)."
        r" $^{*}$in MCS at $\alpha=0.05$; $^{**}$at $\alpha=0.10$; $^{***}$at $\alpha=0.25$.}}",
        rf"\label{{tab:mcs_1step_block{block}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    print("Loading actuals...")
    y_test, test_dates = load_actuals(PARQUET_PATH)
    n_test, max_n = y_test.shape
    total_obs = int(jnp.sum(~jnp.isnan(y_test)))
    print(f"  n_test={n_test}, max_n={max_n}, total_obs={total_obs}")

    pred_names = {
        f[len("predictions_"):-len(".parquet")]
        for f in os.listdir(PERF_DIR)
        if f.startswith("predictions_") and not f.startswith("predictions_h_") and f.endswith(".parquet")
    }
    hpred_names = {
        f[len("predictions_h_"):-len(".parquet")]
        for f in os.listdir(PERF_DIR)
        if f.startswith("predictions_h_") and f.endswith(".parquet")
    }
    model_names = sorted(pred_names & hpred_names)
    print(f"Models: {model_names}")

    step_data = _load_step_data(model_names)
    models_with_step = set(step_data)
    if models_with_step:
        print(f"Step data available for: {sorted(models_with_step)}")
    else:
        print("No per-model step files found; neg-loglik/AIC/BIC will be skipped.")
        print("Re-run full_forecast_fixed.py to generate step_{model}.parquet files.")

    all_horizons = [1] + FCST_HORIZONS
    metrics_1step = METRICS_HSTEP + (METRICS_1STEP_EXTRA if models_with_step else [])

    losses = {}

    print("Loading 1-step predictions...")
    preds_1step = _load_1step(model_names)
    mask_1 = ~jnp.isnan(y_test)
    for name in model_names:
        preds = preds_1step[name]
        losses[(name, 1, "mse")] = jnp.array(per_step_mse(y_test, preds, mask_1))
        losses[(name, 1, "mae")] = jnp.array(per_step_mae(y_test, preds, mask_1))
        if name in step_data:
            df_s = step_data[name]
            neg_ll = (-jnp.array(df_s["oos_loglik"].to_list()))
            n_params = int(df_s["n_params"][0])
            losses[(name, 1, "neg_oos_loglik")] = neg_ll
            losses[(name, 1, "aic")] = jnp.array(per_step_aic(neg_ll.tolist(), n_params, n_test))
            losses[(name, 1, "bic")] = jnp.array(per_step_bic(neg_ll.tolist(), n_params, total_obs, n_test))
    del preds_1step

    for h in FCST_HORIZONS:
        n_valid = n_test - h
        print(f"Loading {h}-step predictions (n_valid={n_valid})...")
        preds_h = _load_hstep(model_names, h)
        y_actual_h = y_test[h:]
        mask_h = ~jnp.isnan(y_actual_h)
        for name in model_names:
            preds = preds_h[name]
            losses[(name, h, "mse")] = jnp.array(per_step_mse(y_actual_h, preds, mask_h))
            losses[(name, h, "mae")] = jnp.array(per_step_mae(y_actual_h, preds, mask_h))
        del preds_h

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for h in all_horizons:
        metrics = metrics_1step if h == 1 else METRICS_HSTEP
        for metric in metrics:
            rows = {
                name: losses[(name, h, metric)].tolist()
                for name in model_names
                if (name, h, metric) in losses
            }
            if rows:
                pl.DataFrame(rows).write_csv(os.path.join(OUTPUT_DIR, f"losses_h{h}_{metric}.csv"))

    in_mcs = {}
    all_rows = []

    for h in all_horizons:
        metrics = metrics_1step if h == 1 else METRICS_HSTEP
        for metric in metrics:
            available = [n for n in model_names if (n, h, metric) in losses]
            valid_models = [n for n in available if not _has_nan(losses[(n, h, metric)])]
            skipped = sorted(set(available) - set(valid_models))
            if skipped:
                print(f"  h={h}, {metric}: skipping {skipped} (NaN)")
            if len(valid_models) < 2:
                print(f"  h={h}, {metric}: fewer than 2 valid models, skipping MCS")
                continue

            for block in MCS_BLOCKS:
                for alpha in MCS_ALPHAS:
                    print(f"  h={h}, {metric}, block={block}, alpha={alpha}...", flush=True)
                    df_losses = pl.DataFrame(
                        {n: losses[(n, h, metric)].tolist() for n in valid_models}
                    )
                    res = mcs(
                        df_losses, alpha=alpha, B=MCS_B, block_length=block,
                        key=jax.random.PRNGKey(MCS_SEED),
                    )
                    mcs_set = set(res["mcs_models"])
                    elim_set = res["elimination_order"]
                    for name in model_names:
                        key_mcs = (h, metric, block, name)
                        if key_mcs not in in_mcs:
                            in_mcs[key_mcs] = {}
                        if name in valid_models:
                            in_mcs[key_mcs][alpha] = name in mcs_set
                            pval = float(res["pvalues"][name])
                            elim_step = elim_set.index(name) + 1 if name in elim_set else None
                        else:
                            in_mcs[key_mcs][alpha] = False
                            pval = float("nan")
                            elim_step = None
                        all_rows.append({
                            "model": name,
                            "horizon": h,
                            "metric": metric,
                            "alpha": alpha,
                            "block": block,
                            "in_mcs": in_mcs[key_mcs].get(alpha, False),
                            "pvalue": pval,
                            "elimination_step": elim_step,
                        })

    pl.DataFrame(all_rows).write_csv(os.path.join(OUTPUT_DIR, "mcs_grid.csv"))

    mean_losses = {}
    for (name, h, metric), arr in losses.items():
        mean_losses[(name, h, metric)] = float(jnp.mean(arr))

    for metric in METRICS_HSTEP:
        for block in MCS_BLOCKS:
            table = _make_latex_table_horizons(model_names, all_horizons, metric, mean_losses, in_mcs, block)
            with open(os.path.join(OUTPUT_DIR, f"mcs_table_{metric}_block{block}.tex"), "w") as f:
                f.write(table)

    for block in MCS_BLOCKS:
        table = _make_latex_table_1step(model_names, metrics_1step, mean_losses, in_mcs, block)
        with open(os.path.join(OUTPUT_DIR, f"mcs_table_1step_block{block}.tex"), "w") as f:
            f.write(table)

    print(f"\n=== MCS Summary (block={CANONICAL_BLOCK}, alpha={CANONICAL_ALPHA}) ===")
    for h in all_horizons:
        metrics = metrics_1step if h == 1 else METRICS_HSTEP
        for metric in metrics:
            winners = [
                n for n in model_names
                if in_mcs.get((h, metric, CANONICAL_BLOCK, n), {}).get(CANONICAL_ALPHA, False)
            ]
            print(f"  h={h:4d}, {metric}: {winners}")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

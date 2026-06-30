import math
import os

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import polars as pl

from fit.smcs import smcs
from fit._forecast_metrics import per_step_mse, per_step_mae, per_step_aic, per_step_bic

PERF_DIR = "out/SPX/otm/full_performance_fixed"
OUTPUT_DIR = "out/SPX/otm/full_smcs_fixed"
PLOT_DIR = "plot/SPX/otm/full_smcs_fixed"
PARQUET_PATH = "data/SPX/otm/full.parquet"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]

FCST_HORIZONS = [5, 22, 66, 260]
SMCS_ALPHAS = [0.05, 0.10, 0.25]
SMCS_HYPOTHESIS = "weak"

METRICS_HSTEP = ["mse", "mae"]
METRICS_1STEP_EXTRA = ["neg_oos_loglik", "aic", "bic"]
METRIC_LABELS = {
    "mse": "MSE",
    "mae": "MAE",
    "neg_oos_loglik": r"Neg.\ Log-Lik.",
    "aic": "AIC",
    "bic": "BIC",
}

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
})

ALPHA_COLORS = {0.05: "#d62728", 0.10: "#ff7f0e", 0.25: "#2ca02c"}
ALPHA_STYLES = {0.05: "--", 0.10: "-.", 0.25: ":"}
ALPHA_LABELS = {0.05: r"$\alpha=0.05$", 0.10: r"$\alpha=0.10$", 0.25: r"$\alpha=0.25$"}

_HEATMAP_CMAP = ListedColormap(["#d62728", "#fdae61", "#a1d99b", "#2ca02c"])
_HEATMAP_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
_HEATMAP_LEGEND = [
    mpatches.Patch(color="#2ca02c", label=r"in sMCS at $\alpha=0.25$ (all)"),
    mpatches.Patch(color="#a1d99b", label=r"in sMCS at $\alpha=0.10$, not $0.25$"),
    mpatches.Patch(color="#fdae61", label=r"in sMCS at $\alpha=0.05$ only"),
    mpatches.Patch(color="#d62728", label=r"excluded from all"),
]


def _model_label(name):
    if name == "ss":
        return "SS"
    if name == "adjSD":
        return "SD"
    if name == "lmSD":
        return "FI-SD"
    if name.startswith("ffSD_K"):
        k = name[len("ffSD_K"):]
        return rf"Mk-SD ($K={k}$)"
    if name.startswith("ffSS_K"):
        k = name[len("ffSS_K"):]
        return rf"Mk-SS ($K={k}$)"
    if name.startswith("msmSD_K"):
        k = name[len("msmSD_K"):]
        return rf"MSM ($K={k}$)"
    return name


def _stars(in_smcs_at_alpha):
    if in_smcs_at_alpha.get(0.25, False):
        return r"$^{***}$"
    if in_smcs_at_alpha.get(0.10, False):
        return r"$^{**}$"
    if in_smcs_at_alpha.get(0.05, False):
        return r"$^{*}$"
    return ""


def _family_k(name):
    for prefix in ("ffSS_K", "ffSD_K", "msmSD_K"):
        if name.startswith(prefix):
            return prefix[:-2], int(name[len(prefix):])
    return None, None


def _model_color(name, all_names):
    if name == "ss":
        return "#e74c3c"
    if name == "adjSD":
        return "#2980b9"
    if name == "lmSD":
        return "#7f8c8d"
    family, k = _family_k(name)
    if family is None:
        return "black"
    same = sorted(int(n[len(family) + 2:]) for n in all_names if n.startswith(family + "_K"))
    k_min, k_max = same[0], same[-1]
    t = (k - k_min) / max(1, k_max - k_min)
    shade = 0.40 + 0.45 * t
    if family == "ffSS":
        return plt.cm.Reds(shade)
    if family == "ffSD":
        return plt.cm.Blues(shade)
    if family == "msmSD":
        return plt.cm.Greens(shade)
    return "black"


def _model_colors(model_names):
    return {name: _model_color(name, model_names) for name in model_names}


def _fmt_date(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d


def _xtick_setup(ax, dates, n_ticks=8):
    T = len(dates)
    locs = np.linspace(0, T - 1, n_ticks, dtype=int)
    ax.set_xticks(locs)
    ax.set_xticklabels([_fmt_date(dates[i]) for i in locs], rotation=30, ha="right", fontsize=7)


def _membership_level(le):
    return np.where(
        le < -np.log(0.25), 3,
        np.where(le < -np.log(0.10), 2,
        np.where(le < -np.log(0.05), 1, 0))
    )


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


def _drop_nan_steps(loss_arrays, dates):
    matrix = np.stack([np.array(a) for a in loss_arrays.values()], axis=1)
    nan_rows = np.any(np.isnan(matrix), axis=1)
    if nan_rows.any():
        n_dropped = int(nan_rows.sum())
        print(f"    dropping {n_dropped} NaN time step(s) out of {len(dates)}")
        matrix = matrix[~nan_rows]
        dates = [d for d, bad in zip(dates, nan_rows) if not bad]
    names = list(loss_arrays.keys())
    clean = {names[j]: matrix[:, j].tolist() for j in range(len(names))}
    return clean, dates


def _plot_eprocess_page(pdf, model_names, log_e_adj, dates, h, metric, colors):
    T = len(dates)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in model_names:
        if name not in log_e_adj:
            continue
        le = np.array(log_e_adj[name])
        ax.plot(np.arange(T), le, color=colors[name], linewidth=1.0, alpha=0.9)
    for alpha in SMCS_ALPHAS:
        ax.axhline(-np.log(alpha), color=ALPHA_COLORS[alpha], linestyle=ALPHA_STYLES[alpha],
                   linewidth=1.1, zorder=3)
    ax.set_xlim(0, T - 1)
    ax.set_title(
        rf"Sequential log e-process: {METRIC_LABELS[metric]}, $h={h}$",
        fontsize=11,
    )
    ax.set_ylabel(r"$\log \tilde{e}_t$", fontsize=10)
    _xtick_setup(ax, dates)
    ax.tick_params(axis="y", labelsize=8)

    model_handles = [
        mpatches.Patch(color=colors[n], label=_model_label(n))
        for n in model_names if n in log_e_adj
    ]
    alpha_handles = [
        plt.Line2D([0], [0], color=ALPHA_COLORS[a], linestyle=ALPHA_STYLES[a],
                   linewidth=1.2, label=ALPHA_LABELS[a])
        for a in SMCS_ALPHAS
    ]
    ax.legend(
        handles=model_handles + alpha_handles,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap_page(pdf, model_names, log_e_adj, dates, h, metric):
    T = len(dates)
    active = [n for n in model_names if n in log_e_adj]
    M = len(active)

    grid = np.zeros((M, T), dtype=int)
    for i, name in enumerate(active):
        grid[i] = _membership_level(np.array(log_e_adj[name]))

    fig_h = max(3.0, 0.45 * M + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.imshow(grid, aspect="auto", cmap=_HEATMAP_CMAP, norm=_HEATMAP_NORM,
              interpolation="nearest")
    ax.set_yticks(np.arange(M))
    ax.set_yticklabels([_model_label(n) for n in active], fontsize=16)
    locs = np.linspace(0, T - 1, 8, dtype=int)
    ax.set_xticks(locs)
    ax.set_xticklabels([_fmt_date(dates[i]) for i in locs], rotation=0, ha="center", fontsize=14)
    fig.legend(
        handles=_HEATMAP_LEGEND,
        fontsize=14,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=4,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _make_latex_table_horizons(model_names, all_horizons, metric, mean_losses, in_smcs_final):
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
            s = _stars(in_smcs_final.get((h, metric, name), {}))
            if val is None or math.isnan(val):
                cells.append("---")
            else:
                cells.append(f"{val:.4f}{s}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{{metric_label} across forecast horizons (SPX OTM options, sMCS {SMCS_HYPOTHESIS} hypothesis)."
        r" $^{*}$in sMCS at $\alpha=0.05$; $^{**}$at $\alpha=0.10$; $^{***}$at $\alpha=0.25$.}}",
        rf"\label{{tab:smcs_{metric}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _make_latex_table_1step(model_names, metrics_1step, mean_losses, in_smcs_final):
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
            s = _stars(in_smcs_final.get((1, metric, name), {}))
            if val is None or math.isnan(val):
                cells.append("---")
            else:
                cells.append(f"{val:.4f}{s}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{One-step-ahead predictive performance (SPX OTM options, sMCS {SMCS_HYPOTHESIS} hypothesis)."
        r" $^{*}$in sMCS at $\alpha=0.05$; $^{**}$at $\alpha=0.10$; $^{***}$at $\alpha=0.25$.}}",
        r"\label{tab:smcs_1step}",
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
    if not models_with_step:
        print("No step files; neg-loglik/AIC/BIC will be skipped.")

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
            neg_ll = -jnp.array(df_s["oos_loglik"].to_list())
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
            losses[(name, h, "mse")] = jnp.array(per_step_mse(y_actual_h, preds_h[name], mask_h))
            losses[(name, h, "mae")] = jnp.array(per_step_mae(y_actual_h, preds_h[name], mask_h))
        del preds_h

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _metrics_for_h(h):
        return metrics_1step if h == 1 else METRICS_HSTEP

    def _raw_dates_for_h(h):
        return test_dates[h:] if h > 1 else test_dates

    log_e_adj_results = {}
    dates_results = {}
    in_smcs_final = {}
    all_rows = []

    for h in all_horizons:
        for metric in _metrics_for_h(h):
            available = [n for n in model_names if (n, h, metric) in losses]
            valid_models = [n for n in available if not _has_nan(losses[(n, h, metric)])]
            skipped = sorted(set(available) - set(valid_models))
            if skipped:
                print(f"  h={h}, {metric}: skipping {skipped} (all-NaN series)")
            if len(valid_models) < 2:
                print(f"  h={h}, {metric}: fewer than 2 valid models, skipping sMCS")
                continue

            raw_loss_arrays = {n: losses[(n, h, metric)] for n in valid_models}
            raw_dates = _raw_dates_for_h(h)
            clean_losses, clean_dates = _drop_nan_steps(raw_loss_arrays, raw_dates)

            if len(clean_dates) < 2:
                print(f"  h={h}, {metric}: too few clean time steps after NaN removal, skipping")
                continue

            print(f"  sMCS h={h}, {metric} ({len(valid_models)} models, T={len(clean_dates)})...", flush=True)
            df_losses = pl.DataFrame(clean_losses)
            res = smcs(df_losses, alpha=0.05, hypothesis=SMCS_HYPOTHESIS, running_intersection=False)
            log_e_adj = res.get("log_e_adjusted") or {}
            log_e_adj_results[(h, metric)] = log_e_adj
            dates_results[(h, metric)] = clean_dates

            for name in model_names:
                key = (h, metric, name)
                in_smcs_final[key] = {}
                if name in valid_models and log_e_adj:
                    le = np.array(log_e_adj[name])
                    last_le = float(le[-1])
                    for alpha in SMCS_ALPHAS:
                        final_in = bool(last_le < -np.log(alpha))
                        in_smcs_final[key][alpha] = final_in
                        all_rows.append({
                            "model": name, "horizon": h, "metric": metric,
                            "alpha": alpha, "in_smcs_final": final_in,
                        })
                else:
                    for alpha in SMCS_ALPHAS:
                        in_smcs_final[key][alpha] = False
                        all_rows.append({
                            "model": name, "horizon": h, "metric": metric,
                            "alpha": alpha, "in_smcs_final": False,
                        })

    pl.DataFrame(all_rows).write_csv(os.path.join(OUTPUT_DIR, "smcs_grid.csv"))

    mean_losses = {
        (name, h, metric): float(jnp.mean(arr))
        for (name, h, metric), arr in losses.items()
    }

    for metric in METRICS_HSTEP:
        table = _make_latex_table_horizons(model_names, all_horizons, metric, mean_losses, in_smcs_final)
        with open(os.path.join(OUTPUT_DIR, f"smcs_table_{metric}.tex"), "w") as f:
            f.write(table)

    table_1step = _make_latex_table_1step(model_names, metrics_1step, mean_losses, in_smcs_final)
    with open(os.path.join(OUTPUT_DIR, "smcs_table_1step.tex"), "w") as f:
        f.write(table_1step)

    colors = _model_colors(model_names)

    combo_list = [
        (h, metric)
        for h in all_horizons
        for metric in _metrics_for_h(h)
        if (h, metric) in log_e_adj_results
    ]

    all_metrics = sorted({metric for _, metric in combo_list})

    print("Generating e-process trajectory plots...")
    for h, metric in combo_list:
        fname = f"eprocess_{metric}_h{h}.pdf"
        with PdfPages(os.path.join(PLOT_DIR, fname)) as pdf:
            _plot_eprocess_page(
                pdf, model_names, log_e_adj_results[(h, metric)],
                dates_results[(h, metric)], h, metric, colors,
            )

    print("Generating SMCS membership heatmaps...")
    for metric in all_metrics:
        horizons = [h for h in all_horizons if (h, metric) in log_e_adj_results]
        if not horizons:
            continue
        fname = f"heatmap_{metric}.pdf"
        with PdfPages(os.path.join(PLOT_DIR, fname)) as pdf:
            for h in horizons:
                _plot_heatmap_page(
                    pdf, model_names, log_e_adj_results[(h, metric)],
                    dates_results[(h, metric)], h, metric,
                )

    print(f"\n=== sMCS Summary (alpha=0.10) ===")
    for h in all_horizons:
        for metric in _metrics_for_h(h):
            winners = [
                n for n in model_names
                if in_smcs_final.get((h, metric, n), {}).get(0.10, False)
            ]
            print(f"  h={h:4d}, {metric}: {winners}")

    print(f"\nSaved to {OUTPUT_DIR} and {PLOT_DIR}")


if __name__ == "__main__":
    main()

import argparse
import os
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = "out/SPX/otm/full_performance_fixed"
COVID_START = "2020-02-01"
COVID_END   = "2020-04-30"
J_LABELS    = ["level", "moneyness", "maturity", "bucket"]

_INS_PALETTE = [
    "#d62728", "#ff7f0e", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
_OOS_PALETTE = [
    "#1f77b4", "#2ca02c", "#17becf", "#aec7e8",
    "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2",
]


def _parse_dates(dates):
    return [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Model name, e.g. ffSD_K3 or ffSS_K3")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--out", default=None, help="Output HTML path (default: OUTPUT_DIR/b_<name>.html)")
    args = parser.parse_args()

    path = os.path.join(OUTPUT_DIR, f"b_{args.name}{args.suffix}.parquet")
    df = pl.read_parquet(path)

    k_vals = sorted(df["k"].unique().to_list())
    j_vals = sorted(df["j"].unique().to_list())
    P = len(j_vals)

    ins_dates_raw = sorted(df.filter(pl.col("is_insample"))["date"].unique().to_list())
    oos_dates_raw = sorted(df.filter(~pl.col("is_insample"))["date"].unique().to_list())
    break_date = _parse_dates([oos_dates_raw[0]])[0]

    fig = make_subplots(
        rows=P, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[J_LABELS[j] if j < len(J_LABELS) else f"j={j}" for j in j_vals],
    )

    for j_idx, j in enumerate(j_vals):
        row = j_idx + 1
        for k_idx, k in enumerate(k_vals):
            ins = (
                df.filter((pl.col("j") == j) & (pl.col("k") == k) & pl.col("is_insample"))
                .sort("date")
            )
            oos = (
                df.filter((pl.col("j") == j) & (pl.col("k") == k) & ~pl.col("is_insample"))
                .sort("date")
            )

            show = (j_idx == 0)
            ins_col = _INS_PALETTE[k_idx % len(_INS_PALETTE)]
            oos_col = _OOS_PALETTE[k_idx % len(_OOS_PALETTE)]

            fig.add_trace(go.Scatter(
                x=_parse_dates(ins["date"].to_list()),
                y=ins["b"].to_list(),
                mode="lines",
                line=dict(color=ins_col, width=1.2),
                name=f"k={k}  in-sample",
                legendgroup=f"k{k}_ins",
                showlegend=show,
            ), row=row, col=1)

            fig.add_trace(go.Scatter(
                x=_parse_dates(oos["date"].to_list()),
                y=oos["b"].to_list(),
                mode="lines",
                line=dict(color=oos_col, width=1.2),
                name=f"k={k}  out-of-sample",
                legendgroup=f"k{k}_oos",
                showlegend=show,
            ), row=row, col=1)

    fig.add_vrect(
        x0=COVID_START, x1=COVID_END,
        fillcolor="gray", opacity=0.15, layer="below", line_width=0,
        row="all", col="all",
    )

    fig.add_vline(
        x=break_date,
        line=dict(color="black", dash="dash", width=1.5),
        row="all", col="all",
    )

    for j_idx in range(P):
        label = J_LABELS[j_idx] if j_idx < len(J_LABELS) else f"j={j_idx}"
        fig.update_yaxes(title_text=label, row=j_idx + 1, col=1)

    fig.update_xaxes(title_text="date", row=P, col=1)
    fig.update_layout(
        title=f"Factor loadings b over time — {args.name}",
        height=250 * P,
        template="plotly_white",
        legend=dict(tracegroupgap=4),
    )

    out_path = args.out or os.path.join(OUTPUT_DIR, f"b_{args.name}{args.suffix}.html")
    fig.write_html(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

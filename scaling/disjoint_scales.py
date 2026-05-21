from scaling_reg import moment_scaling
import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots


def ou_tau(theta, moments):
    dt = np.arange(1, 127, dtype=float)
    tau = []
    for q in moments:
        log_var = np.log(1.0 - np.exp(-2.0 * theta * dt))
        log_sq = (q / 2.0) * log_var
        exponent = np.polyfit(np.log(dt), log_sq, 1)[0]
        tau.append(exponent - 1.0)
    return np.array(tau)


def base_layout(**extra):
    layout = dict(template="simple_white", font=dict(size=12))
    layout.update(extra)
    return layout


def main():
    subfolder = "SPX"
    out_dir = f"plot/{subfolder}/otm/scaling"
    os.makedirs(out_dir, exist_ok=True)

    data = pl.read_parquet(f"data/{subfolder}/otm/bucket_matrix.parquet")
    if "DATE" in data.columns:
        data = data.drop("DATE")

    print(f"nans in dataset {sum(data.null_count().row(0))}")

    moments = np.arange(1, 5) / 2
    tick_days = np.array([1, 5, 21, 63, 116])
    tick_labels = ["1 day", "1 week", "1 month", "3 months", "6 months"]

    n_q = len(moments)
    blue_q = pc.sample_colorscale("Blues", np.linspace(0.3, 0.9, n_q).tolist())

    holder_bm = moments / 2.0 - 1.0

    ou_thetas = [1 / 126, 1 / 63, 1 / 21, 1 / 5]
    ou_names = ["OU 6M", "OU 3M", "OU 1M", "OU 1W"]
    ou_colors = ["rgba(170,170,170,0.5)", "rgba(145,145,145,0.55)", "rgba(115,115,115,0.6)", "rgba(85,85,85,0.65)"]
    _ou_raw = [ou_tau(theta, moments) for theta in ou_thetas]
    ou_taus = [t - t[0] + holder_bm[0] for t in _ou_raw]

    scalings = {}

    for label in data.columns:
        col = data[label]
        print(f"{label}: {col.is_not_null().sum()} valid out of {len(col)}")

    for label in data.columns:
        arr = data[label].to_numpy().astype(float)
        arr[arr == 0] = np.nan
        scaling_dict = moment_scaling(arr, 1.0, 126.0, moments)
        dt = scaling_dict["delta_ts"]

        scalings[label] = np.array([scaling_dict[q]["holder"] for q in moments])

        dt_min = float(dt[0])
        dt_max = float(dt[-1])
        x_range = [np.log10(dt_min) - 0.02, np.log10(dt_max) + 0.02]

        for key, series_key in [("shifted_power_var", f"scaling{label}"), ("log_power_var", f"scaling_raw{label}")]:
            fig = go.Figure()
            for x in tick_days:
                fig.add_vline(x=x, line_dash="dash", line_width=0.8, line_color="lightgray")

            for i, q in enumerate(moments):
                y = np.exp(scaling_dict[q][key])

                if key == "shifted_power_var" and np.isfinite(y[0]) and y[0] > 0:
                    y = y / y[0]

                mask = np.isfinite(y) & (y > 0)
                if not mask.any():
                    continue

                fig.add_trace(go.Scatter(
                    x=dt[mask], y=y[mask], mode="lines",
                    line=dict(color=blue_q[i]),
                    showlegend=False
                ))
                fig.add_annotation(
                    x=float(dt[mask][-1]),
                    y=float(y[mask][-1]),
                    text=f"q={q:.1f}",
                    xref="x", yref="y",
                    xanchor="left",
                    showarrow=False,
                    font=dict(color=blue_q[i], size=10),
                    xshift=6,
                    cliponaxis=False
                )

            fig.update_layout(
                **base_layout(width=800, height=520, margin=dict(l=60, r=65, t=15, b=50)),
                xaxis=dict(type="log", range=x_range, tickvals=list(tick_days), ticktext=tick_labels),
                yaxis_type="log",
                xaxis_title="Δt",
                yaxis_title="S(q, Δt)",
            )
            fig.write_image(f"{out_dir}/{series_key}.pdf")

    underlying_series = pl.read_parquet(f"data/{subfolder}/otm/underlying.parquet")
    log_price = np.log(underlying_series["UNDERLYING_LAST"].to_numpy())
    scaling_dict_underlying = moment_scaling(log_price, 1.0, 126.0, moments)
    underlying_tau = np.array([scaling_dict_underlying[q]["holder"] for q in moments])

    all_tau = np.concatenate(list(scalings.values()) + [holder_bm] + ou_taus + [underlying_tau])
    valid_tau = all_tau[np.isfinite(all_tau)]
    pad = 0.06 * (valid_tau.max() - valid_tau.min())
    tau_range = [float(valid_tau.min()) - pad, float(valid_tau.max()) + pad]

    n_s = len(scalings)
    blue_s = pc.sample_colorscale("Blues", np.linspace(0.3, 0.9, n_s).tolist())

    for show_legend in [True, False]:
        fig = go.Figure()

        for ou_t, gray in zip(ou_taus, ou_colors):
            fig.add_trace(go.Scatter(x=moments, y=ou_t, mode="lines", line=dict(color=gray, width=1), showlegend=False))

        fig.add_trace(go.Scatter(x=moments, y=holder_bm, mode="lines", line=dict(color="black", dash="dash"), name="BM", showlegend=show_legend))
        for i, label in enumerate(scalings):
            fig.add_trace(go.Scatter(x=moments, y=scalings[label], mode="lines", line=dict(color=blue_s[i]), name=label, showlegend=show_legend))

        if show_legend:
            legend_cfg = dict(
                orientation="h",
                yanchor="top", y=-0.12,
                xanchor="left", x=0,
                font=dict(size=8),
                tracegroupgap=0
            )
            margin = dict(l=55, r=25, t=15, b=100)
        else:
            legend_cfg = {}
            margin = dict(l=55, r=25, t=15, b=50)

        fig.update_layout(
            **base_layout(width=700, height=520, margin=margin),
            xaxis=dict(range=[float(moments[0]), float(moments[-1])]),
            yaxis=dict(range=tau_range),
            xaxis_title="q",
            yaxis_title="τ(q)",
            showlegend=show_legend,
            legend=legend_cfg,
        )
        suffix = "_legend" if show_legend else ""
        fig.write_image(f"{out_dir}/moments_scaling{suffix}.pdf")

    bucket_labels = sorted(data.columns)
    moneyness_groups = sorted(set(lbl.split('_')[1] for lbl in bucket_labels))
    maturity_groups = sorted(set(lbl.split('_')[0] for lbl in bucket_labels))

    n_mat = len(maturity_groups)
    n_mon = len(moneyness_groups)
    blue_mat = pc.sample_colorscale("Blues", np.linspace(0.35, 0.9, n_mat).tolist())
    blue_mon = pc.sample_colorscale("Blues", np.linspace(0.35, 0.9, n_mon).tolist())

    fig = make_subplots(rows=1, cols=n_mon, subplot_titles=moneyness_groups)
    for mon_idx, moneyness in enumerate(moneyness_groups):
        col = mon_idx + 1
        show = col == 1
        for ou_t, gray, ou_name in zip(ou_taus, ou_colors, ou_names):
            fig.add_trace(go.Scatter(x=moments, y=ou_t, mode="lines", line=dict(color=gray, width=1), name=ou_name, showlegend=show), row=1, col=col)
        fig.add_trace(go.Scatter(x=moments, y=holder_bm, mode="lines", line=dict(color="black", dash="dash"), name="BM", showlegend=show), row=1, col=col)
        fig.add_trace(go.Scatter(x=moments, y=underlying_tau, mode="lines", line=dict(color="red"), name="Underlying", showlegend=show), row=1, col=col)
        for mat_idx, maturity in enumerate(maturity_groups):
            lbl = f"{maturity}_{moneyness}"
            if lbl in scalings:
                fig.add_trace(go.Scatter(x=moments, y=scalings[lbl], mode="lines", line=dict(color=blue_mat[mat_idx]), name=maturity, showlegend=show), row=1, col=col)

    fig.update_xaxes(range=[float(moments[0]), float(moments[-1])])
    fig.update_yaxes(range=tau_range)
    fig.update_layout(**base_layout(width=1800, height=500, margin=dict(l=55, r=55, t=40, b=50)))
    fig.write_image(f"{out_dir}/panel_tau_by_moneyness.pdf")

    fig = make_subplots(rows=1, cols=n_mat, subplot_titles=maturity_groups)
    for mat_idx, maturity in enumerate(maturity_groups):
        col = mat_idx + 1
        show = col == 1
        for ou_t, gray, ou_name in zip(ou_taus, ou_colors, ou_names):
            fig.add_trace(go.Scatter(x=moments, y=ou_t, mode="lines", line=dict(color=gray, width=1), name=ou_name, showlegend=show), row=1, col=col)
        fig.add_trace(go.Scatter(x=moments, y=holder_bm, mode="lines", line=dict(color="black", dash="dash"), name="BM", showlegend=show), row=1, col=col)
        fig.add_trace(go.Scatter(x=moments, y=underlying_tau, mode="lines", line=dict(color="red"), name="Underlying", showlegend=show), row=1, col=col)
        for mon_idx, moneyness in enumerate(moneyness_groups):
            lbl = f"{maturity}_{moneyness}"
            if lbl in scalings:
                fig.add_trace(go.Scatter(x=moments, y=scalings[lbl], mode="lines", line=dict(color=blue_mon[mon_idx]), name=moneyness, showlegend=show), row=1, col=col)

    fig.update_xaxes(range=[float(moments[0]), float(moments[-1])])
    fig.update_yaxes(range=tau_range)
    fig.update_layout(**base_layout(width=1800, height=500, margin=dict(l=55, r=55, t=40, b=50)))
    fig.write_image(f"{out_dir}/panel_tau_by_maturity.pdf")


if __name__ == "__main__":
    main()

import polars as pl
from pathlib import Path

src = "data/SPX/otm/full.parquet"
out = Path("out")
out.mkdir(exist_ok=True)

N_MON = 6

df = pl.read_parquet(src)

stats = (
    df.with_columns([
        (pl.col("bucket_idx") // N_MON + 1).alias("mat_bucket"),
        (pl.col("bucket_idx") % N_MON + 1).alias("mon_bucket"),
        pl.col("logIV").exp().alias("IV"),
        (pl.col("maturity") * 255).alias("TTM_days"),
    ])
    .group_by(["mat_bucket", "mon_bucket"])
    .agg([
        pl.col("IV").mean().alias("logIV_mean"),
        pl.col("IV").std().alias("logIV_std"),
        pl.col("moneyness").mean().alias("moneyness_mean"),
        pl.col("moneyness").std().alias("moneyness_std"),
        pl.col("TTM_days").mean().alias("maturity_mean"),
        pl.col("TTM_days").std().alias("maturity_std"),
    ])
    .sort(["mon_bucket", "mat_bucket"])
)

mat_labels = {
    1: "7--45 days",
    2: "45--90 days",
    3: "90--180 days",
    4: "180--360 days",
}
mon_labels = {
    1: "DOTM put",
    2: "OTM put",
    3: "ATM put",
    4: "ATM call",
    5: "OTM call",
    6: "DOTM call",
}
variables = [
    ("logIV",     r"IV"),
    ("moneyness", r"Moneyness"),
    ("maturity",  r"TTM"),
]

N_VARS = len(variables)
N_MAT  = len(mat_labels)

col_spec = "ll" + " rr" * N_MAT

lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\caption{Summary statistics by moneyness and maturity bucket.}")
lines.append(r"\label{tab:bucket_stats}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{" + col_spec + r"}")
lines.append(r"\toprule")

mat_multicols = " & ".join(
    rf"\multicolumn{{2}}{{c}}{{{mat_labels[m]}}}" for m in sorted(mat_labels)
)
lines.append(r" & & " + mat_multicols + r" \\")

cmidrules = " ".join(
    rf"\cmidrule(lr){{{3 + (m-1)*2}-{4 + (m-1)*2}}}" for m in sorted(mat_labels)
)
lines.append(cmidrules)

mean_std_header = " & ".join(r"Mean & Std" for _ in sorted(mat_labels))
lines.append(r" & & " + mean_std_header + r" \\")
lines.append(r"\midrule")

for mon in sorted(mon_labels):
    for i, (var, var_label) in enumerate(variables):
        row_cells = []
        for mat in sorted(mat_labels):
            row = stats.filter(
                (pl.col("mat_bucket") == mat) & (pl.col("mon_bucket") == mon)
            )
            if row.is_empty():
                row_cells += ["--", "--"]
            else:
                row_cells.append(f"{row[f'{var}_mean'][0]:.2f}")
                row_cells.append(f"{row[f'{var}_std'][0]:.2f}")

        if i == 0:
            group_col = rf"\multirow{{{N_VARS}}}{{*}}{{{mon_labels[mon]}}}"
        else:
            group_col = ""

        lines.append(group_col + " & " + var_label + " & " + " & ".join(row_cells) + r" \\")

    if mon < max(mon_labels):
        lines.append(r"[4pt]")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\begin{tablenotes}")
lines.append(r"\item \textit{Note:} This table presents summary statistics for the option data, including the mean and standard deviation (Std) over time for the implied volatility (IV), Time to maturity (TTM), moneyness (the strike price divided by the index) across four maturity groups and six moneyness groups. The maturity groups are 7--45, 45--90, 90--180, and 180--360 days. The six moneyness groups are defined as deep out-of-the-money put ($-$0.125 < $\Delta$ < 0, DOTM put), out-of-the-money put ($-$0.375 < $\Delta$ < $-$0.125, OTM put), at-the-money put ($-$0.5 < $\Delta$ < $-$0.375, ATM put), and similarly for call options (with positive $\Delta$s). Each day, all contracts that fall within each maturity-moneyness group are identified, and the numbers represent averages over time and across contracts for each group.")
lines.append(r"\end{tablenotes}")
lines.append(r"\end{table}")

latex = "\n".join(lines)
out_path = out / "table_bucket_stats.tex"
out_path.write_text(latex)
print(f"Written: {out_path}")
print(latex)

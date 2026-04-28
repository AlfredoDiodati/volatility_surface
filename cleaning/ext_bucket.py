import math
import polars as pl
import numpy as np
import gc
from pathlib import Path

subfolder = "SPX"
src = "data/" + subfolder + "/otm/filtered.parquet"
out = "data/" + subfolder + "/otm/"

MATURITY_BREAKS = [7, 14, 21, 30, 45, 60, 75, 90, 110, 130, 150, 180, 210, 270, 330, 360]
MONEYNESS_BREAKS = [-0.50, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
                     0.00,
                     0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

MINIMUM_COVERAGE_FRACTION = 0.5


def midpoint(lower, upper):
    return (lower + upper) / 2.0


maturity_bucket_midpoints = {
    i + 1: midpoint(MATURITY_BREAKS[i], MATURITY_BREAKS[i + 1])
    for i in range(len(MATURITY_BREAKS) - 1)
}

moneyness_bucket_midpoints = {
    i + 1: midpoint(MONEYNESS_BREAKS[i], MONEYNESS_BREAKS[i + 1])
    for i in range(len(MONEYNESS_BREAKS) - 1)
}

n_maturity_buckets = len(MATURITY_BREAKS) - 1
n_moneyness_buckets = len(MONEYNESS_BREAKS) - 1

print(f"Maturity buckets: {n_maturity_buckets}")
print(f"Moneyness buckets: {n_moneyness_buckets}")
print(f"Total joint buckets: {n_maturity_buckets * n_moneyness_buckets}")


def assign_maturity_bucket(maturity_col):
    expr = pl.lit(None, dtype=pl.Int32)
    for i in range(len(MATURITY_BREAKS) - 1):
        lower = MATURITY_BREAKS[i]
        upper = MATURITY_BREAKS[i + 1]
        bucket_id = i + 1
        expr = pl.when(
            (maturity_col >= lower) & (maturity_col < upper)
        ).then(pl.lit(bucket_id)).otherwise(expr)
    return expr


def assign_moneyness_bucket(moneyness_col):
    expr = pl.lit(None, dtype=pl.Int32)
    for i in range(len(MONEYNESS_BREAKS) - 1):
        lower = MONEYNESS_BREAKS[i]
        upper = MONEYNESS_BREAKS[i + 1]
        bucket_id = i + 1
        expr = pl.when(
            (moneyness_col >= lower) & (moneyness_col < upper)
        ).then(pl.lit(bucket_id)).otherwise(expr)
    return expr


def maturity_midpoint_expr(bucket_col):
    expr = pl.lit(None, dtype=pl.Float64)
    for bucket_id, mid in maturity_bucket_midpoints.items():
        expr = pl.when(bucket_col == bucket_id).then(pl.lit(mid)).otherwise(expr)
    return expr


def moneyness_midpoint_expr(bucket_col):
    expr = pl.lit(None, dtype=pl.Float64)
    for bucket_id, mid in moneyness_bucket_midpoints.items():
        expr = pl.when(bucket_col == bucket_id).then(pl.lit(mid)).otherwise(expr)
    return expr


src_cols = ["DATE", "IV", "DELTA", "MATURITY", "MONEYNESS"]
data = pl.read_parquet(src, columns=src_cols)

data = data.with_columns([
    pl.col("IV").log(base=math.e).alias("logIV"),
    (pl.col("MATURITY") / 255.0).alias("maturity"),
    assign_maturity_bucket(pl.col("MATURITY")).alias("MATURITY_BUCKET"),
    assign_moneyness_bucket(pl.col("DELTA")).alias("MONEYNESS_BUCKET"),
])

data = data.with_columns([
    maturity_midpoint_expr(pl.col("MATURITY_BUCKET")).alias("maturity_midpoint"),
    moneyness_midpoint_expr(pl.col("MONEYNESS_BUCKET")).alias("delta_midpoint"),
]).with_columns([
    (
        "mat" + pl.col("MATURITY_BUCKET").cast(pl.Int32).cast(pl.String)
        + "_mon" + pl.col("MONEYNESS_BUCKET").cast(pl.Int32).cast(pl.String)
    ).alias("joint_bucket"),
    (
        10.0 * (pl.col("delta_midpoint") - pl.col("DELTA")) ** 2
        + (pl.col("maturity_midpoint") - pl.col("MATURITY")) ** 2
    ).alias("closeness"),
])

data = data.filter(
    pl.col("MATURITY_BUCKET").is_not_null() & pl.col("MONEYNESS_BUCKET").is_not_null()
)

bucketed = (data
    .filter(
        pl.col("closeness") == pl.col("closeness").min().over(["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
    )
    .unique(subset=["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"], keep="first")
    .sort(["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
)
del data
gc.collect()

total_dates = bucketed["DATE"].n_unique()
coverage = (bucketed
    .group_by("joint_bucket")
    .agg(pl.col("DATE").n_unique().alias("n_dates"))
    .with_columns(
        (pl.col("n_dates") / total_dates).alias("coverage_fraction")
    )
    .sort("joint_bucket")
)

coverage_path = Path(out) / "bucket_coverage_fine.parquet"
coverage.write_parquet(str(coverage_path))

well_covered_buckets = coverage.filter(
    pl.col("coverage_fraction") >= MINIMUM_COVERAGE_FRACTION
)["joint_bucket"].to_list()

print(f"\nTotal dates: {total_dates}")
print(f"Total joint buckets populated: {coverage.height}")
print(f"Buckets with >= {MINIMUM_COVERAGE_FRACTION*100:.0f}% coverage: {len(well_covered_buckets)}")
print(f"Buckets below threshold (will create NaNs in matrix): "
      f"{coverage.height - len(well_covered_buckets)}")

logiv_matrix = (bucketed
    .filter(pl.col("joint_bucket").is_in(well_covered_buckets))
    .select(["DATE", "joint_bucket", "logIV"])
    .pivot(on="joint_bucket", index="DATE", values="logIV")
    .sort("DATE")
)

sorted_bucket_cols = sorted([c for c in logiv_matrix.columns if c != "DATE"])
logiv_matrix = logiv_matrix.select(["DATE"] + sorted_bucket_cols)
logiv_matrix.write_parquet(out + "bucket_matrix_fine.parquet")

coverage_summary = (coverage
    .with_columns([
        pl.col("joint_bucket").str.extract(r"mat(\d+)_mon(\d+)", 1)
          .cast(pl.Int32).alias("mat_bucket"),
        pl.col("joint_bucket").str.extract(r"mat(\d+)_mon(\d+)", 2)
          .cast(pl.Int32).alias("mon_bucket"),
    ])
    .sort(["mat_bucket", "mon_bucket"])
)

print("\nCoverage by bucket (fraction of dates):")
print(coverage_summary.select(["joint_bucket", "n_dates", "coverage_fraction"]))

maturity_coverage = (coverage_summary
    .group_by("mat_bucket")
    .agg(
        pl.col("coverage_fraction").mean().alias("mean_coverage"),
        pl.col("coverage_fraction").min().alias("min_coverage"),
    )
    .sort("mat_bucket")
)

moneyness_coverage = (coverage_summary
    .group_by("mon_bucket")
    .agg(
        pl.col("coverage_fraction").mean().alias("mean_coverage"),
        pl.col("coverage_fraction").min().alias("min_coverage"),
    )
    .sort("mon_bucket")
)

print("\nCoverage by maturity bucket:")
for row in maturity_coverage.iter_rows(named=True):
    lower = MATURITY_BREAKS[row["mat_bucket"] - 1]
    upper = MATURITY_BREAKS[row["mat_bucket"]]
    mid = maturity_bucket_midpoints[row["mat_bucket"]]
    print(f"  mat{row['mat_bucket']} [{lower}-{upper}d, mid={mid:.1f}]: "
          f"mean={row['mean_coverage']:.2f}, min={row['min_coverage']:.2f}")

print("\nCoverage by moneyness bucket:")
for row in moneyness_coverage.iter_rows(named=True):
    lower = MONEYNESS_BREAKS[row["mon_bucket"] - 1]
    upper = MONEYNESS_BREAKS[row["mon_bucket"]]
    mid = moneyness_bucket_midpoints[row["mon_bucket"]]
    print(f"  mon{row['mon_bucket']} [{lower:.2f},{upper:.2f}, mid={mid:.3f}]: "
          f"mean={row['mean_coverage']:.2f}, min={row['min_coverage']:.2f}")

print(f"\nFine bucketed matrix saved to: {out}bucket_matrix_fine.parquet")
print(f"Coverage report saved to: {coverage_path}")
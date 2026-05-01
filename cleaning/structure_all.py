import math
import polars as pl
import gc

def main():
    subfolder = "SPX"
    src = "data/" + subfolder + "/otm/filtered.parquet"
    out = "data/" + subfolder + "/otm/"

    n_mon_buckets = int(
        pl.scan_parquet(src).select(pl.col("MONEYNESS_BUCKET").max()).collect().item()
    )

    (pl.scan_parquet(src)
        .with_columns([
            pl.col("IV").log(base=math.e).alias("logIV"),
            pl.lit(1.0).alias("level"),
            pl.col("MONEYNESS").alias("moneyness"),
            (pl.col("MATURITY") / 255.0).alias("maturity"),
            (
                (pl.col("MATURITY_BUCKET").cast(pl.Int32) - 1) * n_mon_buckets
                + (pl.col("MONEYNESS_BUCKET").cast(pl.Int32) - 1)
            ).alias("bucket_idx"),
        ])
        .select(["DATE", "logIV", "level", "moneyness", "maturity", "bucket_idx"])
        .sink_parquet(out + "full.parquet")
    )

    (pl.scan_parquet(src)
        .select(["DATE", "UNDERLYING_LAST"])
        .unique(subset=["DATE"], keep="first")
        .sort("DATE")
        .sink_parquet(out + "underlying.parquet")
    )

    bucket_src_cols = ["DATE", "IV", "DELTA", "MATURITY", "MATURITY_BUCKET", "MONEYNESS_BUCKET", "MONEYNESS"]
    data = pl.read_parquet(src, columns=bucket_src_cols)

    data = data.with_columns([
        pl.col("IV").log(base=math.e).alias("logIV"),
        pl.lit(1.0).alias("level"),
        pl.col("MONEYNESS").alias("moneyness"),
        (pl.col("MATURITY") / 255.0).alias("maturity"),
        pl.when(pl.col("MATURITY_BUCKET") == 1).then((7.0 + 45.0) / 2.0)
          .when(pl.col("MATURITY_BUCKET") == 2).then((45.0 + 90.0) / 2.0)
          .when(pl.col("MATURITY_BUCKET") == 3).then((90.0 + 180.0) / 2.0)
          .when(pl.col("MATURITY_BUCKET") == 4).then((180.0 + 360.0) / 2.0)
          .otherwise(None).alias("maturity_midpoint"),
        pl.when(pl.col("MONEYNESS_BUCKET") == 1).then((-0.125 + 0.0) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 2).then((-0.375 + -0.125) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 3).then((-0.5 + -0.375) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 4).then((0.375 + 0.5) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 5).then((0.125 + 0.375) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 6).then((0.0 + 0.125) / 2.0)
          .otherwise(None).alias("delta_midpoint"),
        ("mat" + pl.col("MATURITY_BUCKET").cast(pl.Int32).cast(pl.String)
             + "_mon" + pl.col("MONEYNESS_BUCKET").cast(pl.Int32).cast(pl.String)
        ).alias("joint_bucket"),
    ]).with_columns([
        (pl.col("moneyness") ** 2).alias("moneyness2"),
        (pl.col("moneyness") * pl.col("maturity")).alias("interaction"),
        (10.0 * (pl.col("delta_midpoint") - pl.col("DELTA")) ** 2
             + (pl.col("maturity_midpoint") - pl.col("MATURITY")) ** 2
        ).alias("closeness"),
    ])

    bucketed = (data
        .filter(
            pl.col("closeness") == pl.col("closeness").min().over(["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
        )
        .unique(subset=["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"], keep="first")
        .sort(["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
    )
    del data
    gc.collect()

    bj_dummies = bucketed.select(pl.col("joint_bucket")).to_dummies(separator="_").select(
        pl.exclude("joint_bucket_" + sorted(bucketed["joint_bucket"].drop_nulls().unique().to_list())[0])
    )
    bj_dummies = bj_dummies.rename({c: "bucket_" + c.split("joint_bucket_", 1)[1] for c in bj_dummies.columns})
    bj_dummies = bj_dummies.with_columns(pl.col(c).cast(pl.Boolean) for c in bj_dummies.columns)
    bjcol_list = bj_dummies.columns
    bucketed = pl.concat([bucketed, bj_dummies], how="horizontal")
    del bj_dummies
    gc.collect()

    bucket_model_cols = ["DATE", "logIV", "level", "moneyness", "moneyness2", "maturity", "interaction"] + bjcol_list
    bucketed.select(bucket_model_cols).write_parquet(out + "bucket.parquet")

    logiv_matrix = (bucketed
        .select(["DATE", "joint_bucket", "logIV"])
        .pivot(on="joint_bucket", index="DATE", values="logIV")
        .sort("DATE")
    )
    sorted_bucket_cols = sorted([c for c in logiv_matrix.columns if c != "DATE"])
    logiv_matrix.select(["DATE"] + sorted_bucket_cols).write_parquet(out + "bucket_matrix.parquet")

if __name__ == "__main__":
    main()

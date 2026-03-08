import math
import polars as pl
import gc

def main():
    """Creates factors for the models and creates bucketed data.

    Closeness is defined by the summed squared distance 
    for both delta and maturity, where we put ten times
    more weight on delta because the smaller values compared to maturity.

    From:
    van der Wel, M., Ozturk, S.R. and van Dijk, D.J.C. (2015).
    Dynamic Factor Models for the Volatility Surface. 
    SSRN Electronic Journal. doi:https://doi.org/10.2139/ssrn.2558018.
    """
    subfolder = "SPX"
    data = pl.read_parquet("data/"+ subfolder +"/put/filtered.parquet")

    data = data.with_columns([
        pl.col("P_IV").log(base=math.e).alias("logIV"),
        pl.lit(1.0).alias("level"),
        (pl.col("STRIKE") / pl.col("UNDERLYING_LAST")).alias("moneyness"),
        (pl.col("MATURITY") / 255.0).alias("maturity"),
        pl.when(pl.col("MATURITY_BUCKET") == 1).then((7.0 + 45.0) / 2.0)
          .when(pl.col("MATURITY_BUCKET") == 2).then((45.0 + 90.0) / 2.0)
          .when(pl.col("MATURITY_BUCKET") == 3).then((90.0 + 180.0) / 2.0)
          .when(pl.col("MATURITY_BUCKET") == 4).then((180.0 + 360.0) / 2.0)
          .otherwise(None).alias("maturity_midpoint"),
        pl.when(pl.col("MONEYNESS_BUCKET") == 1).then((-0.125 + 0.0) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 2).then((-0.375 + -0.125) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 3).then((-0.5 + -0.375) / 2.0)
          .when(pl.col("MONEYNESS_BUCKET") == 4).then((-1.0 + -0.5) / 2.0)
          .otherwise(None).alias("delta_midpoint"),
        ("mat" + pl.col("MATURITY_BUCKET").cast(pl.Int32).cast(pl.String)
             + "_mon" + pl.col("MONEYNESS_BUCKET").cast(pl.Int32).cast(pl.String)
        ).alias("joint_bucket"),
    ]).with_columns([
        (pl.col("moneyness") ** 2).alias("moneyness2"),
        (pl.col("moneyness") * pl.col("maturity")).alias("interaction"),
        (10.0 * (pl.col("delta_midpoint") - pl.col("P_DELTA"))**2
             + (pl.col("maturity_midpoint") - pl.col("MATURITY"))**2
        ).alias("closeness"),
    ])

    bj_dummies = data.select(pl.col("joint_bucket")).to_dummies(separator="_").select(
        pl.exclude("joint_bucket_" + sorted(data["joint_bucket"].drop_nulls().unique().to_list())[0])
    )
    bj_dummies = bj_dummies.rename({c: "bucket_" + c.split("joint_bucket_", 1)[1] for c in bj_dummies.columns})
    bjcol_list = bj_dummies.columns
    data = pl.concat([data, bj_dummies], how="horizontal")

    model_columns = ["DATE", "logIV", "level", "moneyness", "moneyness2", "maturity", "interaction"] + bjcol_list
    full_model = data.select(model_columns)
    full_model.write_parquet("data/"+ subfolder +"/put/full.parquet")
    del full_model
    gc.collect()

    data = (data
        .sort("closeness")
        .unique(subset=["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"], keep="first", maintain_order=True)
        .sort(["DATE", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
    )

    logiv_matrix = (data
        .select(["DATE", "joint_bucket", "logIV"])
        .pivot(on="joint_bucket", index="DATE", values="logIV")
        .sort("DATE")
    )
    sorted_bucket_cols = sorted([c for c in logiv_matrix.columns if c != "DATE"])
    logiv_matrix = logiv_matrix.select(["DATE"] + sorted_bucket_cols)
    logiv_matrix.write_parquet("data/"+ subfolder +"/put/bucket_matrix.parquet")
    del logiv_matrix
    gc.collect()

    data.select(model_columns).write_parquet("data/"+ subfolder +"/put/bucket.parquet")

if __name__ == "__main__":
    main()
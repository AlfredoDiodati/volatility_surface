import gc
import glob
import polars as pl
from pathlib import Path
import pyarrow.parquet as pq

def main():
    """
        Following from Bollen and Whaley (2004), moneyness filters are created based on Delta
        to account for volatility.
        price_LT005 removes too many observations so it is kept in
        TODO: check relative frequency of excluded observations, group by year
    """
    subfolder = "SPX"
    filtered_path = "data/" +subfolder+ "/put/checks.parquet"
    final_path    = "data/" +subfolder+ "/put/filtered.parquet"
    checks_path   = "data/" +subfolder+ "/put/checks/all.parquet"

    files = glob.glob("data/" +subfolder+ "/raw/*.txt")

    Path("data/" +subfolder+ "/put/").mkdir(parents=True, exist_ok=True)
    Path("data/" +subfolder+ "/put/checks/").mkdir(parents=True, exist_ok=True)

    filtered_writer = None
    final_writer = None
    put_checks_sum = None

    initial_select_columns = [
        "QUOTE_DATE", "EXPIRE_DATE", "P_IV", "P_LAST", "UNDERLYING_LAST", "STRIKE", "P_DELTA",
    ]

    filter_remove_columns = [
        "maturity_GT360", "maturity_LT7", "IV_LT005","IV_GT070", "MISSING_PRICE",
        "MISSING_DELTA", "MISSING_DATE", "MISSING_MATURITY", "MISSING_IV", "is_DELTAinvalid"
    ]

    checks_columns = [
        "TOT", "maturity_GT360", "maturity_LT7", "IV_LT005","IV_GT070", "price_LT005",
        "MISSING_DELTA", "MISSING_PRICE", "MISSING_DATE", "MISSING_MATURITY",
        "MISSING_UNDERLYING", "MISSING_STRIKE", "MISSING_IV","is_DOTM", "is_OTM",
        "is_ATM", "is_DATM","is_maturityLT45", "is_maturity_45_90", "is_maturity_90_180", "is_maturity_180",
        "is_DELTAinvalid", "is_REMOVED", "is_NOT_REMOVED"
    ]

    remove_columns = list(set(checks_columns) | set(filter_remove_columns) |
        set(["QUOTE_DATE", "EXPIRE_DATE"]))

    for f in files:
        merged = pl.read_csv(
            f, separator=",",
            schema_overrides={"QUOTE_DATE": pl.String, "EXPIRE_DATE": pl.String},
            truncate_ragged_lines=True
        )
        merged = merged.rename({c: c.strip().replace("[", "").replace("]", "") for c in merged.columns})

        missing = [c for c in initial_select_columns if c not in merged.columns]
        if missing:
            raise ValueError(f"{f}: missing columns after cleanup: {missing}")

        merged = merged.select(initial_select_columns)

        numeric_columns = ["P_IV", "P_LAST", "UNDERLYING_LAST", "STRIKE", "P_DELTA"]
        merged = merged.with_columns([
            pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
            if merged[c].dtype == pl.String else pl.col(c)
            for c in numeric_columns
        ])

        put_filtered = (
            merged
            .with_columns([
                pl.col("QUOTE_DATE").str.to_date(strict=False).alias("_qd"),
                pl.col("EXPIRE_DATE").str.to_date(strict=False).alias("_ed"),
            ])
            .with_columns([
                pl.col("_qd").dt.strftime("%Y%m%d").alias("DATE"),
                (pl.col("_ed") - pl.col("_qd")).dt.total_days().cast(pl.Float64).alias("MATURITY"),
            ])
            .drop(["_qd", "_ed"])
            .with_columns([
                pl.lit(1).alias("TOT"),
                pl.when(pl.col("MATURITY") <= 45.0).then(1)
                  .when((pl.col("MATURITY") > 45.0) & (pl.col("MATURITY") <= 90.0)).then(2)
                  .when((pl.col("MATURITY") > 90.0) & (pl.col("MATURITY") <= 180.0)).then(3)
                  .when(pl.col("MATURITY") > 180.0).then(4)
                  .otherwise(None).cast(pl.Float64).alias("MATURITY_BUCKET"),
                (pl.col("MATURITY") <= 45.0).alias("is_maturityLT45"),
                ((pl.col("MATURITY") > 45.0) & (pl.col("MATURITY") <= 90.0)).alias("is_maturity_45_90"),
                ((pl.col("MATURITY") > 90.0) & (pl.col("MATURITY") <= 180.0)).alias("is_maturity_90_180"),
                (pl.col("MATURITY") > 180.0).alias("is_maturity_180"),
                pl.when((-0.125 <= pl.col("P_DELTA")) & (pl.col("P_DELTA") < 0.0)).then(1)
                  .when((-0.375 <= pl.col("P_DELTA")) & (pl.col("P_DELTA") < -0.125)).then(2)
                  .when((-0.5 <= pl.col("P_DELTA")) & (pl.col("P_DELTA") < -0.375)).then(3)
                  .when(pl.col("P_DELTA") < -0.5).then(4)
                  .otherwise(None).cast(pl.Float64).alias("MONEYNESS_BUCKET"),
                ((-0.125 <= pl.col("P_DELTA")) & (pl.col("P_DELTA") < 0.0)).alias("is_DOTM"),
                ((-0.375 <= pl.col("P_DELTA")) & (pl.col("P_DELTA") < -0.125)).alias("is_OTM"),
                ((-0.5 <= pl.col("P_DELTA")) & (pl.col("P_DELTA") < -0.375)).alias("is_ATM"),
                (pl.col("P_DELTA") < -0.5).alias("is_DATM"),
                ((pl.col("P_DELTA") <= -1.0) | (pl.col("P_DELTA") >= 0.0)).alias("is_DELTAinvalid"),
                (pl.col("MATURITY") > 360.0).alias("maturity_GT360"),
                (pl.col("MATURITY") < 7.0).alias("maturity_LT7"),
                (pl.col("P_IV") < 0.05).alias("IV_LT005"),
                (pl.col("P_IV") > 0.70).alias("IV_GT070"),
                (pl.col("P_LAST") < 0.05).alias("price_LT005"),
                pl.col("P_DELTA").is_null().alias("MISSING_DELTA"),
                pl.col("P_LAST").is_null().alias("MISSING_PRICE"),
                pl.col("DATE").is_null().alias("MISSING_DATE"),
                pl.col("MATURITY").is_null().alias("MISSING_MATURITY"),
                pl.col("UNDERLYING_LAST").is_null().alias("MISSING_UNDERLYING"),
                pl.col("STRIKE").is_null().alias("MISSING_STRIKE"),
                pl.col("P_IV").is_null().alias("MISSING_IV"),
            ])
            .with_columns([
                pl.any_horizontal(*[pl.col(c) for c in filter_remove_columns]).alias("is_REMOVED"),
            ])
            .with_columns([
                (~pl.col("is_REMOVED")).alias("is_NOT_REMOVED"),
            ])
        )

        put_final = put_filtered.filter(~pl.col("is_REMOVED")).drop(
            [c for c in remove_columns if c in put_filtered.columns]
        )

        put_checks = put_filtered.select(checks_columns).sum()
        if put_checks_sum is None: put_checks_sum = put_checks
        else: put_checks_sum = pl.concat([put_checks_sum, put_checks]).sum()

        table_filtered = put_filtered.to_arrow()
        if filtered_writer is None:
            filtered_writer = pq.ParquetWriter(filtered_path, table_filtered.schema)
        filtered_writer.write_table(table_filtered)

        table_final = put_final.to_arrow()
        if final_writer is None:
            final_writer = pq.ParquetWriter(final_path, table_final.schema)
        final_writer.write_table(table_final)

        del merged, put_filtered, put_final, put_checks, table_filtered, table_final
        gc.collect()

    if filtered_writer is not None: filtered_writer.close()
    if final_writer is not None: final_writer.close()

    pl.read_parquet(filtered_path).sort("DATE").write_parquet(filtered_path)
    pl.read_parquet(final_path).sort("DATE").write_parquet(final_path)

    put_checks_sum.write_parquet(checks_path)

if __name__ == "__main__":
    main()
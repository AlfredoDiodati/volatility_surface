import gc
import glob
import polars as pl
from pathlib import Path
import pyarrow.parquet as pq

def make_option_side(df, side):
    prefix = "P_" if side == "put" else "C_"
    return (
        df
        .select([
            "QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE",
            pl.col(prefix + "IV").alias("IV"),
            pl.col(prefix + "LAST").alias("LAST"),
            pl.col(prefix + "DELTA").alias("DELTA"),
        ])
        .with_columns(pl.lit(side).alias("SIDE"))
    )

def main():
    subfolder = "SPX"
    filtered_path = "data/" + subfolder + "/otm/checks.parquet"
    final_path    = "data/" + subfolder + "/otm/filtered.parquet"
    checks_path   = "data/" + subfolder + "/otm/checks/all.parquet"

    files = sorted(glob.glob("data/" + subfolder + "/raw/*.txt"))

    Path("data/" + subfolder + "/otm/").mkdir(parents=True, exist_ok=True)
    Path("data/" + subfolder + "/otm/checks/").mkdir(parents=True, exist_ok=True)

    filtered_writer = None
    final_writer = None
    checks_sum = None

    raw_select_columns = [
        "QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE",
        "P_IV", "P_LAST", "P_DELTA",
        "C_IV", "C_LAST", "C_DELTA",
    ]

    filter_remove_columns = [
        "maturity_GT360", "maturity_LT7",
        "IV_LT005", "IV_GT070",
        "MISSING_PRICE", "MISSING_DELTA", "MISSING_DATE",
        "MISSING_MATURITY", "MISSING_IV",
        "is_not_OTM",
    ]

    checks_columns = [
        "TOT",
        "maturity_GT360", "maturity_LT7",
        "IV_LT005", "IV_GT070", "price_LT005",
        "MISSING_DELTA", "MISSING_PRICE", "MISSING_DATE",
        "MISSING_MATURITY", "MISSING_UNDERLYING", "MISSING_STRIKE", "MISSING_IV",
        "is_not_OTM",
        "is_DOTM_put", "is_OTM_put", "is_ATM_put",
        "is_ATM_call", "is_OTM_call", "is_DOTM_call",
        "is_maturityLT45", "is_maturity_45_90", "is_maturity_90_180", "is_maturity_180",
        "is_REMOVED", "is_NOT_REMOVED",
    ]

    remove_columns = list(set(checks_columns) | set(filter_remove_columns) |
        set(["QUOTE_DATE", "EXPIRE_DATE"]))

    for f in files:
        raw = pl.read_csv(
            f, separator=",",
            schema_overrides={"QUOTE_DATE": pl.String, "EXPIRE_DATE": pl.String},
            truncate_ragged_lines=True
        )
        raw = raw.rename({c: c.strip().replace("[", "").replace("]", "") for c in raw.columns})

        missing = [c for c in raw_select_columns if c not in raw.columns]
        if missing:
            raise ValueError(f"{f}: missing columns after cleanup: {missing}")

        raw = raw.select(raw_select_columns)

        numeric_columns = [
            "P_IV", "P_LAST", "P_DELTA", "C_IV", "C_LAST", "C_DELTA",
            "UNDERLYING_LAST", "STRIKE",
        ]
        raw = raw.with_columns([
            pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
            if raw[c].dtype == pl.String else pl.col(c)
            for c in numeric_columns
        ])

        puts = make_option_side(raw, "put")
        calls = make_option_side(raw, "call")
        del raw
        merged = pl.concat([puts, calls])
        del puts, calls

        enriched = (
            merged
            .with_columns([
                pl.col("QUOTE_DATE").str.to_date(strict=False).alias("_qd"),
                pl.col("EXPIRE_DATE").str.to_date(strict=False).alias("_ed"),
            ])
            .with_columns([
                pl.col("_qd").dt.strftime("%Y%m%d").alias("DATE"),
                (pl.col("_ed") - pl.col("_qd")).dt.total_days().cast(pl.Float64).alias("MATURITY"),
                (pl.col("STRIKE") / pl.col("UNDERLYING_LAST")).alias("MONEYNESS"),
            ])
            .drop(["_qd", "_ed"])
            .with_columns([
                pl.lit(1).alias("TOT"),

                # OTM condition: puts with -0.5 < delta < 0, calls with 0 < delta < 0.5
                pl.when(pl.col("SIDE") == "put")
                    .then((pl.col("DELTA") <= -0.5) | (pl.col("DELTA") >= 0.0))
                    .otherwise((pl.col("DELTA") >= 0.5) | (pl.col("DELTA") <= 0.0))
                    .alias("is_not_OTM"),

                # maturity buckets
                pl.when(pl.col("MATURITY") <= 45.0).then(1)
                  .when((pl.col("MATURITY") > 45.0) & (pl.col("MATURITY") <= 90.0)).then(2)
                  .when((pl.col("MATURITY") > 90.0) & (pl.col("MATURITY") <= 180.0)).then(3)
                  .when(pl.col("MATURITY") > 180.0).then(4)
                  .otherwise(None).cast(pl.Float64).alias("MATURITY_BUCKET"),
                (pl.col("MATURITY") <= 45.0).alias("is_maturityLT45"),
                ((pl.col("MATURITY") > 45.0) & (pl.col("MATURITY") <= 90.0)).alias("is_maturity_45_90"),
                ((pl.col("MATURITY") > 90.0) & (pl.col("MATURITY") <= 180.0)).alias("is_maturity_90_180"),
                (pl.col("MATURITY") > 180.0).alias("is_maturity_180"),

                # 6 moneyness buckets following the paper's delta breakpoints
                pl.when((-0.125 < pl.col("DELTA")) & (pl.col("DELTA") < 0.0)).then(1)
                  .when((-0.375 < pl.col("DELTA")) & (pl.col("DELTA") <= -0.125)).then(2)
                  .when((-0.5 < pl.col("DELTA")) & (pl.col("DELTA") <= -0.375)).then(3)
                  .when((0.375 <= pl.col("DELTA")) & (pl.col("DELTA") < 0.5)).then(4)
                  .when((0.125 <= pl.col("DELTA")) & (pl.col("DELTA") < 0.375)).then(5)
                  .when((0.0 < pl.col("DELTA")) & (pl.col("DELTA") < 0.125)).then(6)
                  .otherwise(None).cast(pl.Float64).alias("MONEYNESS_BUCKET"),
                ((-0.125 < pl.col("DELTA")) & (pl.col("DELTA") < 0.0)).alias("is_DOTM_put"),
                ((-0.375 < pl.col("DELTA")) & (pl.col("DELTA") <= -0.125)).alias("is_OTM_put"),
                ((-0.5 < pl.col("DELTA")) & (pl.col("DELTA") <= -0.375)).alias("is_ATM_put"),
                ((0.375 <= pl.col("DELTA")) & (pl.col("DELTA") < 0.5)).alias("is_ATM_call"),
                ((0.125 <= pl.col("DELTA")) & (pl.col("DELTA") < 0.375)).alias("is_OTM_call"),
                ((0.0 < pl.col("DELTA")) & (pl.col("DELTA") < 0.125)).alias("is_DOTM_call"),

                # filter flags
                (pl.col("MATURITY") > 360.0).alias("maturity_GT360"),
                (pl.col("MATURITY") < 7.0).alias("maturity_LT7"),
                (pl.col("IV") < 0.05).alias("IV_LT005"),
                (pl.col("IV") > 0.70).alias("IV_GT070"),
                (pl.col("LAST") < 0.05).alias("price_LT005"),
                pl.col("DELTA").is_null().alias("MISSING_DELTA"),
                pl.col("LAST").is_null().alias("MISSING_PRICE"),
                pl.col("DATE").is_null().alias("MISSING_DATE"),
                pl.col("MATURITY").is_null().alias("MISSING_MATURITY"),
                pl.col("UNDERLYING_LAST").is_null().alias("MISSING_UNDERLYING"),
                pl.col("STRIKE").is_null().alias("MISSING_STRIKE"),
                pl.col("IV").is_null().alias("MISSING_IV"),
            ])
            .with_columns(
                pl.any_horizontal(*[pl.col(c) for c in filter_remove_columns]).alias("is_REMOVED"),
            )
            .with_columns(
                (~pl.col("is_REMOVED")).alias("is_NOT_REMOVED"),
            )
        )
        del merged

        final = enriched.filter(~pl.col("is_REMOVED")).drop(
            [c for c in remove_columns if c in enriched.columns]
        )

        checks = enriched.select(checks_columns).sum()
        if checks_sum is None: checks_sum = checks
        else: checks_sum = pl.concat([checks_sum, checks]).sum()
        del checks

        table_filtered = enriched.to_arrow()
        del enriched
        if filtered_writer is None:
            filtered_writer = pq.ParquetWriter(filtered_path, table_filtered.schema)
        filtered_writer.write_table(table_filtered)
        del table_filtered

        table_final = final.to_arrow()
        del final
        if final_writer is None:
            final_writer = pq.ParquetWriter(final_path, table_final.schema)
        final_writer.write_table(table_final)
        del table_final

        gc.collect()

    if filtered_writer is not None: filtered_writer.close()
    if final_writer is not None: final_writer.close()

    checks_sum.write_parquet(checks_path)


if __name__ == "__main__":
    main()
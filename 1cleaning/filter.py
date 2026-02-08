import gc
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
from pathlib import Path
import pyarrow.parquet as pq

def main():
    """
        Following from Bollen and Whaley (2004), moneyness filters are created based on Delta
        to account for volatility.
        price_LT005 removes too many observations so it is kept in
        TODO: check relative frequency of excluded observations, group by year
        create new buckets for DATM
    """
    subfolder = "SPY"
    filtered_path = "data/" +subfolder+ "/put/checks.parquet"
    final_path    = "data/" +subfolder+ "/put/filtered.parquet"
    checks_path   = "data/" +subfolder+ "/put/checks/all.parquet"

    files = glob.glob("data/" +subfolder+ "/raw/*.txt")

    Path("data/" +subfolder+ "/put/").mkdir(parents=True, exist_ok=True)

    filtered_writer = None
    final_writer = None
    put_checks_sum = None

    initial_select_columns = [
        "QUOTE_DATE", "EXPIRE_DATE", "P_IV", "P_LAST", "UNDERLYING_LAST", "STRIKE", "P_DELTA", 
    ]

    # On hold columns: "MISSING_STRIKE" "MISSING_UNDERLYING" "price_LT005"

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

    remove_columns =  list(set(checks_columns) | set(filter_remove_columns) |
        set(["QUOTE_DATE", "EXPIRE_DATE"]))

    for f in files:
        merged = pd.read_csv(f, sep=",", skipinitialspace=True)
        merged.columns = merged.columns.str.replace(r"[\[\]]", "", regex=True)

        missing = [c for c in initial_select_columns if c not in merged.columns]
        if missing:
            raise ValueError(f"{f}: missing columns after cleanup: {missing}")

        merged = merged[initial_select_columns]

        merged.columns = merged.columns.str.replace(r"[\[\]]", "", regex=True)
        quote_date = pd.to_datetime(merged["QUOTE_DATE"], errors="coerce")
        expire_date = pd.to_datetime(merged["EXPIRE_DATE"], errors="coerce")

        put_filtered = merged.assign(
            DATE=quote_date.dt.strftime("%Y%m%d"),
            MATURITY=(expire_date - quote_date).dt.days,
        ).assign(
            TOT = 1,
            MATURITY_BUCKET=lambda x: np.select([
                x["MATURITY"] <= 45.0,
                (x["MATURITY"] > 45.0) & (x["MATURITY"] <= 90.0),
                (x["MATURITY"] > 90.0) & (x["MATURITY"] <= 180.0),
                x["MATURITY"] > 180.0],
                [1, 2, 3, 4],
                default=np.nan
            ),
            is_maturityLT45 = lambda x: x["MATURITY"] <= 45.0,
            is_maturity_45_90= lambda x: (x["MATURITY"] > 45.0) & (x["MATURITY"] <= 90.0),
            is_maturity_90_180= lambda x:(x["MATURITY"] > 90.0) & (x["MATURITY"] <= 180.0),
            is_maturity_180= lambda x: x["MATURITY"] > 180.0,
            MONEYNESS_BUCKET = lambda x: np.select(
                [((-0.125 <= x["P_DELTA"]) & (0.0 > x["P_DELTA"])),
                ((-0.375 <= x["P_DELTA"]) & (-0.125 > x["P_DELTA"])),
                ((-0.5 <= x["P_DELTA"]) & (-0.375 > x["P_DELTA"])),
                ((-0.5 > x["P_DELTA"]))],
                [1, 2, 3, 4],
                default=np.nan
            ),
            is_DOTM=lambda x: ((-0.125 <= x["P_DELTA"]) & (0.0 > x["P_DELTA"])),
            is_OTM=lambda x: ((-0.375 <= x["P_DELTA"]) & (-0.125 > x["P_DELTA"])),
            is_ATM=lambda x: ((-0.5 <= x["P_DELTA"]) & (-0.375 > x["P_DELTA"])),
            is_DATM=lambda x: ((-0.5 > x["P_DELTA"])),
            is_DELTAinvalid = lambda x: ((-1.0 >= x["P_DELTA"]) | (0.0 <= x["P_DELTA"])),
            maturity_GT360=lambda x: (x["MATURITY"] > 360.0),
            maturity_LT7=lambda x: (x["MATURITY"] < 7.0),
            IV_LT005=lambda x: (x["P_IV"] < 0.05),
            IV_GT070=lambda x: (x["P_IV"] > 0.70),
            price_LT005 = lambda x: (x["P_LAST"] < 0.05),
            MISSING_DELTA=lambda x: x["P_DELTA"].isna(),
            MISSING_PRICE=lambda x: x["P_LAST"].isna(),
            MISSING_DATE=lambda x: x["DATE"].isna(),
            MISSING_MATURITY=lambda x: x["MATURITY"].isna(),
            MISSING_UNDERLYING=lambda x: x["UNDERLYING_LAST"].isna(),
            MISSING_STRIKE=lambda x: x["STRIKE"].isna(),
            MISSING_IV=lambda x: x["P_IV"].isna()
        ).assign(
            is_REMOVED = lambda x: x[filter_remove_columns].any(axis=1),
            is_NOT_REMOVED = lambda x: ~x[filter_remove_columns].any(axis=1)
        )
        put_final = put_filtered[(put_filtered[filter_remove_columns] == 0).all(axis=1)
            ].drop(columns=remove_columns)
        put_checks = put_filtered[checks_columns].sum()
        if put_checks_sum is None: put_checks_sum = put_checks
        else: put_checks_sum = put_checks_sum.add(put_checks, fill_value=0)

        table_filtered = pa.Table.from_pandas(put_filtered, preserve_index=False)
        if filtered_writer is None:
            filtered_writer = pq.ParquetWriter(filtered_path, table_filtered.schema)
        filtered_writer.write_table(table_filtered)

        table_final = pa.Table.from_pandas(put_final, preserve_index=False)
        if final_writer is None:
            final_writer = pq.ParquetWriter(final_path, table_final.schema)
        final_writer.write_table(table_final)

        del merged, quote_date, expire_date, put_filtered, put_final, put_checks, table_filtered, table_final
        gc.collect()

    if filtered_writer is not None: filtered_writer.close()
    if final_writer is not None: final_writer.close()

    put_checks = put_checks_sum.to_frame(name="value").T
    put_checks.to_parquet(checks_path)

if __name__ == "__main__":
    main()
import gc
from pathlib import Path
import pandas as pd

def main():
    """For all .parquet files in data/put/ directory and not subfolders,
        creates a .csv copy of the first 20 observations in data/pu/head/ with the same name
    .parquet is convenient for storing high volume data, but not quick to directly open to see the structure of 
    a dataset (columns, dtypes ecc...), therefore it is convenient to have both.

    Once the head is stored, the full file is removed from memory.
    """
    subfolder = "SPY"
    folder = Path("data/"+subfolder+"/put/")
    out_dir = folder / "head"
    out_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in folder.glob("*.parquet"):
        csv_path = out_dir / f"{parquet_path.stem}.csv"
        if csv_path.exists():
            continue

        try:
            df_head = pd.read_parquet(parquet_path, engine="pyarrow").head(20).copy()
        except Exception as e:
            print(f"SKIP (read error): {parquet_path.name} -> {e}")
            continue

        df_head.to_csv(csv_path, index=False)

        del df_head
        gc.collect()

        print(f"WROTE: {csv_path}")

if __name__ == "__main__":
    main()

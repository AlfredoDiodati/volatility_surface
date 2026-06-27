import pandas as pd

df = pd.read_parquet("data/SPX/otm/full.parquet")

date_col = next(c for c in ["DATE", "date", "Date", "trade_date", "quotedate"] if c in df.columns)
days = df[date_col].nunique()
n = len(df)

print(f"Observations : {n:,}")
print(f"Days         : {days:,}")
print(f"Avg per day  : {n / days:.1f}")

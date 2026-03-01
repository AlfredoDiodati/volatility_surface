from __future__ import annotations
from pathlib import Path
import polars as pl

DATASET_PATH = Path("data/SPY/put/bucket.parquet")
OUTPUT_PATH = Path("data/SPY/put/averages.parquet")
BASELINE_MEMBER_LABEL = "baseline"

def _load_dataset(dataset_path: Path) -> pl.DataFrame:
    if dataset_path.suffix.lower() == ".parquet":
        return pl.read_parquet(dataset_path)
    if dataset_path.suffix.lower() == ".csv":
        return pl.read_csv(dataset_path)
    raise ValueError(f"Unsupported input format: {dataset_path.suffix}")

def _detect_date_column_name(dataset: pl.DataFrame) -> str:
    if "DATE" in dataset.columns:
        return "DATE"
    if "date" in dataset.columns:
        return "date"
    for candidate in dataset.columns:
        if "date" in candidate.lower():
            return candidate

    for candidate, dtype in zip(dataset.columns, dataset.dtypes):
        if dtype in (pl.Date, pl.Datetime, pl.Time):
            return candidate

    raise ValueError("Could not detect a date column. Expected a column named DATE/date or a Date/Datetime column.")

def _standardize_date_column(dataset: pl.DataFrame, date_column_name: str) -> pl.DataFrame:
    date_dtype = dataset.schema[date_column_name]

    if date_dtype == pl.Date:
        return dataset

    if date_dtype == pl.Datetime:
        return dataset.with_columns(pl.col(date_column_name).cast(pl.Date))

    if date_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return dataset.with_columns(
            pl.col(date_column_name)
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .alias(date_column_name)
        )

    if date_dtype == pl.Utf8:
        parsed_as_yyyymmdd = pl.col(date_column_name).str.strptime(pl.Date, format="%Y%m%d", strict=False)
        parsed_as_iso = pl.col(date_column_name).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        return dataset.with_columns(pl.coalesce([parsed_as_yyyymmdd, parsed_as_iso]).alias(date_column_name))

    return dataset.with_columns(pl.col(date_column_name).cast(pl.Date))

def _detect_boolean_column_names(dataset: pl.DataFrame) -> list[str]:
    boolean_column_names: list[str] = []
    for column_name, dtype in zip(dataset.columns, dataset.dtypes):
        if dtype == pl.Boolean:
            boolean_column_names.append(column_name)
    if len(boolean_column_names) > 0:
        return boolean_column_names

    guessed_boolean_column_names: list[str] = []
    for column_name in dataset.columns:
        if column_name.lower().startswith("bucket_"):
            guessed_boolean_column_names.append(column_name)

    if len(guessed_boolean_column_names) == 0:
        raise ValueError("No boolean columns detected (and no bucket_* columns found).")

    return guessed_boolean_column_names

def _add_panel_member_column(dataset: pl.DataFrame, boolean_column_names: list[str]) -> pl.DataFrame:
    boolean_label_expressions: list[pl.Expr] = []
    for boolean_column_name in boolean_column_names:
        boolean_label_expressions.append(pl.when(pl.col(boolean_column_name)).then(pl.lit(boolean_column_name)))

    panel_member_expression = (
        pl.coalesce(boolean_label_expressions + [pl.lit(BASELINE_MEMBER_LABEL)]).alias("panel_member")
    )

    dataset_with_member = dataset.with_columns(panel_member_expression)

    if len(boolean_column_names) > 0 and dataset.schema.get(boolean_column_names[0]) == pl.Boolean:
        true_count_expression_terms: list[pl.Expr] = []
        for boolean_column_name in boolean_column_names:
            true_count_expression_terms.append(pl.col(boolean_column_name).cast(pl.Int8))
        max_true_count = dataset_with_member.select(pl.sum_horizontal(true_count_expression_terms).max()).item()
        if max_true_count is not None and max_true_count > 1:
            raise ValueError("At least one row has more than one boolean column set to True.")

    return dataset_with_member

def main() -> None:
    dataset = _load_dataset(DATASET_PATH)
    date_column_name = _detect_date_column_name(dataset)
    dataset = _standardize_date_column(dataset, date_column_name)

    boolean_column_names = _detect_boolean_column_names(dataset)
    dataset_with_member = _add_panel_member_column(dataset, boolean_column_names)

    excluded_column_names = set(boolean_column_names)
    excluded_column_names.add("level")
    excluded_column_names.add("panel_member")
    excluded_column_names.add(date_column_name)

    value_column_names: list[str] = []
    for column_name, dtype in zip(dataset_with_member.columns, dataset_with_member.dtypes):
        if column_name in excluded_column_names:
            continue
        if dtype in (
            pl.Float64,
            pl.Float32,
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.UInt64,
            pl.UInt32,
            pl.UInt16,
            pl.UInt8,
        ):
            value_column_names.append(column_name)

    value_column_names.sort()

    averaged = (
        dataset_with_member
        .group_by("panel_member")
        .agg([pl.col(column_name).mean().alias(column_name) for column_name in value_column_names])
        .sort("panel_member")
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    averaged.write_parquet(OUTPUT_PATH)

if __name__ == "__main__":
    main()
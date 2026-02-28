from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = Path("data/SPY/put/bucket.parquet")
OUTPUT_DIR = Path("plot/SPY/put/ergod")
BASELINE_MEMBER_LABEL = "baseline"
CONFIDENCE_Z = 1.96

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

    column_name = None
    for candidate in dataset.columns:
        if "date" in candidate.lower():
            column_name = candidate
            break

    if column_name is not None:
        return column_name

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

def _plot_column_panels(
    dataset_with_member: pl.DataFrame,
    date_column_name: str,
    value_column_name: str,
    output_dir: Path,
) -> None:
    plotting_frame = (
        dataset_with_member.select([date_column_name, "panel_member", value_column_name])
        .sort([date_column_name, "panel_member"])
        .to_pandas()
    )

    pivoted = plotting_frame.pivot(index=date_column_name, columns="panel_member", values=value_column_name)
    pivoted = pivoted.sort_index()

    panel_member_names = list(pivoted.columns)
    if BASELINE_MEMBER_LABEL in panel_member_names:
        panel_member_names.remove(BASELINE_MEMBER_LABEL)
        panel_member_names.sort()
        panel_member_names.insert(0, BASELINE_MEMBER_LABEL)
        pivoted = pivoted[panel_member_names]
    else:
        panel_member_names.sort()
        pivoted = pivoted[panel_member_names]

    number_of_members = len(panel_member_names)
    number_of_columns = int(np.ceil(np.sqrt(number_of_members)))
    number_of_rows = int(np.ceil(number_of_members / number_of_columns))

    figure_width = max(10.0, 4.0 * number_of_columns)
    figure_height = max(6.0, 2.8 * number_of_rows)

    fig, axes = plt.subplots(number_of_rows, number_of_columns, figsize=(figure_width, figure_height), sharex=True)
    if isinstance(axes, np.ndarray):
        axes_flat = axes.ravel()
    else:
        axes_flat = np.array([axes])

    x_values = pivoted.index.to_pydatetime()

    legend_handles = None
    legend_labels = None

    member_index = 0
    while member_index < number_of_members:
        member_name = panel_member_names[member_index]
        axis = axes_flat[member_index]

        member_series = pivoted[member_name].astype(float).to_numpy()
        valid_mask = np.isfinite(member_series)

        if np.any(valid_mask):
            mean_value = float(np.nanmean(member_series))
            standard_deviation_value = float(np.nanstd(member_series, ddof=1)) if np.sum(valid_mask) > 1 else 0.0
            ci_lower_value = mean_value - CONFIDENCE_Z * standard_deviation_value
            ci_upper_value = mean_value + CONFIDENCE_Z * standard_deviation_value

            line_series, = axis.plot(x_values, member_series)
            line_mean = axis.axhline(mean_value, linestyle="--", linewidth=1.0)
            band = axis.fill_between(x_values, ci_lower_value, ci_upper_value, alpha=0.2)

            if legend_handles is None:
                legend_handles = [line_series, line_mean, band]
                legend_labels = ["series", "mean", f"± {CONFIDENCE_Z:.2f}·sd"]

        axis.set_title(str(member_name).replace("bucket_", ""))
        member_index += 1

    axis_index = number_of_members
    while axis_index < len(axes_flat):
        axes_flat[axis_index].set_visible(False)
        axis_index += 1

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3)

    fig.suptitle(value_column_name, y=0.995)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.98])

    safe_filename = value_column_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{safe_filename}.pdf", format="pdf")
    plt.close(fig)

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

    for value_column_name in value_column_names:
        _plot_column_panels(
            dataset_with_member=dataset_with_member,
            date_column_name=date_column_name,
            value_column_name=value_column_name,
            output_dir=OUTPUT_DIR,
        )

if __name__ == "__main__":
    main()
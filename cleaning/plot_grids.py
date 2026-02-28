from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

    def _build_figure(pivoted_frame, figure_title):
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

        x_values = pivoted_frame.index.to_pydatetime()

        legend_handles = None
        legend_labels = None

        member_index = 0
        while member_index < number_of_members:
            member_name = panel_member_names[member_index]
            axis = axes_flat[member_index]

            member_series = pivoted_frame[member_name].astype(float).to_numpy()
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

        fig.suptitle(figure_title, y=0.995)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.98])

        return fig

    def _build_rolling_std_figure(pivoted_diff_frame, figure_title):
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

        x_values = pivoted_diff_frame.index.to_pydatetime()

        rolling_windows = [30, 90, 252]
        legend_handles = []
        legend_labels = []

        member_index = 0
        while member_index < len(panel_member_names):
            member_name = panel_member_names[member_index]
            axis = axes_flat[member_index]

            rolling_index = 0
            while rolling_index < len(rolling_windows):
                window_length = rolling_windows[rolling_index]
                rolling_standard_deviation = pivoted_diff_frame[member_name].rolling(window_length).std(ddof=1)

                line_handle, = axis.plot(x_values, rolling_standard_deviation.to_numpy())
                if member_index == 0:
                    legend_handles.append(line_handle)
                    legend_labels.append(f"roll std ({window_length})")

                rolling_index += 1

            axis.set_title(str(member_name).replace("bucket_", ""))
            member_index += 1

        axis_index = len(panel_member_names)
        while axis_index < len(axes_flat):
            axes_flat[axis_index].set_visible(False)
            axis_index += 1

        if len(legend_handles) > 0:
            fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3)

        fig.suptitle(figure_title, y=0.995)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.98])

        return fig

    def _build_first_difference_figure(pivoted_diff_frame, figure_title):
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

        x_values = pivoted_diff_frame.index.to_pydatetime()

        member_index = 0
        while member_index < len(panel_member_names):
            member_name = panel_member_names[member_index]
            axis = axes_flat[member_index]

            member_series = pivoted_diff_frame[member_name].astype(float).to_numpy()
            axis.plot(x_values, member_series)
            axis.axhline(0.0, linestyle="--", linewidth=1.0)

            axis.set_title(str(member_name).replace("bucket_", ""))
            member_index += 1

        axis_index = len(panel_member_names)
        while axis_index < len(axes_flat):
            axes_flat[axis_index].set_visible(False)
            axis_index += 1

        fig.suptitle(figure_title, y=0.995)
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.98])

        return fig

    def _save_asymmetric_mean_reversion_diagnostics(pivoted_frame, safe_filename_value):
        output_path_amr = output_dir / f"{safe_filename_value}_amr.pdf"

        with PdfPages(output_path_amr) as pdf:
            member_index = 0
            while member_index < len(panel_member_names):
                member_name = panel_member_names[member_index]

                member_series = pivoted_frame[member_name].astype(float)
                rolling_mean_252 = member_series.rolling(252, min_periods=252).mean()
                deviation_from_mean = member_series - rolling_mean_252
                delta20 = member_series.shift(-20) - member_series

                valid_mask = rolling_mean_252.notna() & delta20.notna()
                valid_mask_array = valid_mask.to_numpy(dtype=bool)

                aligned_dates = pivoted_frame.index[valid_mask_array].to_pydatetime()
                aligned_level_values = member_series.to_numpy(dtype=float)[valid_mask_array]
                aligned_mean_values = rolling_mean_252.to_numpy(dtype=float)[valid_mask_array]
                aligned_deviation_values = deviation_from_mean.to_numpy(dtype=float)[valid_mask_array]
                aligned_delta20_values = delta20.to_numpy(dtype=float)[valid_mask_array]

                fig1, ax1 = plt.subplots(figsize=(12.0, 4.0))
                ax1.plot(aligned_dates, aligned_level_values, label="logIV")
                ax1.plot(aligned_dates, aligned_mean_values, label="m252")
                ax1.set_title(f"{value_column_name} {member_name}")
                ax1.legend()
                fig1.tight_layout()
                pdf.savefig(fig1)
                plt.close(fig1)

                fig2, ax2 = plt.subplots(figsize=(12.0, 4.0))
                ax2.plot(aligned_dates, aligned_deviation_values)
                ax2.axhline(0.0, linestyle="--", linewidth=1.0)
                ax2.set_title(f"{value_column_name} {member_name} d_t")
                fig2.tight_layout()
                pdf.savefig(fig2)
                plt.close(fig2)

                fig3, ax3 = plt.subplots(figsize=(12.0, 4.0))
                ax3.plot(aligned_dates, aligned_delta20_values)
                ax3.axhline(0.0, linestyle="--", linewidth=1.0)
                ax3.set_title(f"{value_column_name} {member_name} delta20")
                fig3.tight_layout()
                pdf.savefig(fig3)
                plt.close(fig3)

                positive_mask = aligned_deviation_values > 0.0
                negative_mask = aligned_deviation_values < 0.0

                delta20_positive_values = aligned_delta20_values[positive_mask]
                delta20_negative_values = aligned_delta20_values[negative_mask]

                A = float(np.nanmedian(delta20_positive_values)) if delta20_positive_values.size > 0 else np.nan
                B = float(np.nanmedian(delta20_negative_values)) if delta20_negative_values.size > 0 else np.nan
                R = float(np.abs(A) / np.abs(B)) if np.isfinite(A) and np.isfinite(B) and np.abs(B) > 0.0 else np.nan
                MR = bool(np.isfinite(A) and np.isfinite(B) and (A < 0.0) and (B > 0.0))

                boxplot_positive = delta20_positive_values if delta20_positive_values.size > 0 else np.array([np.nan])
                boxplot_negative = delta20_negative_values if delta20_negative_values.size > 0 else np.array([np.nan])

                fig4, ax4 = plt.subplots(figsize=(8.0, 5.0))
                ax4.boxplot([boxplot_positive, boxplot_negative], labels=["d_t > 0", "d_t < 0"])
                annotation_text = f"A={A}\nB={B}\nR={R}\nMR={'TRUE' if MR else 'FALSE'}"
                ax4.text(0.02, 0.98, annotation_text, transform=ax4.transAxes, va="top", ha="left")
                ax4.set_title(f"{value_column_name} {member_name} | A={A} B={B} R={R} MR={'TRUE' if MR else 'FALSE'}")
                fig4.tight_layout()
                pdf.savefig(fig4)
                plt.close(fig4)

                member_index += 1
    fig_raw = _build_figure(pivoted, value_column_name)

    safe_filename = value_column_name.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_filename}.pdf"

    is_iv_log_panel = ("iv" in value_column_name.lower()) and ("log" in value_column_name.lower())

    if is_iv_log_panel:
        mean_by_member = pivoted.mean(axis=0, skipna=True)
        standard_deviation_by_member = pivoted.std(axis=0, skipna=True, ddof=1).replace(0.0, np.nan)
        pivoted_standardized = (pivoted - mean_by_member) / standard_deviation_by_member

        fig_standardized = _build_figure(pivoted_standardized, f"{value_column_name} (standardized)")

        with PdfPages(output_path) as pdf:
            pdf.savefig(fig_raw)
            pdf.savefig(fig_standardized)

        plt.close(fig_raw)
        plt.close(fig_standardized)

        pivoted_first_difference = pivoted.diff()

        fig_first_difference = _build_first_difference_figure(
            pivoted_first_difference,
            f"{value_column_name} (first differences)",
        )
        output_path_first_difference = output_dir / f"{safe_filename}_diff.pdf"
        fig_first_difference.savefig(output_path_first_difference, format="pdf")
        plt.close(fig_first_difference)

        fig_rolling_std = _build_rolling_std_figure(
            pivoted_first_difference,
            f"{value_column_name} (diff rolling std: 30/90/252)",
        )
        output_path_rolling_std = output_dir / f"{safe_filename}_diff_rolling_std.pdf"
        fig_rolling_std.savefig(output_path_rolling_std, format="pdf")
        plt.close(fig_rolling_std)

        _save_asymmetric_mean_reversion_diagnostics(pivoted, safe_filename)
        return

    fig_raw.savefig(output_path, format="pdf")
    plt.close(fig_raw)

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
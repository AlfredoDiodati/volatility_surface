import polars as pl
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from scipy import stats

subfolder = "SPX"
underlying_series = pl.read_parquet("data/" + subfolder + "/put/underlying.parquet")

underlying_series = underlying_series.with_columns(
    pl.col("UNDERLYING_LAST").log().diff().alias("log_return")
).drop_nulls(subset=["log_return"])

plot_dir = Path("plot/" + subfolder + "/underlying/")
plot_dir.mkdir(parents=True, exist_ok=True)

returns = underlying_series["log_return"].to_numpy()
lagged_returns = returns[:-1]
current_returns = returns[1:]

positive_lag_mask = lagged_returns > 0
negative_lag_mask = lagged_returns <= 0

results_text = []

results_text.append("Unconditional Return Statistics:")
results_text.append(f"  mean: {np.mean(current_returns):.8f}")
results_text.append(f"  std: {np.std(current_returns, ddof=1):.8f}")
results_text.append(f"  skewness: {stats.skew(current_returns):.8f}")
results_text.append(f"  kurtosis: {stats.kurtosis(current_returns):.8f}")
results_text.append(f"  min: {np.min(current_returns):.8f}")
results_text.append(f"  max: {np.max(current_returns):.8f}")
results_text.append(f"  count: {len(current_returns)}")

results_text.append("\nConditional on Positive Previous Return:")
positive_conditional = current_returns[positive_lag_mask]
results_text.append(f"  mean: {np.mean(positive_conditional):.8f}")
results_text.append(f"  std: {np.std(positive_conditional, ddof=1):.8f}")
results_text.append(f"  skewness: {stats.skew(positive_conditional):.8f}")
results_text.append(f"  kurtosis: {stats.kurtosis(positive_conditional):.8f}")
results_text.append(f"  min: {np.min(positive_conditional):.8f}")
results_text.append(f"  max: {np.max(positive_conditional):.8f}")
results_text.append(f"  count: {len(positive_conditional)}")

results_text.append("\nConditional on Negative Previous Return:")
negative_conditional = current_returns[negative_lag_mask]
results_text.append(f"  mean: {np.mean(negative_conditional):.8f}")
results_text.append(f"  std: {np.std(negative_conditional, ddof=1):.8f}")
results_text.append(f"  skewness: {stats.skew(negative_conditional):.8f}")
results_text.append(f"  kurtosis: {stats.kurtosis(negative_conditional):.8f}")
results_text.append(f"  min: {np.min(negative_conditional):.8f}")
results_text.append(f"  max: {np.max(negative_conditional):.8f}")
results_text.append(f"  count: {len(negative_conditional)}")

absolute_returns = np.abs(current_returns)
squared_returns = current_returns ** 2

def compute_acf(series, nlags=20):
    mean = np.mean(series)
    centered = series - mean
    variance = np.var(series)
    acf_values = [1.0]
    for lag in range(1, nlags + 1):
        numerator = np.mean(centered[:-lag] * centered[lag:])
        acf_values.append(numerator / variance)
    return np.array(acf_values)

acf_absolute = compute_acf(absolute_returns, nlags=20)
acf_squared = compute_acf(squared_returns, nlags=20)

dates = underlying_series["DATE"].to_numpy()[1:]

fig_returns = go.Figure()
fig_returns.add_trace(go.Scatter(x=dates, y=current_returns, mode='lines', name='Log Returns', line=dict(color='steelblue')))
fig_returns.update_layout(
    title="SPX Log Returns Over Time",
    xaxis_title="Date",
    yaxis_title="Log Return",
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=60, r=40, t=60, b=50)
)
fig_returns.write_image(str(plot_dir / "returns_timeseries.pdf"))

fig_absolute = go.Figure()
fig_absolute.add_trace(go.Bar(x=list(range(len(acf_absolute))), y=acf_absolute, marker_color="steelblue"))
fig_absolute.update_layout(
    title="ACF of Absolute Returns",
    xaxis_title="Lag",
    yaxis_title="ACF",
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=60, r=40, t=60, b=50)
)
fig_absolute.write_image(str(plot_dir / "acf_absolute.pdf"))

fig_squared = go.Figure()
fig_squared.add_trace(go.Bar(x=list(range(len(acf_squared))), y=acf_squared, marker_color="steelblue"))
fig_squared.update_layout(
    title="ACF of Squared Returns",
    xaxis_title="Lag",
    yaxis_title="ACF",
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=60, r=40, t=60, b=50)
)
fig_squared.write_image(str(plot_dir / "acf_squared.pdf"))

squared_t = squared_returns[:-1]
nonsquared_t_plus_one = current_returns[1:]

positive_lag_mask_shifted = lagged_returns[:-1] > 0
negative_lag_mask_shifted = lagged_returns[:-1] <= 0

leverage_full = np.corrcoef(squared_t, nonsquared_t_plus_one)[0, 1]
leverage_positive = np.corrcoef(squared_t[positive_lag_mask_shifted], nonsquared_t_plus_one[positive_lag_mask_shifted])[0, 1]
leverage_negative = np.corrcoef(squared_t[negative_lag_mask_shifted], nonsquared_t_plus_one[negative_lag_mask_shifted])[0, 1]

results_text.append("\nLeverage Test - Correlation(squared return t, nonsquared return t+1):")
results_text.append(f"  Full sample: {leverage_full:.8f}")
results_text.append(f"  After positive previous return: {leverage_positive:.8f}")
results_text.append(f"  After negative previous return: {leverage_negative:.8f}")

with open(str(plot_dir / "statistics.txt"), "w") as f:
    f.write("\n".join(results_text))
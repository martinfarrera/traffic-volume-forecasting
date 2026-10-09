"""Deterministic, Pandas-only exploratory summaries for traffic data."""

from __future__ import annotations

from typing import Any

import pandas as pd


def schema_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return stable row, column, dtype, and duplicate counts for a frame."""
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic missing-value row per column."""
    result = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": frame.isna().sum().astype(int).to_numpy(),
        }
    )
    result["missing_fraction"] = result["missing_count"] / len(frame) if len(frame) else 0.0
    return result


def temporal_summary(frame: pd.DataFrame, timestamp_column: str = "TimeStamp") -> dict[str, Any]:
    """Summarize coverage and the most common interval between timestamps."""
    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise").sort_values()
    if timestamps.empty:
        return {"start": None, "end": None, "rows": 0, "mode_interval": None}
    intervals = timestamps.diff().dropna()
    mode_interval = intervals.mode().iloc[0] if not intervals.empty else None
    return {
        "start": timestamps.iloc[0].isoformat(),
        "end": timestamps.iloc[-1].isoformat(),
        "rows": int(len(timestamps)),
        "mode_interval": mode_interval.isoformat() if mode_interval is not None else None,
    }


def aggregate_traffic(
    frame: pd.DataFrame,
    frequency: str,
    timestamp_column: str = "TimeStamp",
    target_column: str = "Traffic_Vol",
) -> pd.DataFrame:
    """Aggregate traffic by a Pandas time frequency, excluding missing targets."""
    if target_column not in frame:
        raise ValueError(f"Missing target column: {target_column}")
    values = frame[[timestamp_column, target_column]].copy()
    values[timestamp_column] = pd.to_datetime(values[timestamp_column], errors="raise")
    values = values.dropna(subset=[target_column]).set_index(timestamp_column)
    result = values[target_column].resample(frequency).agg(["count", "mean", "median", "min", "max"])
    result.index.name = timestamp_column
    return result.reset_index()


def build_eda_report(
    frame: pd.DataFrame,
    timestamp_column: str = "TimeStamp",
    target_column: str = "Traffic_Vol",
) -> dict[str, Any]:
    """Build the core serializable EDA report and hourly/daily aggregations."""
    return {
        "schema": schema_summary(frame),
        "missingness": missingness_summary(frame),
        "temporal": temporal_summary(frame, timestamp_column),
        "hourly_traffic": aggregate_traffic(frame, "h", timestamp_column, target_column),
        "daily_traffic": aggregate_traffic(frame, "D", timestamp_column, target_column),
    }

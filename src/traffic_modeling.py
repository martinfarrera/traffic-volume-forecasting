"""Leakage-safe feature engineering and time-aware traffic model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


DEFAULT_FEATURES = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
)


def build_supervised_features(
    frame: pd.DataFrame,
    timestamp_column: str = "TimeStamp",
    target_column: str = "Traffic_Vol",
) -> pd.DataFrame:
    """Create calendar and lag features using only observations before each row.

    Rows in the warm-up period are removed so every returned model row has a
    complete feature vector. The lagged target values are valid for one-step
    walk-forward evaluation, where observations prior to the prediction time
    are available.
    """
    required = {timestamp_column, target_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    result = frame[[timestamp_column, target_column]].copy()
    timestamps = pd.to_datetime(result[timestamp_column], errors="raise")
    result[timestamp_column] = timestamps
    result = result.sort_values(timestamp_column).reset_index(drop=True)
    target = result[target_column]

    result["hour"] = timestamps.sort_values().dt.hour.to_numpy()
    result["day_of_week"] = timestamps.sort_values().dt.dayofweek.to_numpy()
    result["month"] = timestamps.sort_values().dt.month.to_numpy()
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["lag_1"] = target.shift(1)
    result["lag_24"] = target.shift(24)
    result["lag_168"] = target.shift(168)
    result["rolling_mean_24"] = target.shift(1).rolling(24).mean()
    return result.dropna(subset=[target_column, *DEFAULT_FEATURES]).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically, keeping the most recent fraction for validation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("At least two rows are required for a split")
    cutoff = max(1, int(len(frame) * (1 - validation_fraction)))
    if cutoff >= len(frame):
        cutoff = len(frame) - 1
    ordered = frame.sort_values("TimeStamp").reset_index(drop=True)
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def naive_predictions(values: pd.Series | np.ndarray) -> np.ndarray:
    """Return a copy of a lag-based prediction series for baseline evaluation."""
    return np.asarray(values, dtype=float).copy()


def evaluate_predictions(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    """Calculate deterministic MAE and RMSE metrics."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")
    errors = actual_array - predicted_array
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def fit_tree_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit a histogram gradient-boosting regressor and return model metrics."""
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=random_state)
    model.fit(train.loc[:, feature_columns], train["Traffic_Vol"])
    predictions = model.predict(validation.loc[:, feature_columns])
    return {"model": model, "metrics": evaluate_predictions(validation["Traffic_Vol"], predictions)}

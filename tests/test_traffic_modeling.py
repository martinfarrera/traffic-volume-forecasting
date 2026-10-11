from __future__ import annotations

import numpy as np
import pandas as pd

from src.traffic_modeling import (
    build_supervised_features,
    chronological_split,
    compare_models,
    evaluate_predictions,
    naive_predictions,
)


def _frame(periods: int = 200) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "TimeStamp": timestamps,
            "Traffic_Vol": np.arange(periods, dtype=float),
            "Temperature": np.linspace(10, 20, periods),
            "Rainfall_last_hour": 0.0,
            "Snowfall_last_hour": 0.0,
            "Cloud_Cover": 50.0,
            "Weather": "Clear",
            "Weather_Desc": "clear sky",
            "Holiday": np.nan,
            "Date": timestamps.normalize(),
        }
    )


def test_feature_builder_uses_prior_values_and_drops_warmup_rows():
    result = build_supervised_features(_frame())

    assert {"hour", "day_of_week", "lag_1", "lag_24", "rolling_mean_24"}.issubset(result.columns)
    assert result["lag_1"].iloc[0] == 167.0
    assert result["lag_24"].iloc[0] == 144.0
    assert result["rolling_mean_24"].iloc[0] == 155.5
    assert result["Traffic_Vol"].isna().sum() == 0


def test_chronological_split_preserves_order_and_has_no_overlap():
    result = build_supervised_features(_frame())
    train, validation = chronological_split(result, validation_fraction=0.2)

    assert train["TimeStamp"].max() < validation["TimeStamp"].min()
    assert len(train) + len(validation) == len(result)


def test_naive_predictions_and_metrics_are_explicit():
    actual = pd.Series([10.0, 20.0, 30.0])
    previous = pd.Series([9.0, 22.0, 27.0])

    predictions = naive_predictions(previous)
    metrics = evaluate_predictions(actual, predictions)

    assert predictions.tolist() == previous.tolist()
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == np.sqrt(14 / 3)


def test_compare_models_returns_comparable_metric_records():
    result = compare_models(*chronological_split(build_supervised_features(_frame())))

    assert {"model", "mae", "rmse"}.issubset(result.columns)
    assert "lag_1" in set(result["model"])
    assert "hist_gradient_boosting" in set(result["model"])
    assert result[["mae", "rmse"]].notna().all().all()

from pathlib import Path

import pandas as pd

from src.traffic_data import load_dataset
from src.traffic_eda import (
    aggregate_traffic,
    build_eda_report,
    missingness_summary,
    schema_summary,
    temporal_summary,
)


DATASET_DIR = Path(__file__).parents[1] / "data" / "raw" / "Dataset"


def test_summaries_describe_downloaded_training_data():
    train, _ = load_dataset(DATASET_DIR)

    schema = schema_summary(train)
    assert schema["rows"] == 38373
    assert schema["columns"] == 10
    assert schema["duplicate_rows"] == 0
    assert missingness_summary(train).set_index("column").loc["Holiday", "missing_count"] == 38323


def test_temporal_summary_reports_coverage_and_hourly_frequency():
    train, _ = load_dataset(DATASET_DIR)

    summary = temporal_summary(train)

    assert summary["start"] == "2008-10-02T09:00:00"
    assert summary["end"] == "2014-06-30T23:00:00"
    assert summary["mode_interval"] == "P0DT1H0M0S"


def test_traffic_aggregation_is_deterministic_and_excludes_missing_targets():
    frame = pd.DataFrame(
        {
            "TimeStamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-02 00:00"]),
            "Traffic_Vol": [10, 20, None],
        }
    )

    result = aggregate_traffic(frame, "D")

    assert result.to_dict("records") == [
        {
            "TimeStamp": pd.Timestamp("2024-01-01"),
            "count": 2,
            "mean": 15.0,
            "median": 15.0,
            "min": 10.0,
            "max": 20.0,
        }
    ]


def test_build_report_contains_core_eda_outputs():
    train, _ = load_dataset(DATASET_DIR)

    report = build_eda_report(train)

    assert set(report) == {"schema", "missingness", "temporal", "hourly_traffic", "daily_traffic"}
    assert len(report["hourly_traffic"]) > 0
    assert len(report["daily_traffic"]) > 0

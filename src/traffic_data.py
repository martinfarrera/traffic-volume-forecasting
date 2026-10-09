"""Utilities for loading and validating the traffic forecasting dataset."""

from pathlib import Path
from typing import Tuple

import pandas as pd

REQUIRED_COLUMNS = (
    "Holiday",
    "Temperature",
    "Rainfall_last_hour",
    "Snowfall_last_hour",
    "Cloud_Cover",
    "Weather",
    "Weather_Desc",
    "TimeStamp",
    "Date",
    "Traffic_Vol",
)


def load_dataset(raw_dir: str | Path = "data/raw/Dataset") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs, validate their schema, and parse temporal fields."""
    raw_path = Path(raw_dir)
    train_path = raw_path / "Train.csv"
    test_path = raw_path / "Test.csv"

    missing_files = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Dataset files not found: {', '.join(missing_files)}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for name, frame in (("Train.csv", train), ("Test.csv", test)):
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing required columns: {', '.join(missing)}")

    for frame in (train, test):
        # The source mixes ambiguous dates with day values above 12 (for example
        # `13/10/08`), so day-first parsing is required for a consistent timeline.
        frame["TimeStamp"] = pd.to_datetime(frame["TimeStamp"], dayfirst=True, format="mixed", errors="raise")
        frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, format="mixed", errors="raise")

    return train, test

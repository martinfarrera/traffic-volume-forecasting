from pathlib import Path

import pandas as pd
import pytest

from src.traffic_data import REQUIRED_COLUMNS, load_dataset


DATASET_DIR = Path(__file__).parents[1] / "data" / "raw" / "Dataset"


def test_load_dataset_returns_train_and_test_with_expected_schema():
    train, test = load_dataset(DATASET_DIR)

    assert set(REQUIRED_COLUMNS).issubset(train.columns)
    assert list(train.columns) == list(test.columns)
    assert "Traffic_Vol" in train.columns
    assert train["Traffic_Vol"].notna().all()
    assert test["Traffic_Vol"].isna().all()


def test_loader_parses_timestamp_and_date_columns():
    train, _ = load_dataset(DATASET_DIR)

    assert pd.api.types.is_datetime64_any_dtype(train["TimeStamp"])
    assert pd.api.types.is_datetime64_any_dtype(train["Date"])
    assert train["TimeStamp"].min() < train["TimeStamp"].max()


def test_loader_preserves_missing_holiday_values():
    train, test = load_dataset(DATASET_DIR)

    assert "Holiday" in train.columns
    assert train["Holiday"].isna().sum() == 38323
    assert test["Holiday"].isna().sum() == 2205


def test_loader_rejects_missing_required_columns(tmp_path):
    pd.DataFrame({"Traffic_Vol": [1]}).to_csv(tmp_path / "Train.csv", index=False)
    pd.DataFrame({"Traffic_Vol": [None]}).to_csv(tmp_path / "Test.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(tmp_path)

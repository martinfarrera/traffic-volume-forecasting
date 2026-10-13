from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import traffic_demo, traffic_modeling


def historical_frame():
    # Two-hour gaps make row shifts visibly different from elapsed-hour lags.
    return pd.DataFrame({
        "TimeStamp": pd.date_range("2024-01-01", periods=220, freq="2h"),
        "Traffic_Vol": np.arange(220, dtype=float),
    })


def test_payload_is_repeatable_aligned_and_documents_evaluation():
    frame = historical_frame()
    payload = traffic_demo.build_payload(frame)
    assert payload == traffic_demo.build_payload(frame)
    train, validation = traffic_modeling.chronological_split(
        traffic_modeling.build_supervised_features(frame)
    )
    fitted = traffic_modeling.fit_lightgbm(train, validation)
    records = payload["records"]
    assert len(records) == len(validation) == 11
    assert [row["timestamp"] for row in records] == [
        timestamp.isoformat() for timestamp in validation["TimeStamp"]
    ]
    assert [row["actual"] for row in records] == validation["Traffic_Vol"].tolist()
    assert [row["lightgbm"] for row in records] == fitted["predictions"].tolist()
    assert [row["lag_1"] for row in records] == validation["lag_1"].tolist()
    assert records[1]["lag_1"] == records[0]["actual"]
    assert payload["schema_version"] == 1
    assert payload["features"] == list(traffic_modeling.DEFAULT_FEATURES)
    assert payload["model"]["name"] == "lightgbm"
    assert payload["model"]["parameters"] == fitted["model"].get_params()
    assert payload["split"]["train"]["count"] == len(train)
    assert payload["split"]["validation"] == {
        "count": 11, "start": records[0]["timestamp"], "end": records[-1]["timestamp"]
    }
    assert payload["split"]["source_rows"] == 220
    assert payload["split"]["excluded_rows"] == 168
    assert payload["metrics"]["lightgbm"] == fitted["metrics"]
    assert payload["metrics"]["lag_1"] == traffic_modeling.evaluate_predictions(
        validation["Traffic_Vol"], validation["lag_1"]
    )
    assert payload["dataset"]["license"] == "CC0-1.0"
    assert payload["dataset"]["source"].startswith("https://www.kaggle.com/")
    assert payload["evaluation"]["prior_observed_validation_targets"] is True
    assert "observation-row" in payload["evaluation"]["lag_semantics"]
    assert "historical one-step-ahead" in payload["evaluation"]["mode"]
    assert payload["evaluation"]["metrics_scope"] == "full validation holdout"
    assert "no timezone" in payload["evaluation"]["timestamp_semantics"]
    assert {"python", "numpy", "pandas", "scikit-learn", "lightgbm"} <= payload["dependencies"].keys()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_writer_rejects_nonfinite_json_without_overwriting(tmp_path, value):
    output = tmp_path / "forecast.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError):
        traffic_demo.write_payload({"records": [{"actual": value}]}, output)
    assert output.read_text(encoding="utf-8") == "existing"


def test_export_fails_explicitly_without_lightgbm(monkeypatch):
    monkeypatch.setattr(traffic_modeling, "LGBMRegressor", None)
    with pytest.raises(RuntimeError, match="LightGBM is required"):
        traffic_demo.build_payload(historical_frame())


def test_cli_loads_requested_directory_and_writes_json(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    output = tmp_path / "nested" / "forecast.json"
    calls = []

    def load(raw):
        calls.append(raw)
        return historical_frame(), pd.DataFrame()

    monkeypatch.setattr(traffic_demo, "load_dataset", load)
    traffic_demo.main(["--raw-dir", str(raw_dir), "--output", str(output)])
    assert calls == [raw_dir]
    original_bytes = output.read_bytes()
    assert json.loads(original_bytes) == traffic_demo.build_payload(historical_frame())
    traffic_demo.main(["--raw-dir", str(raw_dir), "--output", str(output)])
    assert output.read_bytes() == original_bytes


def test_timestamps_preserve_existing_offset():
    frame = historical_frame()
    frame["TimeStamp"] = frame["TimeStamp"].dt.tz_localize("Etc/GMT+6")
    payload = traffic_demo.build_payload(frame)
    assert payload["records"][0]["timestamp"].endswith("-06:00")

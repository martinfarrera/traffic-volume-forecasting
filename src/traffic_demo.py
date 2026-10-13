"""Deterministic historical holdout export; not live future forecasting."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd

from src.traffic_data import load_dataset
from src.traffic_modeling import (
    DEFAULT_FEATURES,
    build_supervised_features,
    chronological_split,
    evaluate_predictions,
    fit_lightgbm,
)


def build_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the latest 20% of complete feature rows from labeled Train.csv.

    Features are built before splitting: prior observed validation targets are
    intentionally available for each subsequent one-step prediction.
    """
    features = build_supervised_features(frame)
    train, validation = chronological_split(features)
    fitted = fit_lightgbm(train, validation)
    records = [
        {
            "timestamp": timestamp.isoformat(),
            "actual": float(actual),
            "lightgbm": float(predicted),
            "lag_1": float(lag),
        }
        for timestamp, actual, predicted, lag in zip(
            validation["TimeStamp"], validation["Traffic_Vol"],
            fitted["predictions"], validation["lag_1"],
        )
    ]
    return {
        "schema_version": 1,
        "dataset": {
            "name": "Real-World Traffic Volume Forecasting",
            "source": "https://www.kaggle.com/datasets/sudhanvahg/real-world-traffic-volume-forecasting",
            "license": "CC0-1.0",
            "license_evidence": "As reported by the Kaggle CLI",
            "evaluated_file": "Train.csv",
            "target": "Traffic_Vol",
        },
        "model": {
            "name": "lightgbm",
            "estimator": "lightgbm.LGBMRegressor",
            "parameters": fitted["model"].get_params(),
            "selection": "Provisional comparison winner; not an independent final test",
        },
        "features": list(DEFAULT_FEATURES),
        "split": {
            "method": "chronological by observation row",
            "validation_fraction": 0.2,
            "source_rows": len(frame),
            "feature_rows": len(features),
            "excluded_rows": len(frame) - len(features),
            "train": _boundaries(train),
            "validation": _boundaries(validation),
        },
        "evaluation": {
            "mode": "historical one-step-ahead evaluation, not live future forecasting",
            "lag_semantics": "observation-row shifts, not guaranteed elapsed-hour offsets",
            "rolling_semantics": "Mean of the preceding 24 observation rows",
            "prior_observed_validation_targets": True,
            "fit_semantics": "Fit once on training rows; no validation refitting",
            "metrics_scope": "full validation holdout",
            "timestamp_semantics": "Source timestamps parsed day-first; no timezone invented or converted",
            "excluded_rows": "Warm-up or missing target/feature rows",
            "test_file": "Test.csv is not evaluated because target values are unavailable",
        },
        "metrics": {
            "lightgbm": fitted["metrics"],
            "lag_1": evaluate_predictions(validation["Traffic_Vol"], validation["lag_1"]),
        },
        "dependencies": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm")},
        },
        "records": records,
    }


def _boundaries(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "count": len(frame),
        "start": frame["TimeStamp"].iloc[0].isoformat(),
        "end": frame["TimeStamp"].iloc[-1].isoformat(),
    }


def write_payload(payload: dict[str, Any], output: str | Path) -> None:
    """Serialize strictly before touching the output; NaN/Infinity are errors."""
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/Dataset"))
    parser.add_argument("--output", type=Path, default=Path("docs/data/forecast.json"))
    args = parser.parse_args(argv)
    train, _ = load_dataset(args.raw_dir)
    write_payload(build_payload(train), args.output)


if __name__ == "__main__":
    main()

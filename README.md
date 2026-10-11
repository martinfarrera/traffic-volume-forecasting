# Real-World Traffic Volume Forecasting

Weekly Data Science portfolio project: forecast traffic volume using historical traffic, weather, calendar, and holiday variables.

## Status

- Dataset selected: [Kaggle — Real-World Traffic Volume Forecasting](https://www.kaggle.com/datasets/sudhanvahg/real-world-traffic-volume-forecasting)
- Downloaded: 2026-10-08 through Kaggle CLI into `data/raw/` (raw files are ignored)
- License: `CC0-1.0` as reported by the Kaggle CLI
- Dataset evidence: `Train.csv` has 38,373 rows; `Test.csv` has 2,208 rows; `Traffic_Vol` is absent in test values by design
- EDA status: Pandas-only schema, missingness, duplicate, temporal coverage, and hourly/daily aggregation utilities are implemented in `src/traffic_eda.py`
- Modeling status: leakage-safe lag/calendar features, chronological holdout evaluation, naive metrics, histogram gradient-boosting, and optional LightGBM comparison are implemented in `src/traffic_modeling.py`

## Planned methodology

1. EDA: schema, missingness, duplicates, outliers, distributions, seasonality, correlations.
2. EDA verification: temporal holdout, autocorrelation/seasonality checks, and statistical comparisons.
3. Modeling: chronological holdout with the most recent 20% reserved for validation; calendar features and lagged/rolling target features use only prior observations.
4. Model comparison: the same frozen split produced lag-1 MAE/RMSE 595.62/826.63, HistGradientBoosting 166.29/263.06, and LightGBM 164.87/262.24. LightGBM is selected provisionally for the demo because it had the lowest error, while the improvement over HistGradientBoosting is modest.
5. ML candidates: Gradient Boosting/LightGBM and Random Forest with time-aware validation and tuning.
4. Deep learning: LSTM for sequential windows with an honest comparison against ML baselines.
5. Reporting: metrics, residual analysis, training curves, limitations, and reproducible conclusions.

## Structure

- `data/raw/` — immutable downloaded data (not committed)
- `data/processed/` — generated features (not committed)
- `notebooks/` — exploratory and modeling notebooks
- `src/` — reusable Python modules (`traffic_data.py`, `traffic_eda.py`, `traffic_modeling.py`)
- `models/` — serialized models (not committed)
- `reports/` — figures and final report

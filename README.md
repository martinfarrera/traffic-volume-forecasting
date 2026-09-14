# Real-World Traffic Volume Forecasting

Weekly Data Science portfolio project: forecast traffic volume using historical traffic, weather, calendar, and holiday variables.

## Status

- Dataset selected: [Kaggle — Real-World Traffic Volume Forecasting](https://www.kaggle.com/datasets/sudhanvahg/real-world-traffic-volume-forecasting)
- Download: pending Kaggle API credentials (`~/.kaggle/kaggle.json` is not configured in this environment)
- Download date/version/license: to be recorded after download from the exact Kaggle data card

## Planned methodology

1. EDA: schema, missingness, duplicates, outliers, distributions, seasonality, correlations.
2. EDA verification: temporal holdout, autocorrelation/seasonality checks, and statistical comparisons.
3. ML candidates: Gradient Boosting/LightGBM and Random Forest with time-aware validation and tuning.
4. Deep learning: LSTM for sequential windows with an honest comparison against ML baselines.
5. Reporting: metrics, residual analysis, training curves, limitations, and reproducible conclusions.

## Structure

- `data/raw/` — immutable downloaded data (not committed)
- `data/processed/` — generated features (not committed)
- `notebooks/` — exploratory and modeling notebooks
- `src/` — reusable Python modules
- `models/` — serialized models (not committed)
- `reports/` — figures and final report

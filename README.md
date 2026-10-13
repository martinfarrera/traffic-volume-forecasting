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

## Historical evaluation export

Generate `docs/data/forecast.json` for the static historical evaluation dashboard.
The export contains precomputed predictions, not a backend or live prediction service.

From the repository root, use the project's Python environment with the
requirements installed, including LightGBM (and pytest for verification). Place
Kaggle's `Train.csv` and `Test.csv` under `data/raw/Dataset/`, then run:

```bash
pyenv exec python -m pytest -q tests/test_traffic_data.py tests/test_traffic_eda.py tests/test_traffic_modeling.py tests/test_traffic_demo.py
pyenv exec python -m src.traffic_demo --raw-dir data/raw/Dataset --output docs/data/forecast.json
pyenv exec python -m json.tool docs/data/forecast.json > /dev/null
```

`--raw-dir` and `--output` can point to alternative locations. The loader checks
both CSV schemas, but only labeled `Train.csv` is evaluated. LightGBM is required
for export: an unavailable dependency raises an explicit error, never a fallback
model disguised as LightGBM. The existing model comparison still treats it as
optional.

### Interpretation and reproducibility

- This is **historical one-step-ahead evaluation**, not a multi-step future
  forecast. Calendar and target-derived features are created before a frozen
  chronological 80/20 row split. Warm-up or incomplete feature rows are excluded.
- Lags 1, 24, and 168 are **observation-row shifts**, not guaranteed elapsed-hour
  offsets. The rolling mean uses the preceding 24 observation rows. Gaps in the
  source timeline therefore matter.
- **Prior observed targets are consumed within validation** to predict each
  subsequent row. LightGBM is fit once on the training split, without validation
  refitting. These metrics do not estimate performance when future target
  observations are unavailable.
- Schema version 1 includes aligned timestamp/actual/LightGBM/lag-1 records,
  full-holdout MAE and RMSE for both predictors, split counts and inclusive
  boundaries, feature names, estimator parameters, dependency versions, and
  dataset source/license. Metrics are not calculated on a displayed subset.
- Timestamps preserve the loader's day-first interpretation and are serialized
  as ISO 8601 without inventing a timezone or appending `Z` to naive timestamps.
- The fixed seed and unchanged comparison parameters produce repeatable exports
  for the same input and environment; no generation time is embedded. JSON
  serialization rejects NaN and Infinity before writing. Cross-version/platform
  bitwise equivalence is not promised; dependency versions are recorded, while
  most entries in `requirements.txt` are not pinned.
- LightGBM's selection is provisional and used this same holdout. This is not an
  independent final test or evidence of production forecasting quality.

## Explore the historical dashboard

From the repository root, serve the committed static files locally:

```bash
pyenv exec python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/`; stop the server with Ctrl+C. Opening the HTML as a
`file://` URL is not supported because the dashboard fetches its JSON over HTTP.
No model, raw dataset, framework, CDN, or npm installation is needed to view it.

Select a zero-based starting observation and a 24, 48, or 168-row window.
The chart and scrollable table compare actual traffic, LightGBM, and lag-1;
the final window can contain fewer rows. The chart spaces observed rows equally,
not by elapsed time. Timestamps retain the source representation without a
fabricated timezone. Full-holdout metrics stay fixed as the window changes.
Methodology, limitations, attribution/license, and expandable model/validation
metadata come from the export. Controls support keyboard input; the SVG has a
text description and an accompanying exact-value table.

The `docs/` directory is ready for static hosting with relative asset/data URLs
and `.nojekyll`, including a GitHub Pages project-path prefix. **GitHub Pages
publication is not configured or verified by this work unit.**

Dashboard contract tests require Python/pytest and Node (no Node packages):

```bash
pyenv exec python -m pytest -q tests/test_web_demo.py
node --check docs/app.js
```

These tests exercise loading, errors, empty data, metrics, and bounded window
updates using a minimal DOM. They do not replace visual or screen-reader testing
in a real browser.

## Structure

- `data/raw/` — immutable downloaded data (not committed)
- `data/processed/` — generated features (not committed)
- `notebooks/` — exploratory and modeling notebooks
- `src/` — reusable Python modules (`traffic_data.py`, `traffic_eda.py`, `traffic_modeling.py`, `traffic_demo.py`)
- `docs/` — plain HTML/CSS/JavaScript historical dashboard and `.nojekyll`
- `docs/data/forecast.json` — generated historical evaluation payload
- `models/` — serialized models (not committed)
- `reports/` — figures and final report

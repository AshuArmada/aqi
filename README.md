# AQI PM2.5 Forecasting

A small pipeline that fetches air-quality and weather data from the
[Open-Meteo](https://open-meteo.com/) API, engineers time-series features, and
trains a Random Forest model to predict daily PM2.5 concentration, with
bootstrap-based prediction intervals.

## Project layout

```
aqi/
├── data_pipeline.py      # Fetches air quality + weather data and builds data.csv
├── pipeline.py           # Feature engineering, training, evaluation, artifact export
├── model 4.py            # Thin CLI entry point that calls pipeline.main()
├── modeltrain2.py        # Earlier standalone training script (superseded by pipeline.py)
├── data.csv              # Sample/current dataset (lat, lon, weather + PM25 target)
├── requirements.txt
├── artifacts/            # Generated model + evaluation outputs (created on train)
│   ├── pm25_rf_package.joblib
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── holdout_predictions.csv
└── tests/
    ├── test_data_pipeline.py
    └── test_pipeline_e2e.py
```

## Setup

```bash
pip install -r aqi/requirements.txt
```

## Usage

### 1. Fetch data

Pulls daily-aggregated PM2.5/AOD and weather data for one or more locations
and writes a model-ready CSV.

```bash
python aqi/data_pipeline.py \
  --locations "28.6,77.2;19.07,72.87" \
  --start-date 2023-08-01 \
  --end-date 2023-08-31 \
  --output-path aqi/data.csv
```

### 2. Train the model

```bash
python aqi/pipeline.py --data-path aqi/data.csv --artifact-dir aqi/artifacts
```

This will:
- Build cyclical time features (month, day-of-year, weekday) and wind
  speed/direction features.
- Add per-location lag/rolling features (`PM25_lag1..14`,
  `PM25_roll3/7_mean`, `PM25_roll7_std`) when enough history is available per
  location, otherwise falls back to a weather-only feature set.
- Split the data chronologically (last 20% of dates held out as test).
- Train a baseline `RandomForestRegressor` and a tuned model via
  `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation, keeping
  whichever has the lower holdout RMSE.
- Compute 90% prediction intervals via bootstrap resampling.
- Save the trained model, metrics, feature importances, and holdout
  predictions to `artifacts/`.

### Outputs

| File | Description |
|---|---|
| `pm25_rf_package.joblib` | Trained model + feature list + metrics + metadata |
| `metrics.json` | CV/holdout RMSE, MAE, R², chosen model, best hyperparameters |
| `feature_importance.csv` | Feature importances from the chosen model |
| `holdout_predictions.csv` | Test-set predictions with p05/p50/p95 intervals |

## Testing

```bash
pytest aqi/tests
```

- `test_data_pipeline.py` — unit tests for location parsing, Open-Meteo
  response parsing, and dataset validation (using a fake fetcher, no network
  calls).
- `test_pipeline_e2e.py` — end-to-end test that trains on synthetic data and
  checks that artifacts are produced correctly.

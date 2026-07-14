import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.utils import resample


TARGET = "PM25"
REQUIRED_COLUMNS = {
    "date",
    "lat",
    "lon",
    "aod",
    "d2m",
    "t2m",
    "u10",
    "v10",
    "sp",
    "tp",
    TARGET,
}
BASE_FEATURES = [
    "lat",
    "lon",
    "aod",
    "d2m",
    "t2m",
    "u10",
    "v10",
    "sp",
    "tp",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "weekday_sin",
    "weekday_cos",
]
LAG_FEATURES = [
    "PM25_lag1",
    "PM25_lag2",
    "PM25_lag3",
    "PM25_lag7",
    "PM25_lag14",
    "PM25_roll3_mean",
    "PM25_roll7_mean",
    "PM25_roll7_std",
]


@dataclass(frozen=True)
class PipelineConfig:
    data_path: Path = Path("data.csv")
    artifact_dir: Path = Path("artifacts")
    target: str = TARGET
    random_state: int = 42
    test_fraction: float = 0.2
    cv_splits: int = 5
    baseline_estimators: int = 300
    tuning_iter: int = 12
    bootstrap_rounds: int = 30


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return df.dropna(subset=["date"]).sort_values(["date", "lat", "lon"]).reset_index(drop=True)


def add_features(data):
    df = data.copy()
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekday"] = df["date"].dt.weekday
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

    wind_dir_rad = np.arctan2(-df["u10"], -df["v10"])
    df["wind_dir_sin"] = np.sin(wind_dir_rad)
    df["wind_dir_cos"] = np.cos(wind_dir_rad)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    return df


def add_grouped_lag_features(data, group_cols, target=TARGET):
    df = data.sort_values(group_cols + ["date"]).copy()
    grouped = df.groupby(group_cols, sort=False)[target]

    for lag in [1, 2, 3, 7, 14]:
        df[f"{target}_lag{lag}"] = grouped.shift(lag)

    previous_target = grouped.shift(1)
    group_keys = [df[col] for col in group_cols]
    df[f"{target}_roll3_mean"] = previous_target.groupby(group_keys).transform(
        lambda values: values.rolling(3, min_periods=3).mean()
    )
    df[f"{target}_roll7_mean"] = previous_target.groupby(group_keys).transform(
        lambda values: values.rolling(7, min_periods=7).mean()
    )
    df[f"{target}_roll7_std"] = previous_target.groupby(group_keys).transform(
        lambda values: values.rolling(7, min_periods=7).std()
    )
    return df.sort_values(["date"] + group_cols).reset_index(drop=True)


def build_model_frame(raw_data, target=TARGET):
    df = add_features(raw_data)
    df = add_grouped_lag_features(df, group_cols=["lat", "lon"], target=target)
    lagged_model_df = df.dropna(subset=BASE_FEATURES + LAG_FEATURES + [target]).reset_index(drop=True)

    if lagged_model_df["date"].nunique() >= 5:
        return lagged_model_df, BASE_FEATURES + LAG_FEATURES, "weather_time_location_lagged"

    model_df = df.dropna(subset=BASE_FEATURES + [target]).reset_index(drop=True)
    return model_df, BASE_FEATURES, "weather_time_location"


def chronological_date_split(df, test_fraction=0.2):
    unique_dates = np.array(sorted(df["date"].dropna().unique()))
    if len(unique_dates) < 5:
        raise ValueError("Need at least 5 unique dates for a chronological split.")

    cut_index = max(1, int(len(unique_dates) * (1 - test_fraction)))
    cut_date = unique_dates[cut_index]
    train_mask = df["date"] < cut_date
    test_mask = df["date"] >= cut_date
    return train_mask, test_mask, pd.Timestamp(cut_date)


def make_time_series_cv(row_count, requested_splits):
    max_splits = min(requested_splits, row_count - 1)
    if max_splits < 2:
        raise ValueError("Need at least 3 training rows for time-series cross-validation.")
    return TimeSeriesSplit(n_splits=max_splits)


def bootstrap_prediction_intervals(model, x_train, y_train, x_test, rounds=30, seed=100):
    params = {
        key: value
        for key, value in model.get_params().items()
        if key in RandomForestRegressor().get_params()
    }
    params["n_jobs"] = -1

    preds = []
    rng = np.random.RandomState(seed)
    for _ in range(rounds):
        x_boot, y_boot = resample(x_train, y_train, random_state=rng.randint(0, 1_000_000))
        boot_model = RandomForestRegressor(**params)
        boot_model.set_params(random_state=rng.randint(0, 1_000_000))
        boot_model.fit(x_boot, y_boot)
        preds.append(boot_model.predict(x_test))

    prediction_matrix = np.vstack(preds)
    return {
        "mean": prediction_matrix.mean(axis=0),
        "p05": np.percentile(prediction_matrix, 5, axis=0),
        "p50": np.percentile(prediction_matrix, 50, axis=0),
        "p95": np.percentile(prediction_matrix, 95, axis=0),
    }


def train_pipeline(config=PipelineConfig()):
    raw_df = load_data(config.data_path)
    model_df, features, feature_mode = build_model_frame(raw_df, target=config.target)
    if model_df.empty:
        raise ValueError("No usable rows remain after feature creation. Check dataset coverage and missing values.")

    train_mask, test_mask, cut_date = chronological_date_split(model_df, config.test_fraction)
    x_train = model_df.loc[train_mask, features]
    y_train = model_df.loc[train_mask, config.target]
    x_test = model_df.loc[test_mask, features]
    y_test = model_df.loc[test_mask, config.target]
    if x_train.empty or x_test.empty:
        raise ValueError("Train/test split produced an empty set. Check dataset date coverage.")

    scorer = make_scorer(lambda y_true, y_pred: -rmse(y_true, y_pred))
    tscv = make_time_series_cv(len(x_train), config.cv_splits)
    baseline_model = RandomForestRegressor(
        n_estimators=config.baseline_estimators,
        min_samples_leaf=2,
        random_state=config.random_state,
        n_jobs=-1,
    )
    cv_scores = cross_val_score(baseline_model, x_train, y_train, cv=tscv, scoring=scorer, n_jobs=-1)

    baseline_model.fit(x_train, y_train)
    baseline_pred = baseline_model.predict(x_test)
    baseline_metrics = evaluate(y_test, baseline_pred)

    param_grid = {
        "n_estimators": [
            max(10, int(config.baseline_estimators * 0.67)),
            config.baseline_estimators,
            max(20, int(config.baseline_estimators * 1.67)),
        ],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [2, 4, 8],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=config.random_state, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=config.tuning_iter,
        cv=tscv,
        scoring=scorer,
        n_jobs=-1,
        random_state=config.random_state,
        refit=True,
    )
    search.fit(x_train, y_train)

    tuned_model = search.best_estimator_
    tuned_pred = tuned_model.predict(x_test)
    tuned_metrics = evaluate(y_test, tuned_pred)
    chosen_model = tuned_model if tuned_metrics["RMSE"] <= baseline_metrics["RMSE"] else baseline_model
    chosen_label = "tuned" if chosen_model is tuned_model else "baseline"
    chosen_pred = tuned_pred if chosen_label == "tuned" else baseline_pred

    intervals = bootstrap_prediction_intervals(
        chosen_model,
        x_train,
        y_train,
        x_test,
        rounds=config.bootstrap_rounds,
        seed=config.random_state + 58,
    )
    interval_coverage = float(((y_test.values >= intervals["p05"]) & (y_test.values <= intervals["p95"])).mean())

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_output = model_df.loc[test_mask, ["date", "lat", "lon", config.target]].copy()
    prediction_output["prediction"] = chosen_pred
    prediction_output["pi05"] = intervals["p05"]
    prediction_output["pi50"] = intervals["p50"]
    prediction_output["pi95"] = intervals["p95"]
    prediction_output.to_csv(config.artifact_dir / "holdout_predictions.csv", index=False)

    importances = pd.Series(chosen_model.feature_importances_, index=features).sort_values(ascending=False)
    importances.to_csv(config.artifact_dir / "feature_importance.csv", header=["importance"])

    metrics = {
        "rows": int(len(model_df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "locations": int(model_df[["lat", "lon"]].drop_duplicates().shape[0]),
        "feature_mode": feature_mode,
        "date_range": {
            "start": str(model_df["date"].min().date()),
            "end": str(model_df["date"].max().date()),
            "test_start": str(cut_date.date()),
        },
        "cv_rmse": {"mean": float(-cv_scores.mean()), "std": float(cv_scores.std())},
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
        "chosen": chosen_label,
        "prediction_interval_90_coverage": interval_coverage,
        "best_params": search.best_params_,
    }
    with (config.artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    package = {
        "model": chosen_model,
        "chosen": chosen_label,
        "features": features,
        "target": config.target,
        "metrics": metrics,
        "notes": {
            "split": "chronological by date; last 20% of dates held out",
            "feature_mode": feature_mode,
            "lags": "used only when enough per-location history is available",
            "rolling": "shifted before rolling to prevent target leakage",
            "time_features": "cyclical month, day-of-year, weekday",
            "wind_features": "speed plus sin/cos direction",
        },
    }
    joblib.dump(package, config.artifact_dir / "pm25_rf_package.joblib")

    return {
        "metrics": metrics,
        "features": features,
        "feature_importance": importances,
        "artifact_dir": config.artifact_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AQI PM2.5 forecasting pipeline.")
    parser.add_argument("--data-path", type=Path, default=Path("data.csv"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--baseline-estimators", type=int, default=300)
    parser.add_argument("--tuning-iter", type=int, default=12)
    parser.add_argument("--bootstrap-rounds", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_pipeline(
        PipelineConfig(
            data_path=args.data_path,
            artifact_dir=args.artifact_dir,
            cv_splits=args.cv_splits,
            baseline_estimators=args.baseline_estimators,
            tuning_iter=args.tuning_iter,
            bootstrap_rounds=args.bootstrap_rounds,
        )
    )
    metrics = result["metrics"]
    print("Chosen model:", metrics["chosen"])
    print("CV RMSE:", metrics["cv_rmse"])
    print("Baseline holdout metrics:", metrics["baseline"])
    print("Tuned holdout metrics:", metrics["tuned"])
    print("90% interval coverage:", round(metrics["prediction_interval_90_coverage"], 3))
    print("Top feature importances:")
    print(result["feature_importance"].head(10).to_string())
    print(f"Saved artifacts to {result['artifact_dir']}")


if __name__ == "__main__":
    main()

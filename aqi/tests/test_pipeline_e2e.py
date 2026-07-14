import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import PipelineConfig, train_pipeline


def make_synthetic_aqi_data(days=24, locations=4):
    rng = np.random.default_rng(42)
    rows = []
    dates = pd.date_range("2023-01-01", periods=days, freq="D")

    for location_idx in range(locations):
        lat = 10.0 + location_idx
        lon = 76.0 + location_idx * 0.5
        location_bias = location_idx * 1.75

        for day_idx, date in enumerate(dates):
            seasonal = np.sin(day_idx / 4.0)
            aod = 0.5 + 0.04 * day_idx + location_idx * 0.05 + rng.normal(0, 0.01)
            d2m = 292 + seasonal + rng.normal(0, 0.05)
            t2m = 300 + 1.5 * seasonal + rng.normal(0, 0.05)
            u10 = 2 + np.cos(day_idx / 3.0) + rng.normal(0, 0.02)
            v10 = 1 + np.sin(day_idx / 5.0) + rng.normal(0, 0.02)
            sp = 98000 + location_idx * 80 + rng.normal(0, 5)
            tp = max(0, 0.002 * seasonal + rng.normal(0, 0.0002))
            pm25 = 18 + 8 * aod + 0.03 * (t2m - 273) + location_bias + 0.25 * day_idx

            rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "aod": aod,
                    "d2m": d2m,
                    "t2m": t2m,
                    "u10": u10,
                    "v10": v10,
                    "sp": sp,
                    "tp": tp,
                    "PM25": pm25,
                    "date": date,
                }
            )

    return pd.DataFrame(rows)


def test_train_pipeline_writes_artifacts(tmp_path):
    data_path = tmp_path / "data.csv"
    artifact_dir = tmp_path / "artifacts"
    make_synthetic_aqi_data().to_csv(data_path, index=False)

    result = train_pipeline(
        PipelineConfig(
            data_path=data_path,
            artifact_dir=artifact_dir,
            cv_splits=3,
            baseline_estimators=12,
            tuning_iter=2,
            bootstrap_rounds=3,
        )
    )

    metrics_path = artifact_dir / "metrics.json"
    predictions_path = artifact_dir / "holdout_predictions.csv"
    importance_path = artifact_dir / "feature_importance.csv"
    package_path = artifact_dir / "pm25_rf_package.joblib"

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert importance_path.exists()
    assert package_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    feature_importance = pd.read_csv(importance_path)
    package = joblib.load(package_path)

    assert result["metrics"]["rows"] == metrics["rows"]
    assert metrics["train_rows"] > 0
    assert metrics["test_rows"] > 0
    assert metrics["chosen"] in {"baseline", "tuned"}
    assert metrics["feature_mode"] == "weather_time_location_lagged"
    assert {"prediction", "pi05", "pi50", "pi95"}.issubset(predictions.columns)
    assert len(predictions) == metrics["test_rows"]
    assert len(feature_importance) == len(package["features"])
    assert package["target"] == "PM25"
    assert package["metrics"]["chosen"] == metrics["chosen"]

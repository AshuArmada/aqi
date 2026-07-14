import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline import DataFetchConfig, Location, build_dataset, parse_locations, validate_model_data, write_dataset


def hourly_times():
    return [
        "2023-01-01T00:00",
        "2023-01-01T12:00",
        "2023-01-02T00:00",
        "2023-01-02T12:00",
    ]


def fake_fetcher(url, params, timeout_seconds):
    assert timeout_seconds == 60
    if "air-quality" in url:
        return {
            "hourly": {
                "time": hourly_times(),
                "pm2_5": [10.0, 14.0, 20.0, 24.0],
                "aerosol_optical_depth": [0.2, 0.4, 0.6, 0.8],
            }
        }

    return {
        "hourly": {
            "time": hourly_times(),
            "temperature_2m": [25.0, 27.0, 26.0, 28.0],
            "dew_point_2m": [20.0, 22.0, 21.0, 23.0],
            "wind_speed_10m": [2.0, 2.0, 4.0, 4.0],
            "wind_direction_10m": [90.0, 90.0, 180.0, 180.0],
            "surface_pressure": [1000.0, 1002.0, 999.0, 1001.0],
            "precipitation": [1.0, 2.0, 3.0, 4.0],
        }
    }


def test_parse_locations():
    locations = parse_locations("10.0,76.5;11.25,77.75")

    assert locations == (Location(10.0, 76.5), Location(11.25, 77.75))


def test_build_dataset_from_open_meteo_payloads():
    dataset = build_dataset(
        DataFetchConfig(
            locations=(Location(10.0, 76.5),),
            start_date="2023-01-01",
            end_date="2023-01-02",
        ),
        fetcher=fake_fetcher,
    )

    assert list(dataset.columns) == [
        "lat",
        "lon",
        "aod",
        "d2m",
        "t2m",
        "u10",
        "v10",
        "sp",
        "tp",
        "PM25",
        "date",
        "month",
        "dayofyear",
    ]
    assert len(dataset) == 2
    first_row = dataset.iloc[0]
    assert first_row["PM25"] == 12.0
    assert first_row["aod"] == pytest.approx(0.3)
    assert first_row["t2m"] == pytest.approx(299.15)
    assert first_row["d2m"] == pytest.approx(294.15)
    assert first_row["u10"] == pytest.approx(-2.0)
    assert abs(first_row["v10"]) < 1e-12
    assert first_row["sp"] == 100100.0
    assert first_row["tp"] == 0.003
    assert first_row["month"] == 1
    assert first_row["dayofyear"] == 1


def test_write_dataset_and_validate_schema(tmp_path):
    output_path = tmp_path / "data.csv"

    written = write_dataset(
        DataFetchConfig(
            locations=(Location(10.0, 76.5),),
            start_date="2023-01-01",
            end_date="2023-01-02",
            output_path=output_path,
        ),
        fetcher=fake_fetcher,
    )
    loaded = pd.read_csv(output_path)

    assert output_path.exists()
    assert len(written) == len(loaded)
    assert set(["PM25", "aod", "d2m", "t2m", "u10", "v10", "sp", "tp"]).issubset(loaded.columns)


def test_validate_model_data_rejects_missing_values():
    invalid = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "lat": [10.0],
            "lon": [76.5],
            "aod": [np.nan],
            "d2m": [294.15],
            "t2m": [299.15],
            "u10": [-2.0],
            "v10": [0.0],
            "sp": [100100.0],
            "tp": [0.003],
            "PM25": [12.0],
        }
    )

    with pytest.raises(ValueError, match="Columns contain non-numeric or missing values"):
        validate_model_data(invalid)

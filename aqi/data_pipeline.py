import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from pipeline import REQUIRED_COLUMNS


AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(frozen=True)
class Location:
    lat: float
    lon: float


@dataclass(frozen=True)
class DataFetchConfig:
    locations: tuple[Location, ...]
    start_date: str
    end_date: str
    output_path: Path = Path("data.csv")
    timezone: str = "auto"
    timeout_seconds: int = 60


def parse_locations(value):
    locations = []
    for item in value.split(";"):
        if not item.strip():
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError("Locations must look like 'lat,lon;lat,lon'.")
        locations.append(Location(lat=float(parts[0]), lon=float(parts[1])))

    if not locations:
        raise ValueError("At least one location is required.")
    return tuple(locations)


def fetch_json(url, params, timeout_seconds=60):
    request_url = f"{url}?{urlencode(params)}"
    with urlopen(request_url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo returned an error."))
    return payload


def hourly_payload_to_frame(payload, rename_map):
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("Open-Meteo response did not include hourly time values.")

    frame = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"])})
    for source_name, target_name in rename_map.items():
        if source_name not in hourly:
            raise ValueError(f"Open-Meteo response did not include '{source_name}'.")
        frame[target_name] = hourly[source_name]

    return frame


def fetch_air_quality(location, start_date, end_date, timezone, timeout_seconds=60, fetcher=fetch_json):
    payload = fetcher(
        AIR_QUALITY_URL,
        {
            "latitude": location.lat,
            "longitude": location.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "pm2_5,aerosol_optical_depth",
            "timezone": timezone,
            "domains": "cams_global",
            "cell_selection": "nearest",
        },
        timeout_seconds,
    )
    frame = hourly_payload_to_frame(payload, {"pm2_5": "PM25", "aerosol_optical_depth": "aod"})
    daily = frame.assign(date=frame["datetime"].dt.date).groupby("date", as_index=False).mean(numeric_only=True)
    return daily


def fetch_weather(location, start_date, end_date, timezone, timeout_seconds=60, fetcher=fetch_json):
    payload = fetcher(
        WEATHER_ARCHIVE_URL,
        {
            "latitude": location.lat,
            "longitude": location.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": (
                "temperature_2m,dew_point_2m,wind_speed_10m,"
                "wind_direction_10m,surface_pressure,precipitation"
            ),
            "timezone": timezone,
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
            "cell_selection": "nearest",
        },
        timeout_seconds,
    )
    frame = hourly_payload_to_frame(
        payload,
        {
            "temperature_2m": "t2m_c",
            "dew_point_2m": "d2m_c",
            "wind_speed_10m": "wind_speed_10m",
            "wind_direction_10m": "wind_direction_10m",
            "surface_pressure": "sp_hpa",
            "precipitation": "tp_mm",
        },
    )

    daily = (
        frame.assign(date=frame["datetime"].dt.date)
        .groupby("date", as_index=False)
        .agg(
            t2m_c=("t2m_c", "mean"),
            d2m_c=("d2m_c", "mean"),
            wind_speed_10m=("wind_speed_10m", "mean"),
            wind_direction_10m=("wind_direction_10m", "mean"),
            sp_hpa=("sp_hpa", "mean"),
            tp_mm=("tp_mm", "sum"),
        )
    )
    direction_rad = np.deg2rad(daily["wind_direction_10m"])
    daily["u10"] = -daily["wind_speed_10m"] * np.sin(direction_rad)
    daily["v10"] = -daily["wind_speed_10m"] * np.cos(direction_rad)
    daily["t2m"] = daily["t2m_c"] + 273.15
    daily["d2m"] = daily["d2m_c"] + 273.15
    daily["sp"] = daily["sp_hpa"] * 100.0
    daily["tp"] = daily["tp_mm"] / 1000.0
    return daily[["date", "d2m", "t2m", "u10", "v10", "sp", "tp"]]


def validate_model_data(data):
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("No rows were produced.")

    validated = data.copy()
    validated["date"] = pd.to_datetime(validated["date"], errors="coerce")
    if validated["date"].isna().any():
        raise ValueError("Some rows have invalid dates.")

    numeric_columns = sorted(REQUIRED_COLUMNS - {"date"})
    invalid_columns = [
        column
        for column in numeric_columns
        if pd.to_numeric(validated[column], errors="coerce").isna().any()
    ]
    if invalid_columns:
        raise ValueError(f"Columns contain non-numeric or missing values: {invalid_columns}")

    return validated.sort_values(["date", "lat", "lon"]).reset_index(drop=True)


def build_dataset(config, fetcher=fetch_json):
    frames = []
    for location in config.locations:
        air_quality = fetch_air_quality(
            location,
            config.start_date,
            config.end_date,
            config.timezone,
            config.timeout_seconds,
            fetcher=fetcher,
        )
        weather = fetch_weather(
            location,
            config.start_date,
            config.end_date,
            config.timezone,
            config.timeout_seconds,
            fetcher=fetcher,
        )
        merged = air_quality.merge(weather, on="date", how="inner")
        merged["lat"] = location.lat
        merged["lon"] = location.lon
        frames.append(merged)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = validate_model_data(dataset)
    dataset["month"] = dataset["date"].dt.month
    dataset["dayofyear"] = dataset["date"].dt.dayofyear
    return dataset[["lat", "lon", "aod", "d2m", "t2m", "u10", "v10", "sp", "tp", "PM25", "date", "month", "dayofyear"]]


def write_dataset(config, fetcher=fetch_json):
    dataset = build_dataset(config, fetcher=fetcher)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.output_path, index=False)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch model-ready AQI data from Open-Meteo.")
    parser.add_argument("--locations", required=True, help="Locations as 'lat,lon;lat,lon'.")
    parser.add_argument("--start-date", required=True, help="Start date as YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date as YYYY-MM-DD.")
    parser.add_argument("--output-path", type=Path, default=Path("data.csv"))
    parser.add_argument("--timezone", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = write_dataset(
        DataFetchConfig(
            locations=parse_locations(args.locations),
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output_path,
            timezone=args.timezone,
        )
    )
    print(f"Wrote {len(dataset)} rows to {args.output_path}")


if __name__ == "__main__":
    main()

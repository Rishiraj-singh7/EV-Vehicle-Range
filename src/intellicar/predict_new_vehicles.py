"""Apply the already-trained intellicar_range_model.joblib to whatever
vehicles are currently in data/raw/intellicar_can/ (new vehicles added
after the original training run).

Does NOT retrain or touch intellicar_epochs.csv -- it reuses the same epoch
extraction logic from build_epochs.py, aggregates per-vehicle typical
driving conditions the same way train_range_model.py does for its
per-vehicle table, and scores them with the saved pipeline. Vehicles unseen
during training fall back to the model's numeric-feature behaviour only
(the one-hot encoder zeroes out an unknown vehicle id), which is the point
of having trained one global model instead of per-vehicle models.

Handles vehicles that have more than one raw CSV (e.g. a June file and a
July continuation file) by concatenating all rows for that vehicle before
splitting into epochs.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv

import joblib
import pandas as pd

from src.common.paths import (
    INTELLICAR_EPOCHS_PATH,
    INTELLICAR_MODEL_PATH,
    INTELLICAR_NEW_VEHICLE_RANGE_PATH,
    INTELLICAR_RAW_DIR,
)
from src.intellicar.build_epochs import build_epochs, load_clean_rows


def vehicle_of(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = next(csv.DictReader(f), None)
    return first["vehicleNo"] if first else None


def main():
    bundle = joblib.load(INTELLICAR_MODEL_PATH)
    pipeline = bundle["pipeline"]
    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]

    known_vehicles = set()
    if INTELLICAR_EPOCHS_PATH.exists():
        known_vehicles = set(pd.read_csv(INTELLICAR_EPOCHS_PATH)["vehicle"].unique())

    files_by_vehicle = defaultdict(list)
    skipped = []
    for path in sorted(INTELLICAR_RAW_DIR.glob("*.csv")):
        v = vehicle_of(path)
        if v is None:
            skipped.append(path.name)
            continue
        files_by_vehicle[v].append(path)

    all_epochs = []
    no_usable_data = []
    for vehicle, paths in files_by_vehicle.items():
        clean = []
        for path in paths:
            rows = load_clean_rows(path)
            if not rows:
                continue
            clean.extend(rows)
        if not clean:
            no_usable_data.append(vehicle)
            continue
        clean.sort(key=lambda x: x["time"])
        epochs = build_epochs(clean, vehicle, ", ".join(p.name for p in paths))
        all_epochs.extend(epochs)

    df = pd.DataFrame(all_epochs)
    df["implied_range_km"] = df["distance_km"] / df["soc_used"] * 100
    df = df[
        (df["soc_used"] >= 5)
        & (df["distance_km"] >= 3)
        & (df["duration_hours"] <= 6)
        & (df["avg_speed"] >= 5)
    ].reset_index(drop=True)
    df["avg_batt_temp"] = df["avg_batt_temp"].fillna(df["avg_batt_temp"].median())

    print(f"Vehicles found in data/raw/intellicar_can: {len(files_by_vehicle)}")
    print(f"Vehicles with no usable soc+odometer data: {no_usable_data}")
    print(f"Clean epochs after noise filter: {len(df)} across {df['vehicle'].nunique()} vehicles")

    rows = []
    for vehicle, grp in df.groupby("vehicle"):
        rows.append(
            {
                "vehicle": vehicle,
                "avg_speed": grp["avg_speed"].median(),
                "max_speed": grp["max_speed"].median(),
                "avg_batt_temp": grp["avg_batt_temp"].median(),
                "duration_hours": grp["duration_hours"].median(),
                "odometer_start": grp["odometer_start"].median(),
                "n_epochs": len(grp),
                "seen_in_training": vehicle in known_vehicles,
            }
        )
    query = pd.DataFrame(rows)
    preds = pipeline.predict(query[numeric_features + categorical_features])
    query["predicted_range_km"] = preds.round(1)

    out = query[["vehicle", "seen_in_training", "n_epochs", "predicted_range_km"]].sort_values(
        "predicted_range_km", ascending=False
    )
    print("\nPredicted full range (100% SOC) for vehicles in current raw data:")
    print(out.to_string(index=False))

    out.to_csv(INTELLICAR_NEW_VEHICLE_RANGE_PATH, index=False)
    print(f"\nSaved to {INTELLICAR_NEW_VEHICLE_RANGE_PATH}")


if __name__ == "__main__":
    main()

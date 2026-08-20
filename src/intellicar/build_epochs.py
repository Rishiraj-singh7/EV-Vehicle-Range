"""Turn data/raw/intellicar_can/*.csv (per-vehicle CAN telemetry) into a
per-epoch training table for the Intellicar range model.

An "epoch" is one uninterrupted discharge stretch (start SOC -> lower end
SOC, no charging in between) -- see src/common/epoch_splitting.py. Extended
here with the extra features the model needs (speed, temperature,
odometer/age, duration, SOC band).

Output: data/processed/intellicar_epochs.csv, one row per epoch, columns:
    vehicle, file, start_time, end_time, duration_hours,
    odometer_start, soc_start, soc_end, soc_used, distance_km,
    avg_speed, max_speed, avg_batt_temp,
    implied_range_km   (= distance_km / soc_used * 100 -- the training target)
"""

import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv

import pandas as pd

from src.common.epoch_splitting import split_by_soc_direction
from src.common.paths import INTELLICAR_EPOCHS_PATH, INTELLICAR_RAW_DIR

REQUIRED_COLS = {"soc", "odometer", "createdAt", "vehicleNo"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_clean_rows(path):
    """Rows usable for a *discharge* epoch: needs a real (non-parked)
    odometer reading, since odometer is blank/near-zero while the vehicle
    is charging (see src/intellicar/charging.py for the charging-side view
    of this same file)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or not REQUIRED_COLS.issubset(rows[0].keys()):
        return []

    clean = []
    for r in rows:
        soc = _num(r.get("soc"))
        odo = _num(r.get("odometer"))
        if soc is None or odo is None or odo < 1000:
            continue
        clean.append(
            {
                "time": datetime.strptime(r["createdAt"][:19], "%Y-%m-%dT%H:%M:%S"),
                "soc": soc,
                "odo": odo,
                "speed": _num(r.get("vehicle_speed")),
                "temp": _num(r.get("battery_temp")),
            }
        )
    clean.sort(key=lambda x: x["time"])
    return clean


def build_epochs(clean, vehicle, filename):
    """Split into discharge segments and keep the ones that represent a
    real, measurable discharge (soc dropped, distance covered)."""
    epochs = []
    for seg in split_by_soc_direction(clean, get_soc=lambda r: r["soc"], rising=False):
        if len(seg) < 2:
            continue
        soc_start, soc_end = seg[0]["soc"], seg[-1]["soc"]
        drop = soc_start - soc_end
        dist = seg[-1]["odo"] - seg[0]["odo"]
        if drop <= 0.01 or dist <= 0:
            continue
        duration_h = (seg[-1]["time"] - seg[0]["time"]).total_seconds() / 3600
        if duration_h <= 0:
            continue

        speeds = [r["speed"] for r in seg if r["speed"] is not None]
        temps = [r["temp"] for r in seg if r["temp"] is not None]
        epochs.append(
            {
                "vehicle": vehicle,
                "file": filename,
                "start_time": seg[0]["time"],
                "end_time": seg[-1]["time"],
                "duration_hours": duration_h,
                "odometer_start": seg[0]["odo"],
                "soc_start": soc_start,
                "soc_end": soc_end,
                "soc_used": drop,
                "distance_km": dist,
                "avg_speed": mean(speeds) if speeds else None,
                "max_speed": max(speeds) if speeds else None,
                "avg_batt_temp": mean(temps) if temps else None,
            }
        )
    return epochs


def main():
    files = sorted(INTELLICAR_RAW_DIR.glob("*.csv"))
    all_epochs = []
    skipped = []

    for path in files:
        clean = load_clean_rows(path)
        if not clean:
            skipped.append(path.name)
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            first = next(csv.DictReader(f))
        vehicle = first["vehicleNo"]

        epochs = build_epochs(clean, vehicle, path.name)
        all_epochs.extend(epochs)

    df = pd.DataFrame(all_epochs)
    df["implied_range_km"] = df["distance_km"] / df["soc_used"] * 100

    # Drop noisy epochs. SOC is reported at ~1% resolution, so small soc_used
    # values blow up the implied-range ratio (e.g. 1 km over 1% SOC => a
    # nonsensical 100 km/100% reading); long low-speed epochs are mostly
    # standby/idle drain between short drives, not real driving efficiency.
    before = len(df)
    df = df[
        (df["soc_used"] >= 5)
        & (df["distance_km"] >= 3)
        & (df["duration_hours"] <= 6)
        & (df["avg_speed"] >= 5)
    ].reset_index(drop=True)
    print(
        f"Epochs extracted: {before}, kept after noise filter "
        f"(soc_used>=5%, dist>=3km, duration<=6h, avg_speed>=5km/h): {len(df)}"
    )

    if skipped:
        print(f"Skipped files (no usable soc+odometer): {skipped}")

    INTELLICAR_EPOCHS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTELLICAR_EPOCHS_PATH, index=False)
    print(f"Wrote {len(df)} rows to {INTELLICAR_EPOCHS_PATH}")
    print(f"Vehicles covered: {df['vehicle'].nunique()}")
    print(df.groupby("vehicle").size().sort_values(ascending=False))


if __name__ == "__main__":
    main()

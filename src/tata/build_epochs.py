"""Turn data/raw/tata_gps/*.csv (5-min GPS+CAN pings, Tata EV fleet) into a
per-epoch training table for a range model dedicated to this fleet.

Same "discharge epoch" idea as src/intellicar/build_epochs.py (split the SOC
timeline at every charge event -- see src/common/epoch_splitting.py), but
tailored to this dataset's quirks and richer signal set:

  - data.odometer reads exactly 0 whenever ignitionOn=false -- a sleep/wake
    artifact (confirmed: 100% of odo==0 rows have ignitionOn=false, and the
    value resumes exactly where it left off afterwards), not a real
    odometer reset. Those rows are dropped, same idea as the odo<1000
    filter in the Intellicar pipeline.
  - data.distanceToEmpty is unreliable in this dataset (hits a sentinel
    1023 at every SOC level, and swings wildly even at SOC=100%), so it is
    NOT used, unlike the old CAN pipeline's dte_cross_check.
  - "Driving" rows are identified directly via ignitionOn=true and
    evGearInfo in {4,5} (confirmed against speed: those are the only gears
    with meaningfully nonzero speed), rather than inferred from a speed
    threshold.
  - No battery-temperature signal exists here, so it's replaced with two
    new features this dataset does have: % of driving time with AC on,
    and an accelerometer-based harsh-driving proxy (avg/max of the X/Y
    acceleration magnitude -- Z is excluded, it's gravity-dominated at
    rest, not driving dynamics).

Output: data/processed/tata_epochs.csv, one row per epoch.
"""

import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv

import pandas as pd

from src.common.epoch_splitting import split_by_soc_direction
from src.common.paths import TATA_EPOCHS_PATH, TATA_RAW_DIR

DRIVING_GEARS = {"4", "5"}
PING_INTERVAL_MIN = 5  # nominal reporting cadence


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_clean_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    clean = []
    for r in rows:
        soc = _num(r.get("data.hvBattSocPercentage"))
        odo = _num(r.get("data.odometer"))
        if soc is None or odo is None or odo <= 0:
            continue  # odo==0 -> ignition-off sleep/wake glitch, not real
        ax = _num(r.get("data.accelX")) or 0.0
        ay = _num(r.get("data.accelY")) or 0.0
        clean.append(
            {
                # createdAt is the server ingestion time; data.eventDateTime
                # in this export is offset by ~5.5h in a way consistent
                # with an IST/UTC labeling bug in the device, so createdAt
                # is used as the trustworthy, consistently-ordered clock.
                "time": datetime.strptime(r["createdAt"][:19], "%Y-%m-%dT%H:%M:%S"),
                "soc": soc,
                "odo": odo,
                "speed": _num(r.get("data.speed")),
                "driving": (
                    r.get("data.ignitionOn") == "true"
                    and r.get("data.evGearInfo") in DRIVING_GEARS
                ),
                "ac_on": r.get("data.acState") == "true",
                "accel_xy": (ax**2 + ay**2) ** 0.5,
                "altitude": _num(r.get("data.gpsAltitude")),
            }
        )
    clean.sort(key=lambda x: x["time"])
    return clean


def build_epochs(clean, vehicle, filename):
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

        driving_rows = [r for r in seg if r["driving"]]
        if len(driving_rows) < 2:
            continue  # too little confirmed driving to trust the averages

        speeds = [r["speed"] for r in driving_rows if r["speed"] is not None]
        alts = [r["altitude"] for r in driving_rows if r["altitude"] is not None]
        accels = [r["accel_xy"] for r in driving_rows]

        epochs.append(
            {
                "vehicle": vehicle,
                "file": filename,
                "start_time": seg[0]["time"],
                "end_time": seg[-1]["time"],
                "duration_hours": duration_h,
                "driving_minutes": len(driving_rows) * PING_INTERVAL_MIN,
                "odometer_start": seg[0]["odo"],
                "soc_start": soc_start,
                "soc_end": soc_end,
                "soc_used": drop,
                "distance_km": dist,
                "avg_speed": mean(speeds) if speeds else None,
                "max_speed": max(speeds) if speeds else None,
                "pct_ac_on": 100 * sum(r["ac_on"] for r in driving_rows) / len(driving_rows),
                "avg_accel_xy": mean(accels),
                "max_accel_xy": max(accels),
                "avg_altitude": mean(alts) if alts else None,
                "altitude_range": (max(alts) - min(alts)) if alts else None,
            }
        )
    return epochs


def main():
    files = sorted(TATA_RAW_DIR.glob("*.csv"))
    all_epochs = []

    for path in files:
        clean = load_clean_rows(path)
        if not clean:
            print(f"Skipping {path.name}: no usable rows")
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            first = next(csv.DictReader(f))
        vehicle = first["vehicleNo"]
        epochs = build_epochs(clean, vehicle, path.name)
        all_epochs.extend(epochs)

    df = pd.DataFrame(all_epochs)
    df["implied_range_km"] = df["distance_km"] / df["soc_used"] * 100

    before = len(df)
    df = df[
        (df["soc_used"] >= 5)
        & (df["distance_km"] >= 3)
        & (df["duration_hours"] <= 8)
        & (df["driving_minutes"] >= 10)
    ].reset_index(drop=True)
    print(
        f"\nEpochs extracted: {before}, kept after noise filter "
        f"(soc_used>=5%, dist>=3km, duration<=8h, driving_minutes>=10): {len(df)}"
    )

    TATA_EPOCHS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TATA_EPOCHS_PATH, index=False)
    print(f"Wrote {len(df)} rows to {TATA_EPOCHS_PATH}")
    print(f"Vehicles covered: {df['vehicle'].nunique()}")
    print(df.groupby("vehicle").size().sort_values(ascending=False))
    print(f"\nimplied_range_km stats:\n{df['implied_range_km'].describe()}")


if __name__ == "__main__":
    main()

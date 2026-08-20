"""Build the single, centralised training table spanning all three devices.

    data/raw/intellicar_can/*.csv  ]
    data/raw/tata_gps/*.csv        ]--> data/processed/unified_epochs.csv
    data/raw/citroen/*.csv         ]

One row per discharge epoch, with the same columns regardless of which
device produced it -- so one model trains on the whole fleet instead of one
model per telematics vendor.

Pipeline per file:
    device adapter (src/common/device_adapters.py)   raw CSV -> normalized rows
    epochs_from_rows (src/common/epoch_pipeline.py)  -> epochs with unified features

All the rules -- noise filters, the odometer-jump guard, deduplication,
rest_hours_before -- live in src/common/epoch_pipeline.py, because the
webapp's live path has to apply exactly the same ones or it would serve
predictions computed on a different distribution than the model was fit on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.device_adapters import LOADERS, vehicle_of
from src.common.epoch_pipeline import (
    MAX_ACTIVE_HOURS,
    MIN_AVG_SPEED,
    MIN_DISTANCE_KM,
    MIN_MOVING_HOURS,
    MIN_SOC_USED,
    add_rest_hours,
    add_target,
    apply_quality_filters,
    dedupe,
    epochs_from_rows,
)
from src.common.paths import (
    CITROEN_RAW_DIR,
    INTELLICAR_RAW_DIR,
    TATA_RAW_DIR,
    UNIFIED_EPOCHS_PATH,
)

SOURCES = [
    ("Intellicar", INTELLICAR_RAW_DIR),
    ("Tata", TATA_RAW_DIR),
    ("Citroen", CITROEN_RAW_DIR),
]


def build_for_device(device, raw_dir):
    rows = []
    loader = LOADERS[device]
    files = sorted(p for p in raw_dir.glob("*.csv"))
    if not files:
        print(f"  {device}: no CSVs in {raw_dir}")
        return rows

    for path in files:
        try:
            clean = loader(path)
        except Exception as e:  # noqa: BLE001 - one bad export shouldn't kill the build
            print(f"  {device}: {path.name} failed to load ({type(e).__name__}: {e})")
            continue
        if len(clean) < 2:
            continue
        vehicle = vehicle_of(path)
        if not vehicle:
            print(f"  {device}: {path.name} has no vehicleNo, skipping")
            continue
        rows.extend(epochs_from_rows(clean, vehicle, device, path.name))

    print(f"  {device}: {len(rows)} raw epochs from {len(files)} files")
    return rows


def main():
    print("Building unified epoch table")
    all_rows = []
    for device, raw_dir in SOURCES:
        all_rows.extend(build_for_device(device, raw_dir))

    if not all_rows:
        raise SystemExit("No epochs built -- is data/raw/ populated?")

    df = add_target(pd.DataFrame(all_rows))

    before_dedup = len(df)
    df = dedupe(df)
    if before_dedup != len(df):
        print(f"\nDeduplicated overlapping exports: {before_dedup} -> {len(df)} epochs")

    df = add_rest_hours(df)

    before = len(df)
    df = apply_quality_filters(df, report=print)

    lead = ["device", "vehicle", "file", "start_time", "end_time"]
    df = df[lead + [c for c in df.columns if c not in lead]]

    UNIFIED_EPOCHS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIFIED_EPOCHS_PATH, index=False)

    print(
        f"\nEpochs: {before} deduped -> {len(df)} kept "
        f"(soc_used>={MIN_SOC_USED}%, dist>={MIN_DISTANCE_KM}km, "
        f"avg_speed>={MIN_AVG_SPEED}km/h, moving>={MIN_MOVING_HOURS}h, "
        f"active<={MAX_ACTIVE_HOURS}h, odometer/speed consistent)"
    )
    print(f"Wrote {UNIFIED_EPOCHS_PATH}")

    print("\nPer-device coverage:")
    summary = df.groupby("device").agg(
        epochs=("implied_range_km", "size"),
        vehicles=("vehicle", "nunique"),
        median_range=("implied_range_km", "median"),
        median_avg_speed=("avg_speed", "median"),
    ).round(1)
    print(summary.to_string())

    print("\nMissing values per candidate feature:")
    miss = df.isna().sum()
    print(miss[miss > 0].to_string() if miss.any() else "  none")


if __name__ == "__main__":
    main()

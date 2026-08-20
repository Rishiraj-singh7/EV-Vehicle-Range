"""The epoch-building rules, in one place, so the batch trainer and the live
webapp cannot drift apart.

This matters more than it looks. If the live service filtered epochs even
slightly differently from src/unified/build_epochs.py, every prediction it
served would be computed on a different distribution than the model was fit
on -- and nothing would visibly break, the numbers would just quietly be
wrong. Both paths therefore call the same three functions here:

    epochs_from_rows()      normalized rows  -> raw epoch dicts
    apply_quality_filters() raw epochs       -> the ones fit to train/predict on
    add_rest_hours()        fills the one cross-epoch feature

Every threshold below is shared by both paths for the same reason.
"""

import pandas as pd

from .epoch_splitting import split_by_soc_direction
from .unified_features import epoch_features

# --- noise filters, identical for all three devices ---------------------
# The legacy per-device pipelines used different thresholds (Intellicar
# capped duration at 6h, Tata at 8h, and Tata alone required 10 driving
# minutes), which would bias a merged table toward whichever fleet got the
# looser rule.
MIN_SOC_USED = 5.0        # SOC is integer-resolution on Intellicar; small drops blow up the ratio
MIN_DISTANCE_KM = 3.0
MIN_AVG_SPEED = 5.0       # below this it's standby drain, not driving
MIN_MOVING_HOURS = 0.15   # ~9 minutes of actual movement
MAX_ACTIVE_HOURS = 12.0

# rest_hours_before saturates: past roughly a day parked, more parking tells
# the model nothing new, and uncapped multi-week gaps (vehicle off the road)
# would otherwise dominate the feature's scale.
REST_HOURS_CAP = 24.0

# --- odometer-jump guard ------------------------------------------------
# The odometer and the speed trace are two independent measurements of the
# same journey, so distance / moving time must land near the measured
# average speed. A device swap, counter rollover or dropped-then-resumed
# feed makes the odometer leap and the two disagree by 10x or more -- one
# Tata epoch claimed 1409 km of travel and a 15,656 km implied range.
# Because the target is a ratio, a handful of those dominates the error on a
# small fleet.
#
# The band is deliberately wide: a factor of 2 either way tolerates genuine
# GPS-vs-odometer disagreement and coarse 5-minute sampling missing speed
# peaks, while still catching jumps.
MIN_ODO_SPEED_RATIO = 0.4
MAX_ODO_SPEED_RATIO = 2.5


def epochs_from_rows(clean_rows, vehicle, device, source_name):
    """Normalized rows (from a device adapter) -> list of raw epoch dicts.

    No filtering here -- apply_quality_filters() does that, so callers can
    see how many epochs were found before and after."""
    epochs = []
    for seg in split_by_soc_direction(clean_rows, get_soc=lambda r: r["soc"], rising=False):
        feats = epoch_features(seg)
        if feats is None:
            continue
        feats["device"] = device
        feats["vehicle"] = vehicle
        feats["file"] = source_name
        epochs.append(feats)
    return epochs


def add_target(df):
    df = df.copy()
    df["implied_range_km"] = df["distance_km"] / df["soc_used"] * 100
    return df


def apply_quality_filters(df, report=None):
    """Drop epochs that are too small/short to measure range from, then drop
    the ones whose odometer disagrees with their speed trace.

    `report` -- optional callable(str) for progress messages; the batch
    builder prints, the webapp stays quiet."""
    if df.empty:
        return df

    df = df[
        (df["soc_used"] >= MIN_SOC_USED)
        & (df["distance_km"] >= MIN_DISTANCE_KM)
        & (df["avg_speed"] >= MIN_AVG_SPEED)
        & (df["moving_hours"] >= MIN_MOVING_HOURS)
        & (df["active_hours"] <= MAX_ACTIVE_HOURS)
    ]
    if df.empty:
        return df

    ratio = (df["distance_km"] / df["moving_hours"]) / df["avg_speed"]
    consistent = ratio.between(MIN_ODO_SPEED_RATIO, MAX_ODO_SPEED_RATIO)
    if report is not None and (~consistent).any():
        rejected = df[~consistent]
        report("\nOdometer/speed inconsistency rejects (device: n, worst implied range):")
        for dev, grp in rejected.groupby("device"):
            report(f"  {dev}: {len(grp)}, worst {grp['implied_range_km'].max():.0f} km")
    return df[consistent].reset_index(drop=True)


def add_rest_hours(df):
    """Idle time since the same vehicle's previous epoch ended.

    Needs the whole per-vehicle sequence, so it can't be computed inside
    epoch_features() one segment at a time. Physically it proxies both how
    long the pack sat cooling and how recently it charged -- the closest
    universal stand-in for the battery temperature only Intellicar reports."""
    if df.empty:
        return df
    df = df.copy()
    # Rows built in-process carry real datetimes; rows read back from the
    # central CSV arrive as strings. Coerce both so the subtraction below
    # works either way.
    for col in ("start_time", "end_time"):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.sort_values(["vehicle", "start_time"])
    prev_end = df.groupby("vehicle")["end_time"].shift()
    df["rest_hours_before"] = (
        (df["start_time"] - prev_end).dt.total_seconds() / 3600.0
    ).clip(upper=REST_HOURS_CAP)
    # First epoch per vehicle has no predecessor; median-fill so the column
    # has no gaps rather than silently dropping those rows later.
    median = df["rest_hours_before"].median()
    df["rest_hours_before"] = df["rest_hours_before"].fillna(
        0.0 if pd.isna(median) else median
    )
    return df


def dedupe(df):
    """Same discharge, pulled twice.

    data/raw/ holds overlapping export windows for many vehicles --
    historical bulk pulls, the webapp's trailing-30-day `live-` caches, and
    (for Intellicar) several bulk pulls overlapping each other. Left alone,
    ~25% of rows were re-counted copies of the same real epoch, which skews
    the training distribution and each vehicle's effective weight.

    An epoch is identified by (vehicle, start_time). Where a window boundary
    clipped one copy short, the copy with the larger soc_used is the more
    complete one, so sort by it and keep the first."""
    if df.empty:
        return df
    return (
        df.sort_values("soc_used", ascending=False)
        .drop_duplicates(subset=["vehicle", "start_time"], keep="first")
        .drop_duplicates(subset=["vehicle", "end_time"], keep="first")
    )

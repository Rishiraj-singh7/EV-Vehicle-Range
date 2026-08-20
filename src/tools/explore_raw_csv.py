import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.paths import INTELLICAR_RAW_DIR, TATA_RAW_DIR


def load_rows(path):
    # utf-8-sig strips a leading BOM if present (e.g. intellicar exports),
    # and is harmless for plain utf-8 files.
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_event_level(rows):
    return "Status" in rows[0]


def is_raw_telemetry(rows):
    return "soc" in rows[0] and "odometer" in rows[0] and "createdAt" in rows[0]


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def compute_epochs(rows):
    """Split the SOC timeline at every charging event (mid-trip, idle, or
    overnight) so each epoch is a pure discharge stretch: no hidden charging
    mixed into the distance/SOC-used numbers.

    Rows with a missing/non-numeric SOC reading (e.g. "null") are dropped;
    they carry no distance so skipping them loses no driving data.

    Returns a list of (start_soc, end_soc, distance_km, soc_used) tuples.
    """
    rows = [r for r in rows if _is_number(r["Start SOC"]) and _is_number(r["End SOC"])]
    if not rows:
        return []

    epochs = []
    cur_epoch_start_soc = float(rows[0]["Start SOC"])
    cur_distance = 0.0

    def close_epoch(end_soc):
        nonlocal cur_epoch_start_soc, cur_distance
        drop = cur_epoch_start_soc - end_soc
        if drop > 0 and cur_distance > 0:
            epochs.append((cur_epoch_start_soc, end_soc, cur_distance, drop))
        cur_epoch_start_soc = end_soc
        cur_distance = 0.0

    prev_end = float(rows[0]["Start SOC"])
    for r in rows:
        s = float(r["Start SOC"])
        e = float(r["End SOC"])
        dist = float(r["Distance (km)"])

        if s > prev_end:
            close_epoch(prev_end)
            cur_epoch_start_soc = s

        if e > s:
            close_epoch(s)
            cur_epoch_start_soc = e
        else:
            cur_distance += dist

        prev_end = e

    close_epoch(prev_end)
    return epochs


def compute_telemetry_epochs(rows):
    """For raw CAN-log telemetry (one row per ~minute with soc + cumulative
    odometer): sort by time, drop unusable rows (missing soc/odometer, or an
    odometer glitch reading near 0), then split into pure-discharge epochs
    the same way as compute_epochs, using odometer deltas as distance.
    """
    clean = []
    for r in rows:
        if r["soc"].strip() == "" or r["odometer"].strip() == "":
            continue
        odo = float(r["odometer"])
        if odo < 1000:
            continue
        clean.append(
            {
                "time": datetime.strptime(r["createdAt"][:19], "%Y-%m-%dT%H:%M:%S"),
                "soc": float(r["soc"]),
                "odo": odo,
            }
        )
    clean.sort(key=lambda x: x["time"])
    if not clean:
        return []

    epochs = []
    epoch_start_soc = clean[0]["soc"]
    epoch_start_odo = clean[0]["odo"]
    prev_soc = epoch_start_soc
    prev_odo = epoch_start_odo

    def close_epoch(end_soc, end_odo):
        nonlocal epoch_start_soc, epoch_start_odo
        drop = epoch_start_soc - end_soc
        dist = end_odo - epoch_start_odo
        if drop > 0.01 and dist > 0:
            epochs.append((epoch_start_soc, end_soc, dist, drop))
        epoch_start_soc = end_soc
        epoch_start_odo = end_odo

    for row in clean[1:]:
        if row["soc"] > prev_soc + 0.01:
            close_epoch(prev_soc, prev_odo)
        prev_soc = row["soc"]
        prev_odo = row["odo"]

    close_epoch(prev_soc, prev_odo)
    return epochs


def dte_cross_check(rows, soc_threshold=85.0):
    """Cross-check using the vehicle's own onboard 'distance to empty'
    estimate: at a given SOC, full_range ~= dte / (soc/100). Restricting to
    high-SOC readings avoids the non-linear reserve-buffer behaviour EVs
    often show near empty.
    """
    implied = []
    for r in rows:
        if r.get("dte", "").strip() == "" or r["soc"].strip() == "":
            continue
        soc = float(r["soc"])
        if soc < soc_threshold:
            continue
        dte = float(r["dte"])
        implied.append(dte / (soc / 100))

    if not implied:
        return None
    return {
        "count": len(implied),
        "mean": statistics.mean(implied),
        "median": statistics.median(implied),
        "min": min(implied),
        "max": max(implied),
    }


def compute_daily_valid_days(rows):
    """For daily-aggregated logs: keep only days where End SOC < Start SOC
    (no net charging during the day). Returns (start_time, distance, soc_used)
    tuples for those days.
    """
    valid = []
    for r in rows:
        dist = float(r["Distance (km)"])
        start = float(r["Start SOC"])
        end = float(r["End SOC"])
        if dist == 0 and start == 0 and end == 0:
            continue
        if end < start:
            valid.append((r["Start Time"], dist, start - end))
    return valid


def summarize(distances_and_soc, label):
    total_dist = sum(d for d, _ in distances_and_soc)
    total_soc = sum(s for _, s in distances_and_soc)
    ratios = [d / s for d, s in distances_and_soc]

    print(f"\n=== {label} ===")
    print(f"Segments used: {len(distances_and_soc)}")
    print(f"Total distance: {total_dist:.2f} km")
    print(f"Total SOC consumed: {total_soc:.0f} %")
    print(f"Efficiency: {total_dist / total_soc:.3f} km per 1% SOC")
    print(f"Estimated full range (100% SOC): {total_dist / total_soc * 100:.1f} km")
    print(
        f"Per-segment ratio: min {min(ratios):.2f}, max {max(ratios):.2f}, "
        f"mean {statistics.mean(ratios):.2f}, median {statistics.median(ratios):.2f}, "
        f"stdev {statistics.stdev(ratios):.2f}"
    )


def analyze_group(rows, vehicle, filename, fmt):
    if fmt == "telemetry":
        epochs = compute_telemetry_epochs(rows)
        distances_and_soc = [(ep[2], ep[3]) for ep in epochs]
        summarize(distances_and_soc, f"{vehicle} (raw telemetry, {filename})")

        cross = dte_cross_check(rows)
        if cross:
            print(
                f"DTE cross-check (onboard 'distance to empty', SOC>85%, "
                f"n={cross['count']}): mean {cross['mean']:.1f} km, "
                f"median {cross['median']:.1f} km, range "
                f"{cross['min']:.1f}-{cross['max']:.1f} km"
            )
    elif fmt == "event":
        epochs = compute_epochs(rows)
        distances_and_soc = [(ep[2], ep[3]) for ep in epochs]
        summarize(distances_and_soc, f"{vehicle} (event-level, {filename})")
    else:
        valid_days = compute_daily_valid_days(rows)
        distances_and_soc = [(d, s) for _, d, s in valid_days]
        summarize(distances_and_soc, f"{vehicle} (daily-aggregated, {filename})")


def analyze_file(path):
    rows = load_rows(path)
    if not rows:
        print(f"{path}: no data")
        return

    if is_raw_telemetry(rows):
        fmt = "telemetry"
        vehicle_key = "vehicleNo"
    elif is_event_level(rows):
        fmt = "event"
        vehicle_key = "Vehicle No"
    else:
        fmt = "daily"
        vehicle_key = "Vehicle No"

    vehicles = sorted(set(r.get(vehicle_key, "") for r in rows))

    if len(vehicles) <= 1:
        vehicle = vehicles[0] if vehicles else path.stem
        analyze_group(rows, vehicle, path.name, fmt)
        return

    print(f"\n{'#' * 60}")
    print(f"# {path.name}: {len(vehicles)} vehicles found -> {', '.join(vehicles)}")
    print(f"{'#' * 60}")
    for vehicle in vehicles:
        sub_rows = [r for r in rows if r.get(vehicle_key) == vehicle]
        analyze_group(sub_rows, vehicle, path.name, fmt)


def main():
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(INTELLICAR_RAW_DIR.glob("*.csv")) + sorted(TATA_RAW_DIR.glob("*.csv"))

    if not paths:
        print("No CSV files found.")
        return

    for path in paths:
        analyze_file(path)


if __name__ == "__main__":
    main()

"""Charging-session detection for Intellicar CAN telemetry.

Unlike build_epochs.py's discharge-epoch loader, this does NOT filter rows
by odometer -- odometer reads blank while the vehicle is parked and
charging, which is exactly the data this needs, so dropping those rows
would erase the charging signal instead of just noise.
"""

import csv
from datetime import datetime

from src.common.charging_sessions import find_charging_sessions


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool01(v):
    v = (v or "").strip()
    if v == "1":
        return True
    if v == "0":
        return False
    return None  # blank/unknown -> caller falls back to rate-based classification


def load_charging_rows(csv_path):
    """Minimal rows needed for charging-session detection: time, soc, and
    the device's own fast-charge flag (if present)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    clean = []
    for r in rows:
        soc = _num(r.get("soc"))
        created = (r.get("createdAt") or "").strip()
        if soc is None or not created:
            continue
        clean.append(
            {
                "time": datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S"),
                "soc": soc,
                "fast_flag": _bool01(r.get("fast_charge_indicator")),
            }
        )
    return clean


def charging_sessions_for(csv_path, vehicle):
    """All qualifying charging sessions (soc gained > 10%) found in one
    vehicle's raw CAN CSV, oldest first."""
    rows = load_charging_rows(csv_path)
    return find_charging_sessions(
        rows,
        vehicle,
        get_time=lambda r: r["time"],
        get_soc=lambda r: r["soc"],
        get_fast_flag=lambda r: r["fast_flag"],
    )

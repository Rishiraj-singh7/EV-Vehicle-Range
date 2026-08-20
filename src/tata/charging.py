"""Charging-session detection for the Tata fleet, using the dedicated
"type=charging" export (src/common/tata_api_client.fetch_charging_csv) --
not the gps feed. That feed is charging-specific telemetry (SOC,
data.hvChargingState, data.hvChargeType) sampled while the vehicle is
plugged in, and is much cleaner for this purpose than picking charging
periods out of the general gps ping stream.

data.hvChargeType looked like it might be a ready-made fast/slow flag (like
Intellicar's fast_charge_indicator), but checked against real sessions'
measured charge rate it showed no clean separation (both values spanned
the same wide rate range) -- so, unlike Intellicar, sessions here fall back
to the rate-based classification in charging_sessions.py
(get_fast_flag=None).
"""

import csv
from datetime import datetime

from src.common.charging_sessions import find_charging_sessions


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_charging_rows(csv_path):
    """Minimal rows needed for charging-session detection: time + soc."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    clean = []
    for r in rows:
        soc = _num(r.get("data.hvBattSocPercentage"))
        created = (r.get("createdAt") or "").strip()
        if soc is None or not created:
            continue
        clean.append(
            {
                "time": datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S"),
                "soc": soc,
            }
        )
    return clean


def charging_sessions_for(csv_path, vehicle):
    """All qualifying charging sessions (soc gained > 10%) found in one
    vehicle's charging-feed CSV, oldest first."""
    rows = load_charging_rows(csv_path)
    return find_charging_sessions(
        rows,
        vehicle,
        get_time=lambda r: r["time"],
        get_soc=lambda r: r["soc"],
        get_fast_flag=None,
    )

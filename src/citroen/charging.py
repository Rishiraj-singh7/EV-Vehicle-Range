"""Charging-session detection for the Citroen fleet.

Citroen needs no separate charging endpoint: its single feed already carries
the charging signals alongside the driving ones, so the same CSV the range
prediction is built from also yields the sessions.

Unlike Tata, this fleet reports a **usable fast/slow flag**. Two fields
cover it, and they agree cleanly on real data:

    batteryInfo.chargingStatus  HMI_charge_disconnected / _inprogress / _finished
    batteryInfo.typeOfCharge    No_charging / Quick_charging / Slow_charging

Checked over a sample month, every `_finished` and essentially every
`_inprogress` row carries a non-`No_charging` type, and no `_disconnected`
row does -- i.e. the type field is populated exactly when the vehicle is
actually plugged in. So `Quick_charging` is taken as the device's own
fast-charge flag, the same standing Intellicar's fast_charge_indicator has,
rather than falling back to the 15%/hr rate cutoff that Tata needs.

That also makes this fleet the natural place to validate that cutoff: these
sessions have both a device flag and a measured rate, so the two can be
compared directly. See the README's "Known limitations".
"""

import csv
from datetime import datetime

from src.common.charging_sessions import find_charging_sessions

SOC_COL = "data.vehicleData.batteryInfo.soc"
STATUS_COL = "data.vehicleData.batteryInfo.chargingStatus"
TYPE_COL = "data.vehicleData.batteryInfo.typeOfCharge"

QUICK = "Quick_charging"
SLOW = "Slow_charging"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_charging_rows(csv_path):
    """Minimal rows for session detection: time, soc, and the device's own
    charge-type flag (None when the vehicle isn't plugged in, so
    find_charging_sessions ignores it rather than counting it as 'slow')."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    clean = []
    for r in rows:
        soc = _num(r.get(SOC_COL))
        created = (r.get("createdAt") or "").strip()
        if soc is None or not created:
            continue
        charge_type = (r.get(TYPE_COL) or "").strip()
        if charge_type == QUICK:
            fast_flag = True
        elif charge_type == SLOW:
            fast_flag = False
        else:
            fast_flag = None  # not charging -- no opinion
        clean.append(
            {
                "time": datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S"),
                "soc": soc,
                "fast_flag": fast_flag,
            }
        )
    return clean


def charging_sessions_for(csv_path, vehicle):
    """All qualifying charging sessions (soc gained > 10%) in one vehicle's
    Citroen CSV, oldest first, classified by the device's own flag."""
    rows = load_charging_rows(csv_path)
    return find_charging_sessions(
        rows,
        vehicle,
        get_time=lambda r: r["time"],
        get_soc=lambda r: r["soc"],
        get_fast_flag=lambda r: r["fast_flag"],
    )

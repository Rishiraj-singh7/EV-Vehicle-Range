"""Raw per-device CSV -> one normalized row schema, so a single feature
builder (src/common/unified_features.py) and a single model can span all
three telematics devices.

The three feeds agree on almost nothing at the column level. Their *raw*
intersection is only four signals:

    time (createdAt), soc, odometer, speed

Everything richer is device-specific and therefore unusable in a merged
model as a feature:

    battery temperature   Intellicar only
    AC state              Tata + Citroen only  (and constant-on in the Tata sample)
    ignition / gear       Tata + Citroen only
    GPS altitude          Tata only
    accelerometer         Tata only (Citroen has hard-brake/accel event counts instead)
    charge type flag      Intellicar + Citroen only

So the normalized row carries exactly the four common signals plus the
identity columns. Device-specific *cleaning* still happens here (each feed
has its own junk to drop); device-specific *features* deliberately do not
survive this layer -- that's the whole point of the merge.

Normalized row:
    {"time": datetime, "soc": float, "odo": float, "speed": float|None}
"""

import csv
from datetime import datetime

# Above this, a speed reading is a sentinel/garbage rather than a real
# reading, and is treated as missing: Citroen emits a literal 199 (55 rows
# in the probe sample, with nothing at all between 90 and 199), and
# Intellicar has stray 135s on vehicles whose real ceiling is ~90.
MAX_PLAUSIBLE_SPEED_KMH = 120.0


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _speed(v):
    s = _num(v)
    if s is None or s < 0 or s >= MAX_PLAUSIBLE_SPEED_KMH:
        return None
    return s


def _time(v):
    try:
        return datetime.strptime(str(v)[:19], "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _finish(clean):
    clean.sort(key=lambda r: r["time"])
    return clean


def load_intellicar(path):
    """CAN feed. Odometer reads near-zero while parked/charging, so rows
    under 1000 km are dropped -- the same filter the original
    src/intellicar/build_epochs.py applies."""
    clean = []
    for r in _read(path):
        soc, odo, t = _num(r.get("soc")), _num(r.get("odometer")), _time(r.get("createdAt"))
        if soc is None or odo is None or t is None or odo < 1000:
            continue
        clean.append({"time": t, "soc": soc, "odo": odo, "speed": _speed(r.get("vehicle_speed"))})
    return _finish(clean)


def load_tata(path):
    """GPS+CAN feed. data.odometer reads exactly 0 whenever ignitionOn is
    false -- a sleep/wake artifact, not a real reset -- so odo<=0 is
    dropped. createdAt (server ingest time) is used rather than
    data.eventDateTime, which is offset ~5.5h by an IST/UTC device bug."""
    clean = []
    for r in _read(path):
        soc = _num(r.get("data.hvBattSocPercentage"))
        odo = _num(r.get("data.odometer"))
        t = _time(r.get("createdAt"))
        if soc is None or odo is None or t is None or odo <= 0:
            continue
        clean.append({"time": t, "soc": soc, "odo": odo, "speed": _speed(r.get("data.speed"))})
    return _finish(clean)


def load_citroen(path):
    """Single combined feed. Cleanest of the three: the odometer is strictly
    monotonic with no sleep artifact (verified over the sample -- zero
    backward steps), and SOC is reported at sub-1% resolution rather than
    Intellicar's integer steps. Only the 199 km/h speed sentinel needs
    dropping, which _speed() handles."""
    clean = []
    for r in _read(path):
        soc = _num(r.get("data.vehicleData.batteryInfo.soc"))
        odo = _num(r.get("data.vehicleData.odoMeterReadingInKM"))
        t = _time(r.get("createdAt"))
        if soc is None or odo is None or t is None or odo <= 0:
            continue
        clean.append(
            {
                "time": t,
                "soc": soc,
                "odo": odo,
                "speed": _speed(r.get("data.vehicleData.vehicleSpeedInKMph")),
            }
        )
    return _finish(clean)


LOADERS = {
    "Intellicar": load_intellicar,
    "Tata": load_tata,
    "Citroen": load_citroen,
}

VEHICLE_COLUMN = "vehicleNo"  # all three feeds happen to agree on this one


def vehicle_of(path):
    """Vehicle number from the file's first data row (all three feeds carry
    a `vehicleNo` column, and each file covers exactly one vehicle)."""
    for r in _read(path):
        v = (r.get(VEHICLE_COLUMN) or "").strip().upper()
        if v:
            return v
    return None

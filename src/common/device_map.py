"""Vehicle -> telematics device lookup, loaded from
data/reference/vehicle_device_map.csv.

The CSV's raw values ("Intellicar", "Tata", "Citroen", "NO_DEVICE") are
normalized to lowercase keys used everywhere else in the app.
"""

import csv
from functools import lru_cache

from .paths import VEHICLE_DEVICE_MAP_PATH

DEVICE_KEYS = ("intellicar", "tata", "citroen", "no_device")


def _normalize(raw_device_type):
    key = (raw_device_type or "").strip().lower().replace(" ", "_")
    return key if key in DEVICE_KEYS else "no_device"


@lru_cache(maxsize=None)
def _load_map():
    """vehicle_number -> device_key, cached for the life of the process."""
    mapping = {}
    with open(VEHICLE_DEVICE_MAP_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vehicle = row["vehicleNumber"].strip().upper()
            mapping[vehicle] = _normalize(row["telematicDeviceType"])
    return mapping


def device_of(vehicle_number):
    """device_key for a vehicle, or None if it isn't in the reference sheet."""
    return _load_map().get(vehicle_number.strip().upper())


def vehicles_for_device(device_key):
    """Sorted vehicle numbers whose telematicDeviceType matches device_key."""
    return sorted(v for v, d in _load_map().items() if d == device_key)


def all_vehicles_with_device():
    """Every (vehicle_number, device_key) pair in the reference sheet,
    sorted by vehicle number -- backs the webapp's single vehicle-search
    box (search across everything, device auto-populates on pick) instead
    of making the user pick a device before they can even search)."""
    return sorted(_load_map().items())


def counts_by_device():
    """device_key -> number of vehicles mapped to it (for the dropdown UI)."""
    counts = {k: 0 for k in DEVICE_KEYS}
    for d in _load_map().values():
        counts[d] = counts.get(d, 0) + 1
    return counts

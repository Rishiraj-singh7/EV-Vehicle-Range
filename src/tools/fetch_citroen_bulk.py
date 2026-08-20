"""One-off bulk pull of Citroen raw exports, to seed data/raw/citroen/ with
the historical training sample (the live webapp path uses
src/common/citroen_api_client.py instead).

Usage:  python src/tools/fetch_citroen_bulk.py [n_vehicles] [start] [end]

The endpoint rejects any window longer than 31 days ("Date range must not
exceed 31 days"), so a longer requested span is split into <=30-day chunks
and stitched back together (header from the first chunk only).

Files land as citroenraw-<VEH>-<start>-to-<end>.csv, mirroring the
"tataraw-" naming already used for Tata's historical bulk exports.
"""

import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv

from src.common.config import get_api_token
from src.common.paths import CITROEN_RAW_DIR, VEHICLE_DEVICE_MAP_PATH

API_URL = "http://15.206.222.173:3000/api/training/temp/export-citroen-raw"
PAUSE_SECONDS = 2  # be polite to the export API between requests
MAX_WINDOW_DAYS = 30  # endpoint's hard cap is 31; stay one under it


def citroen_vehicles():
    with open(VEHICLE_DEVICE_MAP_PATH, newline="", encoding="utf-8-sig") as f:
        return [
            r["vehicleNumber"].strip().upper()
            for r in csv.DictReader(f)
            if r["telematicDeviceType"].strip() == "Citroen"
        ]


def _chunks(start, end):
    """Split [start, end] into consecutive windows of at most MAX_WINDOW_DAYS."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    while s < e:
        stop = min(s + timedelta(days=MAX_WINDOW_DAYS), e)
        yield s.isoformat(), stop.isoformat()
        s = stop


def _fetch_window(vehicle, start, end, token):
    params = {"vehicleNumber": vehicle, "startDate": start, "endDate": end}
    req = urllib.request.Request(
        f"{API_URL}?{urllib.parse.urlencode(params)}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch(vehicle, start, end, token):
    """Full span as one CSV, fetched in <=30-day chunks and concatenated with
    a single header row."""
    parts = []
    for i, (cs, ce) in enumerate(_chunks(start, end)):
        data = _fetch_window(vehicle, cs, ce, token)
        if data.count(b"\n") < 2:
            continue
        if i and parts:
            data = data.split(b"\n", 1)[1]  # drop the repeated header
        parts.append(data if data.endswith(b"\n") else data + b"\n")
        time.sleep(PAUSE_SECONDS)
    return b"".join(parts)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    start = sys.argv[2] if len(sys.argv) > 2 else "2026-06-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-08-01"

    token = get_api_token()
    CITROEN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    vehicles = citroen_vehicles()[:n]
    print(f"Fetching {len(vehicles)} Citroen vehicles, {start} -> {end}")

    for i, veh in enumerate(vehicles, 1):
        out = CITROEN_RAW_DIR / f"citroenraw-{veh}-{start}-to-{end}.csv"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[{i}/{len(vehicles)}] {veh}: already have it, skipping")
            continue
        try:
            data = fetch(veh, start, end, token)
        except Exception as e:  # noqa: BLE001 - report and move to the next vehicle
            print(f"[{i}/{len(vehicles)}] {veh}: FAILED {type(e).__name__}: {e}")
            continue
        lines = data.count(b"\n")
        if lines < 2:
            print(f"[{i}/{len(vehicles)}] {veh}: empty, not saving")
            continue
        out.write_bytes(data)
        print(f"[{i}/{len(vehicles)}] {veh}: {lines} rows -> {out.name}")
        time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()

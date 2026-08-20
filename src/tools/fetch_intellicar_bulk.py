"""Pull Intellicar CAN exports for a named list of vehicles.

Used to fetch the held-out test vehicles (the ones in
data/reference/test_actual_range.csv that weren't already in
data/raw/intellicar_can/), so they can be scored as genuinely unseen.

Usage:
    python src/tools/fetch_intellicar_bulk.py VEH1 VEH2 ...
    python src/tools/fetch_intellicar_bulk.py --missing-from-test
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.config import get_api_token
from src.common.intellicar_api_client import fetch_can_csv
from src.common.paths import PROCESSED_DIR, REFERENCE_DIR, UNIFIED_EPOCHS_PATH

PAUSE_SECONDS = 2


def missing_from_test():
    test = pd.read_csv(REFERENCE_DIR / "test_actual_range.csv")
    have = set(pd.read_csv(UNIFIED_EPOCHS_PATH)["vehicle"])
    return [v for v in test["vehicle"] if v not in have]


def main():
    args = sys.argv[1:]
    vehicles = missing_from_test() if args == ["--missing-from-test"] else args
    if not vehicles:
        raise SystemExit("Nothing to fetch.")

    token = get_api_token()
    print(f"Fetching {len(vehicles)} vehicles (trailing 30 days each)")
    for i, veh in enumerate(vehicles, 1):
        try:
            path, source = fetch_can_csv(veh, token)
        except Exception as e:  # noqa: BLE001 - keep going through the list
            print(f"[{i}/{len(vehicles)}] {veh}: FAILED {type(e).__name__}: {e}")
            continue
        size_mb = path.stat().st_size / 1e6
        print(f"[{i}/{len(vehicles)}] {veh}: {source}, {size_mb:.1f} MB -> {path.name}")
        if source == "live":
            time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()

"""Live data client for the Tata export API -- two feeds off the same
endpoint, selected by `type`:

    type=gps       GPS+CAN ping stream (same shape as the historical
                   data/raw/tata_gps/*.csv files) -- range-model input.
    type=charging  A dedicated, much cleaner charging-only feed (SOC,
                   charging state, charge type) -- used for session
                   detection instead of trying to infer charging periods
                   out of the noisier gps stream.

    GET http://15.206.222.173:3000/api/training/temp/export-tata-raw
        ?vehicleNo=<VEH>&type=gps|charging&startDate=<YYYY-MM-DD>&endDate=<YYYY-MM-DD>
    Header: Authorization: Bearer <token>   (same token as Intellicar's --
    confirmed working against both endpoints, see src/common/config.py)

Unlike Intellicar's endpoint, dates here are plain YYYY-MM-DD with no time
component (matches the sample curl).

Window/cache: same fixed 30-day trailing window and cache-freshness rule as
Intellicar -- see intellicar_api_client.py's docstring for the reasoning.
Cache paths: data/raw/tata_gps/<VEH>-gps-<start>_<end>.csv and
data/raw/tata_charging/<VEH>-charging-<start>_<end>.csv.
"""

from datetime import datetime, timedelta, timezone

from .export_api_client import ExportApiError, fetch_csv
from .paths import TATA_CHARGING_RAW_DIR, TATA_RAW_DIR

API_URL = "http://15.206.222.173:3000/api/training/temp/export-tata-raw"
LOOKBACK_DAYS = 30
CACHE_TTL_HOURS = 6
REQUEST_TIMEOUT_SECONDS = 120

TataApiError = ExportApiError


def _date(dt):
    return dt.strftime("%Y-%m-%d")


def _window(lookback_days):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    return start, end


def _fetch(kind, raw_dir, vehicle, token, lookback_days, force_refresh):
    vehicle = vehicle.strip().upper()
    start, end = _window(lookback_days)
    # "live-" prefix keeps these visually and glob-distinct from the
    # historical "tataraw-<VEH>-gps-<start>-to-<end>.csv" bulk exports
    # already in data/raw/tata_gps/ -- same folder, two provenances.
    path = raw_dir / f"live-{vehicle}-{kind}-{start.date()}_{end.date()}.csv"
    params = {
        "vehicleNo": vehicle,
        "type": kind,
        "startDate": _date(start),
        "endDate": _date(end),
    }
    return fetch_csv(
        API_URL, params, token, path, CACHE_TTL_HOURS, REQUEST_TIMEOUT_SECONDS,
        force_refresh, source_label="Tata",
    )


def fetch_gps_csv(vehicle, token, lookback_days=LOOKBACK_DAYS, force_refresh=False):
    """(Path, source) for this vehicle's GPS+CAN feed -- range-model input."""
    return _fetch("gps", TATA_RAW_DIR, vehicle, token, lookback_days, force_refresh)


def fetch_charging_csv(vehicle, token, lookback_days=LOOKBACK_DAYS, force_refresh=False):
    """(Path, source) for this vehicle's dedicated charging feed -- session detection."""
    return _fetch("charging", TATA_CHARGING_RAW_DIR, vehicle, token, lookback_days, force_refresh)

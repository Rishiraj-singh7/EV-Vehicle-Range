"""Live data client for the Intellicar export API, with a local on-disk
cache so repeated 'Get Range' clicks for the same vehicle don't re-hit the
API every time.

Endpoint:
    GET http://15.206.222.173:3000/api/intellicar/export
        ?vehicleNumber=<VEH>&deviceType=Intellicar&type=can
        &startDate=<ISO>&endDate=<ISO>
    Header: Authorization: Bearer <token>

Window: always the trailing LOOKBACK_DAYS days from "now" (fixed 30-day
window, per product decision -- a vehicle that charged less than 16 times
in that window will simply show fewer sessions, with a note in the UI that
the count is a 30-day window, not lifetime).

Cache: data/raw/intellicar_can/<VEHICLE>-can-<start-date>_<end-date>.csv.
A cached file younger than CACHE_TTL_HOURS is reused as-is; otherwise it's
refetched and overwritten.
"""

from datetime import datetime, timedelta, timezone

from .export_api_client import ExportApiError, fetch_csv
from .paths import INTELLICAR_RAW_DIR

API_URL = "http://15.206.222.173:3000/api/intellicar/export"
LOOKBACK_DAYS = 30
CACHE_TTL_HOURS = 6
REQUEST_TIMEOUT_SECONDS = 120

IntellicarApiError = ExportApiError


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _cache_path(vehicle, start, end):
    return INTELLICAR_RAW_DIR / f"{vehicle}-can-{start.date()}_{end.date()}.csv"


def fetch_can_csv(vehicle, token, lookback_days=LOOKBACK_DAYS, force_refresh=False):
    """Returns (Path, source) for this vehicle's CAN telemetry CSV covering
    the trailing `lookback_days` days -- source is "cache" or "live"."""
    vehicle = vehicle.strip().upper()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    path = _cache_path(vehicle, start, end)

    params = {
        "vehicleNumber": vehicle,
        "deviceType": "Intellicar",
        "type": "can",
        "startDate": _iso(start),
        "endDate": _iso(end),
    }
    return fetch_csv(
        API_URL, params, token, path, CACHE_TTL_HOURS, REQUEST_TIMEOUT_SECONDS,
        force_refresh, source_label="Intellicar",
    )

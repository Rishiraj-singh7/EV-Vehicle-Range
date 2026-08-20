"""Live data client for the Citroen export API -- one feed, unlike Tata's
two: the single stream carries both the driving signals (speed, odometer,
SOC) and the charging signals (batteryInfo.chargingStatus /
batteryInfo.typeOfCharge), so range and charging sessions both come out of
the same pull.

    GET http://15.206.222.173:3000/api/training/temp/export-citroen-raw
        ?vehicleNumber=<VEH>&startDate=<YYYY-MM-DD>&endDate=<YYYY-MM-DD>
    Header: Authorization: Bearer <token>   (same shared token as the
    Intellicar/Tata endpoints -- same host, see src/common/config.py)

Dates are plain YYYY-MM-DD, like Tata's endpoint (not Intellicar's full ISO
timestamps). Note the param is `vehicleNumber` here, matching Intellicar --
Tata's is `vehicleNo`.

**31-day cap**: this endpoint rejects any window longer than 31 days with
HTTP 400 ("Date range must not exceed 31 days"). The standard 30-day
trailing window used everywhere else fits under that, so a normal fetch is
a single request -- but MAX_WINDOW_DAYS is enforced here anyway so a caller
asking for more gets a clear error instead of a bare 400.

Cache: data/raw/citroen/live-<VEH>-<start>_<end>.csv, same "live-" prefix
convention as Tata, keeping webapp pulls glob-distinct from the historical
"citroenraw-" bulk exports fetched by src/tools/fetch_citroen_bulk.py.
"""

from datetime import datetime, timedelta, timezone

from .export_api_client import ExportApiError, fetch_csv
from .paths import CITROEN_RAW_DIR

API_URL = "http://15.206.222.173:3000/api/training/temp/export-citroen-raw"
LOOKBACK_DAYS = 30
CACHE_TTL_HOURS = 6
REQUEST_TIMEOUT_SECONDS = 180
MAX_WINDOW_DAYS = 31  # server-side hard cap

CitroenApiError = ExportApiError


def _date(dt):
    return dt.strftime("%Y-%m-%d")


def fetch_raw_csv(vehicle, token, lookback_days=LOOKBACK_DAYS, force_refresh=False):
    """Returns (Path, source) for this vehicle's Citroen telemetry covering
    the trailing `lookback_days` days -- source is "cache" or "live"."""
    if lookback_days > MAX_WINDOW_DAYS:
        raise CitroenApiError(
            f"Citroen export accepts at most {MAX_WINDOW_DAYS} days per request "
            f"(asked for {lookback_days}); fetch in chunks -- see "
            f"src/tools/fetch_citroen_bulk.py for the chunking helper."
        )

    vehicle = vehicle.strip().upper()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    path = CITROEN_RAW_DIR / f"live-{vehicle}-{start.date()}_{end.date()}.csv"

    params = {
        "vehicleNumber": vehicle,
        "startDate": _date(start),
        "endDate": _date(end),
    }
    return fetch_csv(
        API_URL, params, token, path, CACHE_TTL_HOURS, REQUEST_TIMEOUT_SECONDS,
        force_refresh, source_label="Citroen",
    )

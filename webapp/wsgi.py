"""Production WSGI entry point -- the deployable form of the dashboard.

    gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 webapp.wsgi:app

webapp/server.py is a bare ThreadingHTTPServer: fine on a laptop, but it is
a development server and gunicorn cannot serve it (gunicorn needs a WSGI
callable, which an HTTPServer is not). This module exposes the same routes
as a real WSGI app, so the two stay interchangeable -- server.py for local
work, this for anything hosted.

All the logic still lives in range_service.py, which never knew about HTTP
in the first place; this file is routing, auth and error mapping only.

Configuration (environment variables):
    MILODRIVE_BEARER_TOKEN   upstream export-API token (see src/common/config.py)
    DASHBOARD_USER           HTTP basic-auth username  -- REQUIRED unless AUTH_DISABLED
    DASHBOARD_PASSWORD       HTTP basic-auth password  -- REQUIRED unless AUTH_DISABLED
    DASHBOARD_AUTH_DISABLED  set to "1" to run without auth (local only)

**Auth is on by default and the app refuses to start without credentials.**
That is deliberate: this service holds a token for an upstream API that
bills against your quota, so an unauthenticated public deployment would let
anyone spend it. Opting out has to be an explicit, visible act
(DASHBOARD_AUTH_DISABLED=1), not the consequence of forgetting to set a
variable.
"""

import hmac
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from functools import wraps
from pathlib import Path

# Repo root for `src.*`, and this directory for `range_service` / `server`,
# so the module imports identically whether gunicorn is pointed at
# `webapp.wsgi:app` from the repo root or `wsgi:app` from inside webapp/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, Response, jsonify, request, send_from_directory

import range_service
from server import INTELLICAR_UPSTREAM_URL, month_range_iso

STATIC_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=None)


# --- auth ---------------------------------------------------------------

AUTH_DISABLED = os.environ.get("DASHBOARD_AUTH_DISABLED", "").strip() == "1"
AUTH_USER = os.environ.get("DASHBOARD_USER", "").strip()
AUTH_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")

if not AUTH_DISABLED and not (AUTH_USER and AUTH_PASSWORD):
    raise RuntimeError(
        "Refusing to start without credentials. Set DASHBOARD_USER and "
        "DASHBOARD_PASSWORD, or set DASHBOARD_AUTH_DISABLED=1 to run "
        "unauthenticated (local development only -- this service holds an "
        "upstream API token)."
    )


def _credentials_ok(auth):
    """compare_digest on both fields so a wrong username and a wrong
    password take the same time to reject -- a plain == leaks which half was
    right via timing."""
    if auth is None:
        return False
    user_ok = hmac.compare_digest(auth.username or "", AUTH_USER)
    pass_ok = hmac.compare_digest(auth.password or "", AUTH_PASSWORD)
    return user_ok and pass_ok


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if AUTH_DISABLED or _credentials_ok(request.authorization):
            return view(*args, **kwargs)
        return Response(
            "Authentication required.",
            401,
            {"WWW-Authenticate": 'Basic realm="EV Range Dashboard"'},
        )

    return wrapped


# --- pages --------------------------------------------------------------

@app.get("/")
@app.get("/index.html")
@require_auth
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/export-tool")
@app.get("/export-tool.html")
@require_auth
def export_tool():
    return send_from_directory(STATIC_DIR, "intellicar_export_tool.html")


@app.get("/healthz")
def healthz():
    """Unauthenticated so the platform's health check can reach it. Reports
    only liveness -- no fleet data, nothing that needs protecting."""
    return jsonify({"status": "ok"})


# --- api ----------------------------------------------------------------

@app.get("/api/devices")
@require_auth
def api_devices():
    return jsonify({"devices": range_service.list_devices()})


@app.get("/api/vehicles")
@require_auth
def api_vehicles():
    device = (request.args.get("device") or "").strip().lower()
    try:
        vehicles = range_service.list_vehicles(device)
    except range_service.RangeServiceError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"device": device, "vehicles": vehicles})


@app.get("/api/all-vehicles")
@require_auth
def api_all_vehicles():
    return jsonify({"vehicles": range_service.list_all_vehicles()})


@app.get("/api/range")
@require_auth
def api_range():
    # `device` is optional -- range_service resolves it from the reference
    # sheet, so an operator only supplies a vehicle number.
    device = (request.args.get("device") or "").strip().lower() or None
    vehicle = (request.args.get("vehicle") or "").strip()
    if not vehicle:
        return jsonify({"error": "vehicle is required"}), 400
    try:
        result = range_service.get_range_and_sessions(device, vehicle)
    except range_service.RangeServiceError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 - always answer as JSON, never HTML
        app.logger.exception("range lookup failed for %s", vehicle)
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    return jsonify(result)


# --- legacy manual CSV export ------------------------------------------

@app.get("/download")
@require_auth
def download():
    """Proxies a vehicle-month CSV straight from the Intellicar endpoint.

    Unlike every other route, the upstream token comes from the caller's own
    Authorization header rather than the server's configured one -- this
    predates the live-fetch path and is kept as-is so the export tool page
    keeps working."""
    vehicle = (request.args.get("vehicleNumber") or "").strip().upper()
    year_raw = request.args.get("year") or ""
    month_raw = request.args.get("month") or ""
    token = (request.headers.get("Authorization") or "").strip()

    if not vehicle or not year_raw or not month_raw:
        return "vehicleNumber, year and month are required", 400
    if not token:
        return "Missing Authorization token", 401
    if not token.lower().startswith("bearer "):
        token = "Bearer " + token
    try:
        year, month = int(year_raw), int(month_raw)
        if not 1 <= month <= 12:
            raise ValueError
    except ValueError:
        return "Invalid year/month", 400

    start_iso, end_iso = month_range_iso(year, month)
    params = {
        "vehicleNumber": vehicle,
        "deviceType": "Intellicar",
        "type": "can",
        "startDate": start_iso,
        "endDate": end_iso,
    }
    req = urllib.request.Request(
        f"{INTELLICAR_UPSTREAM_URL}?{urllib.parse.urlencode(params)}",
        headers={"Authorization": token},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"Upstream error {e.code}: {body}", e.code
    except urllib.error.URLError as e:
        return f"Could not reach upstream API: {e.reason}", 502

    filename = f"intellicar-{vehicle}-{start_iso[:10]}_{end_iso[:10]}.csv"
    return Response(
        data,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    # Local convenience only. In production run this module under gunicorn;
    # Flask's built-in server is no more production-ready than server.py's.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8787)), debug=False)

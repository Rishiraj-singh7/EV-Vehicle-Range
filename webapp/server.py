"""Backend for the EV Range Dashboard.

Serves the dashboard page and a small JSON API on top of range_service.py:
    GET  /api/devices                          -> device dropdown options
    GET  /api/vehicles?device=<key>             -> vehicle dropdown options
    GET  /api/range?device=<key>&vehicle=<veh>   -> range + charging sessions

Also keeps the original Intellicar CSV export tool alive at /export-tool
(manual vehicle+month CSV download -- separate from the live-fetch path the
dashboard itself uses).

Run:
    python webapp/server.py
Then open http://127.0.0.1:8787 in a browser.
"""

import calendar
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import range_service
from src.common.intellicar_api_client import API_URL as INTELLICAR_UPSTREAM_URL

STATIC_DIR = Path(__file__).parent
PORT = 8787


def month_range_iso(year, month):
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    return start.strftime(fmt), end.strftime(fmt)


class Handler(BaseHTTPRequestHandler):
    # --- small helpers ----------------------------------------------------
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel_path, content_type):
        path = STATIC_DIR / rel_path
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # --- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif parsed.path in ("/export-tool", "/export-tool.html"):
            self._serve_static("intellicar_export_tool.html", "text/html; charset=utf-8")
        elif parsed.path == "/api/devices":
            self._handle_devices()
        elif parsed.path == "/api/vehicles":
            self._handle_vehicles(qs)
        elif parsed.path == "/api/all-vehicles":
            self._handle_all_vehicles()
        elif parsed.path == "/api/range":
            self._handle_range(qs)
        elif parsed.path == "/download":
            self._handle_legacy_download(parsed)
        else:
            self.send_error(404)

    # --- new dashboard API ---------------------------------------------------
    def _handle_devices(self):
        self._send_json(200, {"devices": range_service.list_devices()})

    def _handle_vehicles(self, qs):
        device = qs.get("device", [""])[0].strip().lower()
        try:
            vehicles = range_service.list_vehicles(device)
        except range_service.RangeServiceError as e:
            self._send_json(400, {"error": str(e)})
            return
        self._send_json(200, {"device": device, "vehicles": vehicles})

    def _handle_all_vehicles(self):
        self._send_json(200, {"vehicles": range_service.list_all_vehicles()})

    def _handle_range(self, qs):
        # `device` is optional: the service resolves it from the reference
        # sheet, so an operator only has to supply a vehicle number. When it
        # IS supplied the service cross-checks it and errors on a mismatch.
        device = qs.get("device", [""])[0].strip().lower() or None
        vehicle = qs.get("vehicle", [""])[0].strip()
        if not vehicle:
            self._send_json(400, {"error": "vehicle is required"})
            return
        try:
            result = range_service.get_range_and_sessions(device, vehicle)
        except range_service.RangeServiceError as e:
            self._send_json(400, {"error": str(e)})
            return
        except Exception as e:  # unexpected failure -- still respond as JSON
            self._send_json(500, {"error": f"Unexpected error: {e}"})
            return
        self._send_json(200, result)

    # --- legacy manual CSV export tool ---------------------------------------
    def _handle_legacy_download(self, parsed):
        qs = urllib.parse.parse_qs(parsed.query)
        vehicle = qs.get("vehicleNumber", [""])[0].strip().upper()
        year_raw = qs.get("year", [""])[0]
        month_raw = qs.get("month", [""])[0]
        token = self.headers.get("Authorization", "").strip()

        if not vehicle or not year_raw or not month_raw:
            self.send_error(400, "vehicleNumber, year and month are required")
            return
        if not token:
            self.send_error(401, "Missing Authorization token")
            return
        if not token.lower().startswith("bearer "):
            token = "Bearer " + token

        try:
            year, month = int(year_raw), int(month_raw)
            if not 1 <= month <= 12:
                raise ValueError
        except ValueError:
            self.send_error(400, "Invalid year/month")
            return

        start_iso, end_iso = month_range_iso(year, month)
        params = {
            "vehicleNumber": vehicle,
            "deviceType": "Intellicar",
            "type": "can",
            "startDate": start_iso,
            "endDate": end_iso,
        }
        url = f"{INTELLICAR_UPSTREAM_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Authorization": token})

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            self.send_response(e.code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Upstream error {e.code}: {body}".encode("utf-8"))
            return
        except urllib.error.URLError as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Could not reach upstream API: {e.reason}".encode("utf-8"))
            return

        filename = f"intellicar-{vehicle}-{start_iso[:10]}_{end_iso[:10]}.csv"
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

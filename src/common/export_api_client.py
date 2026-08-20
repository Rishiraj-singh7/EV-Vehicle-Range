"""Shared HTTP plumbing for the Milodrive export APIs (Intellicar, Tata --
same host, same bearer auth, each with its own URL/params/date format).
Device-specific modules (intellicar_api_client.py, tata_api_client.py) each
supply their own URL and params; this handles the fetch-or-reuse-cache
decision and the actual request once that's decided.
"""

import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone


class ExportApiError(RuntimeError):
    pass


def _bearer(token):
    token = token.strip()
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"


def _is_fresh(path, ttl_hours):
    if not path.exists():
        return False
    age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_hours < ttl_hours


def fetch_csv(url, params, token, cache_path, ttl_hours, timeout_seconds, force_refresh=False, source_label="export"):
    """Returns (Path, source) where source is "cache" (reused a fresh local
    file) or "live" (just pulled from the API and cached it)."""
    if not force_refresh and _is_fresh(cache_path, ttl_hours):
        return cache_path, "cache"

    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full_url, headers={"Authorization": _bearer(token)})

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ExportApiError(f"{source_label} API error {e.code}: {body[:300]}") from e
    except urllib.error.URLError as e:
        raise ExportApiError(f"Could not reach {source_label} API: {e.reason}") from e

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return cache_path, "live"

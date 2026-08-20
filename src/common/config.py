"""Local secrets loader.

Reads config/secrets.local.json if present (kept out of the main source
tree deliberately -- see the file's own _comment). Falls back to an
environment variable so the token can be overridden per-machine without
editing the file (e.g. MILODRIVE_BEARER_TOKEN=... python webapp/server.py).

One token, one host: Intellicar's export API and Tata's export API both
live on 15.206.222.173:3000 under the same auth -- confirmed by using the
same bearer token against both endpoints -- so a single "api_bearer_token"
covers every device's live fetch, not one per device.
"""

import json
import os

from .paths import SECRETS_PATH


def _load_secrets():
    if SECRETS_PATH.exists():
        return json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    return {}


def get_api_token():
    secrets = _load_secrets()
    token = secrets.get("api_bearer_token", "").strip()
    if not token:
        token = secrets.get("intellicar_bearer_token", "").strip()  # old key, back-compat
    if token:
        return token
    token = os.environ.get("MILODRIVE_BEARER_TOKEN", "").strip()
    if token:
        return token
    return os.environ.get("INTELLICAR_BEARER_TOKEN", "").strip()  # old name, back-compat

"""The single central lookup behind the webapp: given a vehicle number,
return its predicted full-charge range and its recent charging sessions --
whatever device it happens to be on.

One path, three devices. The operator types a vehicle number and nothing
else; the device is resolved from the reference sheet, and from there the
flow is identical:

    1. resolve device            data/reference/vehicle_device_map.csv
    2. fetch telemetry           that device's export API (30-day window,
                                 6h on-disk cache, so repeat lookups of the
                                 same vehicle don't re-hit the API)
    3. normalize                 src/common/device_adapters.py
    4. build epochs              src/common/epoch_pipeline.py -- the SAME
                                 rules the trainer used
    5. merge into central store  src/common/central_store.py, so the pull is
                                 kept and the table grows with use
    6. predict                   models/unified_range_model.joblib, one model
                                 for every device
    7. charging sessions         from the same pull (Tata uses its dedicated
                                 charging feed when available)

This replaces the previous per-device functions, which each loaded their own
model and ran their own epoch builder. Citroen, which used to return "not
connected yet", is now a device like any other.

**Known and new vehicles both work**, but not equally well. The model takes
`vehicle` as a categorical, so a vehicle it trained on gets its own learned
offset while an unseen one falls back on `device` plus driving conditions
alone -- which, on a held-out test, compressed predictions toward the fleet
mean and under-read long-range vehicles badly. The response therefore
carries `prediction_basis` and `confidence` so the UI can say which kind of
answer it is giving instead of presenting both as equally solid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.citroen.charging import charging_sessions_for as citroen_charging_sessions
from src.common import central_store, device_map
from src.common.charging_sessions import summarize_last_n
from src.common.citroen_api_client import CitroenApiError, fetch_raw_csv as citroen_fetch_csv
from src.common.config import get_api_token
from src.common.device_adapters import LOADERS
from src.common.epoch_pipeline import epochs_from_rows
from src.common.export_api_client import ExportApiError
from src.common.intellicar_api_client import fetch_can_csv
from src.common.paths import UNIFIED_MODEL_PATH
from src.common.tata_api_client import fetch_charging_csv as tata_fetch_charging_csv
from src.common.tata_api_client import fetch_gps_csv as tata_fetch_gps_csv
from src.intellicar.charging import charging_sessions_for as intellicar_charging_sessions
from src.tata.charging import charging_sessions_for as tata_charging_sessions

SESSIONS_REQUESTED = 16
SESSIONS_NOTE = (
    "Sessions are counted from the last 30 days of data only "
    "(a session counts only when SOC gained is more than 10%)."
)

# Minimum epochs before a prediction is treated as well-supported. Below
# this the median driving conditions it is built from are themselves noisy.
MIN_EPOCHS_FOR_CONFIDENCE = 10

DEVICE_LABELS = {
    "intellicar": "Intellicar",
    "tata": "Tata",
    "citroen": "Citroen",
    "no_device": "No device",
}

# device_map's lowercase keys -> the capitalised names used as the model's
# `device` feature value and as device_adapters.LOADERS keys.
DEVICE_NAMES = {"intellicar": "Intellicar", "tata": "Tata", "citroen": "Citroen"}

_model_cache = {}


class RangeServiceError(Exception):
    """User-facing error (bad vehicle, no data, no telemetry, etc)."""


def _load_model():
    if UNIFIED_MODEL_PATH not in _model_cache:
        if not UNIFIED_MODEL_PATH.exists():
            raise RangeServiceError(
                f"Model file missing: {UNIFIED_MODEL_PATH.name}. "
                "Run src/unified/train_range_model.py first."
            )
        _model_cache[UNIFIED_MODEL_PATH] = joblib.load(UNIFIED_MODEL_PATH)
    return _model_cache[UNIFIED_MODEL_PATH]


def list_devices():
    counts = device_map.counts_by_device()
    return [
        {"key": k, "label": DEVICE_LABELS[k], "vehicle_count": counts.get(k, 0)}
        for k in device_map.DEVICE_KEYS
    ]


def list_vehicles(device_key):
    if device_key not in device_map.DEVICE_KEYS:
        raise RangeServiceError(f"Unknown device type: {device_key}")
    return device_map.vehicles_for_device(device_key)


def list_all_vehicles():
    """Every vehicle in the reference sheet with its device, for the
    webapp's search-as-you-type vehicle box."""
    return [
        {"vehicle": v, "device": d, "device_label": DEVICE_LABELS.get(d, d)}
        for v, d in device_map.all_vehicles_with_device()
    ]


# --- step 2: fetch, per device ------------------------------------------

def _fetch_telemetry(device_key, vehicle, token):
    """(csv_path, source) for the vehicle's driving telemetry.

    Each device's client handles its own URL, parameter names and date
    format; all three share the 30-day window and 6-hour cache."""
    if device_key == "intellicar":
        return fetch_can_csv(vehicle, token)
    if device_key == "tata":
        return tata_fetch_gps_csv(vehicle, token)
    if device_key == "citroen":
        return citroen_fetch_csv(vehicle, token)
    raise RangeServiceError(f"No export API wired for device '{device_key}'.")


# --- step 7: charging sessions, per device ------------------------------

def _charging_sessions(device_key, vehicle, token, telemetry_path):
    """(sessions, human-readable description of how fast/slow was decided).

    Intellicar and Citroen both report their own charge-type flag, so the
    sessions are classified from the device rather than inferred. Tata has
    no trustworthy flag and needs its dedicated charging feed plus the
    rate-based cutoff."""
    if device_key == "intellicar":
        return (
            intellicar_charging_sessions(telemetry_path, vehicle),
            "device flag (fast_charge_indicator)",
        )

    if device_key == "citroen":
        return (
            citroen_charging_sessions(telemetry_path, vehicle),
            "device flag (typeOfCharge: Quick/Slow)",
        )

    if device_key == "tata":
        # Dedicated charging feed first -- much cleaner than picking charging
        # periods out of the gps ping stream. Fall back to the gps pull we
        # already have if that call fails.
        if token:
            try:
                charging_path, _ = tata_fetch_charging_csv(vehicle, token)
                return (
                    tata_charging_sessions(charging_path, vehicle),
                    "dedicated charging feed (rate-based fast/slow)",
                )
            except ExportApiError:
                pass
        return (
            tata_charging_sessions(telemetry_path, vehicle),
            "inferred from driving feed (rate-based fast/slow)",
        )

    return [], "unavailable"


# --- step 6: predict ----------------------------------------------------

def _typical_features(epochs_df, numeric_features):
    """Median of each feature across this vehicle's stored epochs -- the same
    'typical driving conditions' query used at training time."""
    row = {}
    for f in numeric_features:
        val = epochs_df[f].median()
        row[f] = 0.0 if pd.isna(val) else float(val)
    return row


def _predict(bundle, device_name, vehicle, epochs_df):
    feature_row = _typical_features(epochs_df, bundle["numeric_features"])
    query = pd.DataFrame([{**feature_row, "device": device_name, "vehicle": vehicle}])
    cols = bundle["numeric_features"] + bundle["categorical_features"]
    return round(float(bundle["pipeline"].predict(query[cols])[0]), 1)


def _assess_confidence(bundle, vehicle, n_epochs):
    """How much to trust this particular prediction.

    The distinction that matters is whether the model has seen this vehicle
    before: `vehicle` is a categorical feature, so a known vehicle carries a
    learned per-vehicle offset, while an unknown one is scored from device
    and driving conditions alone. On the held-out test set that second case
    compressed predictions toward the fleet mean -- it under-read genuinely
    long-range vehicles by ~27 km and never once put one in the top band."""
    known = vehicle in set(bundle.get("vehicles", []))
    if not known:
        return (
            "device_and_conditions",
            "low",
            "This vehicle is not in the trained set, so the estimate comes from its "
            "device type and driving pattern alone. Expect it to read low for a "
            "genuinely long-range vehicle; retrain to include it.",
        )
    if n_epochs < MIN_EPOCHS_FOR_CONFIDENCE:
        return (
            "vehicle_specific",
            "medium",
            f"Only {n_epochs} usable driving epochs for this vehicle, so its typical "
            "conditions are noisy. More data will firm this up.",
        )
    return (
        "vehicle_specific",
        "high",
        f"Vehicle is in the trained set with {n_epochs} usable driving epochs.",
    )


# --- the one entry point ------------------------------------------------

def get_range_and_sessions(device_key, vehicle):
    """Predicted full-charge range + recent charging sessions for one
    vehicle. `device_key` may be None -- it is resolved from the reference
    sheet either way, and only cross-checked when supplied."""
    vehicle = vehicle.strip().upper()

    mapped = device_map.device_of(vehicle)
    if mapped is None:
        raise RangeServiceError(f"{vehicle} is not in the vehicle/device reference sheet.")
    if device_key and mapped != device_key:
        raise RangeServiceError(f"{vehicle} is mapped to device '{mapped}', not '{device_key}'.")
    device_key = mapped

    if device_key == "no_device":
        raise RangeServiceError(f"{vehicle} has no telematics device installed (NO_DEVICE).")
    device_name = DEVICE_NAMES.get(device_key)
    if device_name is None:
        raise RangeServiceError(f"Unsupported device '{device_key}' for {vehicle}.")

    token = get_api_token()
    if not token:
        raise RangeServiceError("No API token configured (config/secrets.local.json).")

    bundle = _load_model()

    # 2-3. fetch + normalize
    try:
        csv_path, source = _fetch_telemetry(device_key, vehicle, token)
    except (ExportApiError, CitroenApiError) as e:
        raise RangeServiceError(str(e)) from e

    clean_rows = LOADERS[device_name](csv_path)

    # 4-5. build epochs and fold them into the central store, then predict
    # from everything the store holds for this vehicle -- this pull plus
    # every earlier one.
    new_epochs = epochs_from_rows(clean_rows, vehicle, device_name, csv_path.name)
    vehicle_epochs, n_added = central_store.merge_epochs(new_epochs)

    if vehicle_epochs.empty:
        vehicle_epochs = central_store.epochs_for(vehicle)

    range_km = None
    basis, confidence, confidence_note = _assess_confidence(bundle, vehicle, len(vehicle_epochs))
    if not vehicle_epochs.empty:
        range_km = _predict(bundle, device_name, vehicle, vehicle_epochs)
    else:
        confidence, confidence_note = "none", (
            f"No usable driving epochs for {vehicle} in the last 30 days -- the "
            "vehicle may be idle, or its telemetry may be too sparse to measure a "
            "discharge from."
        )

    # 7. charging sessions
    try:
        sessions, sessions_source = _charging_sessions(device_key, vehicle, token, csv_path)
    except Exception:  # noqa: BLE001 - a range answer is still worth returning
        sessions, sessions_source = [], "unavailable (session lookup failed)"
    session_summary = summarize_last_n(sessions, SESSIONS_REQUESTED)

    return _format_result(
        device_key=device_key,
        vehicle=vehicle,
        range_km=range_km,
        bundle=bundle,
        data_source=source,
        vehicle_epochs=len(vehicle_epochs),
        epochs_added=n_added,
        basis=basis,
        confidence=confidence,
        confidence_note=confidence_note,
        session_summary=session_summary,
        sessions_source=sessions_source,
    )


def _format_result(device_key, vehicle, range_km, bundle, data_source, vehicle_epochs,
                   epochs_added, basis, confidence, confidence_note,
                   session_summary, sessions_source):
    sessions_out = [
        {
            "start": s.start_time.isoformat(),
            "end": s.end_time.isoformat(),
            "soc_start": s.soc_start,
            "soc_end": s.soc_end,
            "soc_gained": round(s.soc_gained, 1),
            "duration_hours": round(s.duration_hours, 2),
            "rate_pct_per_hour": round(s.rate_pct_per_hour, 1),
            "is_fast": s.is_fast,
            "classified_by": s.classified_by,
        }
        for s in session_summary["sessions"]
    ]
    return {
        "device": device_key,
        "device_label": DEVICE_LABELS.get(device_key, device_key),
        "vehicle": vehicle,
        "range_km": range_km,
        "range_available": range_km is not None,
        "prediction_basis": basis,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "model_name": bundle.get("model_name"),
        "model_trained_on_epochs": bundle.get("n_epochs_trained_on"),
        "model_features": bundle.get("numeric_features"),
        "vehicle_epochs_in_store": vehicle_epochs,
        "epochs_added_this_lookup": epochs_added,
        "data_source": data_source,
        "charging_sessions": {
            "requested": session_summary["requested"],
            "sessions_found": session_summary["sessions_found"],
            "fast": session_summary["fast"],
            "slow": session_summary["slow"],
            "note": SESSIONS_NOTE,
            "classification_source": sessions_source,
            "sessions": sessions_out,
        },
    }

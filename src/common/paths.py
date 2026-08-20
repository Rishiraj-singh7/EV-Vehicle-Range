"""Central path registry for the whole project.

Every script imports its file/folder locations from here instead of each
computing its own `Path(__file__).parent` -- so when the layout changes,
it changes in one place.

Layout:
    data/reference/    vehicle -> device-type lookup
    data/raw/           per-device raw telemetry pulls (one CSV per vehicle/period)
    data/processed/     per-device epoch tables + predicted-range tables
    models/              trained model bundles (.joblib)
    config/              local secrets (API tokens) -- not meant to be committed
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- data ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

VEHICLE_DEVICE_MAP_PATH = REFERENCE_DIR / "vehicle_device_map.csv"

INTELLICAR_RAW_DIR = RAW_DIR / "intellicar_can"
TATA_RAW_DIR = RAW_DIR / "tata_gps"
TATA_CHARGING_RAW_DIR = RAW_DIR / "tata_charging"
CITROEN_RAW_DIR = RAW_DIR / "citroen"

INTELLICAR_EPOCHS_PATH = PROCESSED_DIR / "intellicar_epochs.csv"
INTELLICAR_VEHICLE_RANGE_PATH = PROCESSED_DIR / "intellicar_vehicle_range.csv"
INTELLICAR_NEW_VEHICLE_RANGE_PATH = PROCESSED_DIR / "intellicar_new_vehicle_range.csv"

TATA_EPOCHS_PATH = PROCESSED_DIR / "tata_epochs.csv"
TATA_VEHICLE_RANGE_PATH = PROCESSED_DIR / "tata_vehicle_range.csv"

# Centralised cross-device table: every fleet's epochs, one schema, built by
# src/unified/build_epochs.py and consumed by the single merged model.
UNIFIED_EPOCHS_PATH = PROCESSED_DIR / "unified_epochs.csv"
UNIFIED_FEATURE_RANKING_PATH = PROCESSED_DIR / "unified_feature_ranking.csv"
UNIFIED_VEHICLE_RANGE_PATH = PROCESSED_DIR / "unified_vehicle_range.csv"

# --- models ---------------------------------------------------------------
MODELS_DIR = ROOT / "models"
INTELLICAR_MODEL_PATH = MODELS_DIR / "intellicar_range_model.joblib"
TATA_MODEL_PATH = MODELS_DIR / "tata_range_model.joblib"
UNIFIED_MODEL_PATH = MODELS_DIR / "unified_range_model.joblib"

# --- config ---------------------------------------------------------------
CONFIG_DIR = ROOT / "config"
SECRETS_PATH = CONFIG_DIR / "secrets.local.json"

# --- webapp ---------------------------------------------------------------
WEBAPP_DIR = ROOT / "webapp"

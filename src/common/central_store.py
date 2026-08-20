"""The central epoch store: read, merge, persist.

data/processed/unified_epochs.csv is both the model's training table and the
live service's memory. Every lookup an operator does adds whatever new
epochs that pull revealed, so the store grows with use and the next
retraining automatically sees everything the fleet has looked at.

That gives the two behaviours the service needs, through one code path:

  * vehicle already known -> its stored epochs are reused, and the fresh
    pull only contributes epochs the store didn't already have
  * vehicle never seen    -> the fresh pull is all there is, and it lands in
    the store so it is "known" from then on

Concurrency: writes go to a temp file and are then moved into place, so a
reader never observes a half-written CSV. Two simultaneous writers can still
have one overwrite the other's addition -- acceptable here because the lost
epochs are recovered by the next lookup or the next batch rebuild, and the
alternative (a real lock or a database) is not worth it at this scale.
"""

import os
import tempfile

import pandas as pd

from .epoch_pipeline import add_rest_hours, add_target, apply_quality_filters, dedupe
from .paths import UNIFIED_EPOCHS_PATH

TIME_COLUMNS = ["start_time", "end_time"]


def load(path=UNIFIED_EPOCHS_PATH):
    """The whole central table, or an empty frame if it doesn't exist yet."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in TIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def epochs_for(vehicle, path=UNIFIED_EPOCHS_PATH):
    """Just this vehicle's stored epochs."""
    df = load(path)
    if df.empty:
        return df
    return df[df["vehicle"] == vehicle].copy()


def merge_epochs(new_epochs, path=UNIFIED_EPOCHS_PATH):
    """Fold freshly-built epochs into the central table and persist it.

    Returns (vehicle_epochs, n_added) where vehicle_epochs is everything the
    store now holds for the vehicles in `new_epochs` -- old and new
    together, which is what the prediction should be based on.

    The incoming epochs go through exactly the same target/dedupe/filter
    chain as the batch builder (src/common/epoch_pipeline.py), so a
    prediction is never computed on rows the trainer would have thrown out.
    """
    if not new_epochs:
        return pd.DataFrame(), 0

    incoming = add_target(pd.DataFrame(new_epochs))
    existing = load(path)

    combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
    before = len(combined)
    combined = dedupe(combined)
    combined = add_rest_hours(combined)
    combined = apply_quality_filters(combined)

    n_added = len(combined) - len(existing)
    if n_added > 0 or before != len(combined):
        _atomic_write(combined, path)

    vehicles = set(incoming["vehicle"])
    return combined[combined["vehicle"].isin(vehicles)].copy(), max(n_added, 0)


def _atomic_write(df, path):
    """Write via a temp file in the same directory, then replace -- so a
    concurrent reader never catches a partially-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = ["device", "vehicle", "file", "start_time", "end_time"]
    ordered = [c for c in lead if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            df.to_csv(f, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

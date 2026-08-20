"""Device-agnostic epoch features, computed from the four signals every
feed shares (time, soc, odometer, speed -- see device_adapters.py).

Why this module exists
----------------------
The per-device pipelines each invented their own feature set, which made a
merged training table impossible: Intellicar had battery temperature, Tata
had altitude and accelerometers, Citroen has neither. Worse, the two
pipelines' shared-looking features were not actually comparable, because
the feeds sample at wildly different rates:

    Intellicar   ~3-30 s between pings
    Tata          ~5 min
    Citroen       ~5 min

That is up to a 100x difference, and it silently breaks any feature built
by *counting* rows. `driving_minutes` in the Tata pipeline is literally
`n_rows * 5`, which would be off by ~100x if the same code ran over
Intellicar data. Row counts, stop counts, and sampled variance are all
cadence-dependent and therefore not portable.

The fix used throughout this module: **integrate over time, never count
rows.** Every row is given a dwell -- the time until the next ping, capped
at GAP_CAP_MINUTES so one overnight sleep gap can't swamp a segment -- and
every average is dwell-weighted. A dwell-weighted mean speed means the same
thing whether the underlying feed pings every 3 seconds or every 5 minutes,
which is exactly the property a merged model needs.

The cap also makes `active_hours` more honest than raw wall-clock duration:
a vehicle parked for 14 hours mid-epoch contributes GAP_CAP_MINUTES, not 14
hours, so the feature reflects time the vehicle was plausibly in use.
"""

from statistics import median

# A row's dwell is capped here so a single sleep/offline gap contributes a
# bounded amount instead of dominating the epoch. 15 min comfortably covers
# a 5-min nominal cadence plus jitter/missed pings.
GAP_CAP_MINUTES = 15.0

# Speeds at or below this count as stopped (idling, traffic, parked).
MOVING_SPEED_KMH = 1.0

# Duty-cycle bands. Chosen against the pooled speed distribution rather than
# road-type conventions: these fleets are urban commercial EVs whose median
# moving speed is ~30 km/h, so "highway" here means sustained open running
# for these vehicles, not motorway speeds.
CONGESTED_SPEED_KMH = 15.0
HIGHWAY_SPEED_KMH = 40.0


def _dwells(seg):
    """Per-row dwell in hours: time to the next ping, capped. The final row
    has no successor, so it inherits the segment's median dwell rather than
    0 -- at a 5-min cadence a dropped final row would lose a real chunk of a
    short epoch."""
    gaps = []
    for a, b in zip(seg, seg[1:]):
        minutes = (b["time"] - a["time"]).total_seconds() / 60.0
        gaps.append(min(max(minutes, 0.0), GAP_CAP_MINUTES))
    gaps.append(median(gaps) if gaps else 0.0)
    return [g / 60.0 for g in gaps]


def _wmean(values, weights):
    tw = sum(weights)
    if tw <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / tw


def _wstd(values, weights):
    m = _wmean(values, weights)
    if m is None or len(values) < 2:
        return None
    var = _wmean([(v - m) ** 2 for v in values], weights)
    return var**0.5 if var is not None else None


def _wquantile(values, weights, q):
    """Weight-aware quantile -- used instead of max() for the 'peak speed'
    feature, because a plain max over a 3-second feed picks up a single
    stray sample while the same max over a 5-minute feed almost never
    does."""
    pairs = sorted((v, w) for v, w in zip(values, weights) if w > 0)
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    target, run = q * total, 0.0
    for v, w in pairs:
        run += w
        if run >= target:
            return v
    return pairs[-1][0]


def _wiqr(values, weights):
    """Weighted inter-quartile range, or None if the quantiles aren't
    computable (every dwell zero, or fewer than two readings)."""
    if len(values) < 2:
        return None
    hi, lo = _wquantile(values, weights, 0.75), _wquantile(values, weights, 0.25)
    return None if hi is None or lo is None else hi - lo


def epoch_features(seg):
    """Unified features for one discharge segment (time-ordered rows from a
    device adapter). Returns None if the segment can't support them.

    Emits the target components (soc_used, distance_km) alongside the
    features; the trainer excludes those from X since they define the
    label."""
    if len(seg) < 2:
        return None

    soc_start, soc_end = seg[0]["soc"], seg[-1]["soc"]
    soc_used = soc_start - soc_end
    distance_km = seg[-1]["odo"] - seg[0]["odo"]
    if soc_used <= 0 or distance_km <= 0:
        return None

    dwell = _dwells(seg)
    active_hours = sum(dwell)
    if active_hours <= 0:
        return None

    moving = [(r, d) for r, d in zip(seg, dwell) if r["speed"] is not None and r["speed"] > MOVING_SPEED_KMH]
    moving_hours = sum(d for _, d in moving)
    speeds = [r["speed"] for r, _ in moving]
    sw = [d for _, d in moving]

    # Duty-cycle split. Stop-go crawling and steady highway running load the
    # battery very differently, and averaging them into one mean speed hides
    # that -- these two say *how* the driving time was spent, which no single
    # average can. Both are time-integrated, so they stay comparable across
    # the three feeds' very different ping rates.
    congested_hours = sum(d for r, d in moving if r["speed"] <= CONGESTED_SPEED_KMH)
    highway_hours = sum(d for r, d in moving if r["speed"] > HIGHWAY_SPEED_KMH)

    # SOC averaged over the whole segment (dwell-weighted), as a band proxy.
    socs = [r["soc"] for r in seg]

    wall_hours = (seg[-1]["time"] - seg[0]["time"]).total_seconds() / 3600.0
    start = seg[0]["time"]

    return {
        # --- identity / bookkeeping -------------------------------------
        "start_time": seg[0]["time"],
        "end_time": seg[-1]["time"],
        "n_pings": len(seg),          # diagnostic only -- cadence-dependent, never a feature
        "wall_hours": wall_hours,     # diagnostic only -- see active_hours
        # --- target components ------------------------------------------
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_used": soc_used,
        "distance_km": distance_km,
        # --- unified candidate features ---------------------------------
        "avg_speed": _wmean(speeds, sw) if speeds else None,
        "peak_speed": _wquantile(speeds, sw, 0.95) if speeds else None,
        "speed_std": _wstd(speeds, sw) if len(speeds) > 1 else None,
        "speed_iqr": _wiqr(speeds, sw),
        "moving_hours": moving_hours,
        "active_hours": active_hours,
        "pct_time_moving": 100.0 * moving_hours / active_hours,
        "pct_time_congested": 100.0 * congested_hours / moving_hours if moving_hours else None,
        "pct_time_highway": 100.0 * highway_hours / moving_hours if moving_hours else None,
        "avg_soc": _wmean(socs, dwell),
        "odometer_start": seg[0]["odo"],
        "start_hour": start.hour + start.minute / 60.0,
        "is_weekend": float(start.weekday() >= 5),
        "month": float(start.month),
    }


# Every unified feature the builder emits. The trainer takes its final
# subset from data/processed/unified_feature_ranking.csv rather than
# hardcoding one here, so re-running the selection re-tunes the model.
#
# Deliberately NOT candidates:
#   soc_start / soc_end  -- almost every epoch is a partial discharge in the
#       25-95% band, so a model keying on absolute SOC extrapolates wildly
#       at the 100%->0% point we actually predict. An earlier per-device
#       version of this did exactly that and emitted negative ranges.
#   distanceToEmpty      -- present on Intellicar and Citroen, but it is the
#       OEM's own range estimate, not an input: on the Citroen sample it
#       correlates 0.986 with SOC, i.e. it is SOC times a fixed constant and
#       carries no driving-condition signal. Broken outright on Tata (pinned
#       to a 1023 sentinel).
#   mean |dv/dt| accel   -- the obvious harshness proxy, but it is the one
#       feature that cannot survive the cadence gap: a delta over a 3-second
#       Intellicar step and one over a 5-minute Citroen step are not the
#       same quantity, and no reweighting fixes that.
CANDIDATE_FEATURES = [
    "avg_speed",
    "peak_speed",
    "speed_std",
    "speed_iqr",
    "moving_hours",
    "active_hours",
    "pct_time_moving",
    "pct_time_congested",
    "pct_time_highway",
    "avg_soc",
    "odometer_start",
    "start_hour",
    "is_weekend",
    "month",
    "rest_hours_before",  # cross-epoch, filled in by the builder
]

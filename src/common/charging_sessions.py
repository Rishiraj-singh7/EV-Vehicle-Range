"""Detect charging sessions from a vehicle's raw telemetry and classify each
as fast or slow. Device-agnostic: each device's own module (see
src/intellicar/charging.py, src/tata/charging.py) adapts its raw CSV columns
to the (time, soc, fast_flag) shape this module expects.

A "session" is a stretch of consecutive readings where SOC is actually
rising, tolerating small reporting gaps (see MAX_GAP_FOR_SAME_SESSION)
between two rising readings but not a large one. This is deliberately NOT
"any stretch where SOC never decreases" (see epoch_splitting's discharge-
epoch splitter for that simpler rule) -- real charging telemetry showed a
vehicle plugged in for 3+ days that only actively charged for ~2 hours and
then sat flat at 100% the rest of the time; treating the whole plugged-in
window as one "session" reports a nonsensical ~1%/hour rate for what was
actually a normal charge. Ending a session at the last point it was still
rising (and starting fresh after a gap or a drop) keeps duration/rate
meaningful. Only sessions where the total SOC gained is >
MIN_SOC_GAIN_FOR_SESSION count as a real charging session -- short
reconnects/blips below that are noise (relay chatter, a brief plug wiggle),
not someone actually charging.

Fast vs slow:
  - If the device reports its own charging flag (Intellicar's
    fast_charge_indicator), that's the source of truth: a session is "fast"
    if the flag is set on a majority of its rows.
  - Devices without a *validated* flag (Tata, Citroen) fall back to
    inferring from the charge rate: %SOC gained per hour. Validated against
    Intellicar's own flag on real data: slow sessions cluster at ~10-11%/hr,
    fast sessions at ~30-75%/hr, a clean gap either side of the 15%/hr
    threshold used below. (Tata's export does carry a data.hvChargeType
    field, but checked against real sessions' rates it showed no clean
    separation -- see src/tata/charging.py -- so it isn't treated as a
    trustworthy flag; only the charge rate is used.)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

MIN_SOC_GAIN_FOR_SESSION = 10.0  # % SOC -- below this, not counted as a session
FAST_RATE_THRESHOLD_PCT_PER_HOUR = 15.0  # fallback classification cutoff
MAX_GAP_FOR_SAME_SESSION = timedelta(minutes=20)  # ~4x Tata's 5-min ping cadence
SOC_NOISE_EPS = 0.01


@dataclass
class ChargingSession:
    vehicle: str
    start_time: datetime
    end_time: datetime
    soc_start: float
    soc_end: float
    soc_gained: float
    duration_hours: float
    rate_pct_per_hour: float
    is_fast: bool
    classified_by: str  # "device_flag" or "rate"


def _group_rising_segments(ordered, get_time, get_soc):
    """Group time-ordered rows into segments of consecutive *actually rising*
    readings, closing a segment (instead of extending it) whenever SOC fails
    to increase within MAX_GAP_FOR_SAME_SESSION of the last accepted point --
    that's either a real drop (driving started) or a long flat stretch
    (parked, already topped off) neither of which should count as "still
    charging" time. See module docstring for why this differs from a plain
    non-decreasing split.
    """
    if not ordered:
        return

    seg = [ordered[0]]
    for row in ordered[1:]:
        last = seg[-1]
        rose = get_soc(row) > get_soc(last) + SOC_NOISE_EPS
        within_gap = (get_time(row) - get_time(last)) <= MAX_GAP_FOR_SAME_SESSION
        if rose and within_gap:
            seg.append(row)
        else:
            if len(seg) >= 2:
                yield seg
            seg = [row]
    if len(seg) >= 2:
        yield seg


def find_charging_sessions(rows, vehicle, get_time, get_soc, get_fast_flag=None):
    """rows: raw per-reading records, any order (dicts or similar).
    get_time(row) -> datetime
    get_soc(row) -> float
    get_fast_flag(row) -> True/False/None, optional (device's own fast-charge flag)

    Returns list[ChargingSession], oldest first, restricted to sessions
    with soc_gained > MIN_SOC_GAIN_FOR_SESSION.
    """
    ordered = sorted(rows, key=get_time)
    sessions = []

    for seg in _group_rising_segments(ordered, get_time, get_soc):
        soc_start, soc_end = get_soc(seg[0]), get_soc(seg[-1])
        gained = soc_end - soc_start
        if gained <= MIN_SOC_GAIN_FOR_SESSION:
            continue

        start_t, end_t = get_time(seg[0]), get_time(seg[-1])
        duration_h = max((end_t - start_t).total_seconds() / 3600, 1e-6)
        rate = gained / duration_h

        if get_fast_flag is not None:
            flags = [f for f in (get_fast_flag(r) for r in seg) if f is not None]
        else:
            flags = []

        if flags:
            is_fast = sum(flags) >= len(flags) / 2
            classified_by = "device_flag"
        else:
            is_fast = rate >= FAST_RATE_THRESHOLD_PCT_PER_HOUR
            classified_by = "rate"

        sessions.append(
            ChargingSession(
                vehicle=vehicle,
                start_time=start_t,
                end_time=end_t,
                soc_start=soc_start,
                soc_end=soc_end,
                soc_gained=gained,
                duration_hours=duration_h,
                rate_pct_per_hour=rate,
                is_fast=is_fast,
                classified_by=classified_by,
            )
        )

    return sessions


def summarize_last_n(sessions, n=16):
    """Most recent n sessions (by end time) + a fast/slow count.

    Returns fewer than n if fewer are available -- callers should show
    `sessions_found` alongside `requested` so the UI can say "12 of the
    last 16" rather than silently padding.
    """
    recent = sorted(sessions, key=lambda s: s.end_time)[-n:]
    fast = sum(1 for s in recent if s.is_fast)
    return {
        "requested": n,
        "sessions_found": len(recent),
        "fast": fast,
        "slow": len(recent) - fast,
        "sessions": recent,
    }

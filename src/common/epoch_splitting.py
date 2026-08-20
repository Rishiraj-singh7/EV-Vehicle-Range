"""Shared "split a SOC timeline into segments" logic, used by both the
discharge-epoch builders (src/intellicar, src/tata -- for range-model
training) and charging_sessions.py (for the webapp's session history).

Two mirror-image cuts over the same kind of data:
  - a *discharge* segment is a run where SOC does not increase (driving/idle
    draining the battery) -- see split_by_soc_direction(..., rising=False)
  - a *charging* segment is a run where SOC does not decrease (plugged in)
    -- split_by_soc_direction(..., rising=True)

Both are "cut every time SOC moves the wrong way for this kind of segment,
start a new one" -- kept in one place so the two build_epochs.py scripts and
charging_sessions.py don't each reimplement the same loop slightly differently.
"""

SOC_NOISE_EPS = 0.01  # ignore SOC jitter smaller than this when deciding direction


def split_by_soc_direction(rows_sorted_by_time, get_soc, rising):
    """rows_sorted_by_time: time-ordered list of row objects (dict or similar).
    get_soc(row) -> float.
    rising=True  -> yield charging segments (SOC non-decreasing within a segment).
    rising=False -> yield discharge segments (SOC non-increasing within a segment).

    Yields lists of rows, each list length >= 1, in time order.
    """
    if not rows_sorted_by_time:
        return

    seg = [rows_sorted_by_time[0]]
    for row in rows_sorted_by_time[1:]:
        soc = get_soc(row)
        prev_soc = get_soc(seg[-1])
        moved_wrong_way = (
            (soc > prev_soc + SOC_NOISE_EPS) if not rising
            else (soc < prev_soc - SOC_NOISE_EPS)
        )
        if moved_wrong_way:
            yield seg
            seg = [row]
        else:
            seg.append(row)
    yield seg

"""Date-range helpers for ingestion."""

from __future__ import annotations

import datetime as dt

# Re-fetch this many recent days on each incremental run, to pick up Savant's
# overnight corrections (pitch reclassifications, late data).
DEFAULT_LOOKBACK_DAYS = 3


def optional_date(value: str) -> dt.date | None:
    """Parse YYYY-MM-DD; an empty string (unset job parameter) means None."""
    return dt.date.fromisoformat(value) if value else None


def resolve_window(
    start: dt.date | None,
    end: dt.date | None,
    today: dt.date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[dt.date, dt.date]:
    """Return the (start, end) date window to fetch, inclusive.

    With no dates, it's the incremental window: the last `lookback_days` days, ending
    yesterday. A start date with no end runs through yesterday.
    """
    yesterday = today - dt.timedelta(days=1)
    end = end or yesterday
    start = start or (yesterday - dt.timedelta(days=lookback_days - 1))
    if start > end:
        raise ValueError(f"start date {start} is after end date {end}")
    return start, end


def split_by_month(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Split an inclusive date range at calendar-month boundaries."""
    chunks = []
    while start <= end:
        next_month = (start.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        chunk_end = min(next_month - dt.timedelta(days=1), end)
        chunks.append((start, chunk_end))
        start = chunk_end + dt.timedelta(days=1)
    return chunks

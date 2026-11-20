"""Baseball Savant Statcast search (pitch-level CSV)."""

from __future__ import annotations

import csv
import datetime as dt
import io
import urllib.parse

SEARCH_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Regular season + postseason (wild card, division series, LCS, World Series).
# Spring training (S) and exhibitions are excluded; Savant includes them by default.
GAME_TYPES = ("R", "F", "D", "L", "W")

# Savant truncates a single search at this many rows. A full day is ~5,000 pitches, so
# hitting the cap means the response is incomplete.
ROW_CAP = 25_000

EXPECTED_FIRST_COLUMN = "pitch_type"


class SavantResponseError(ValueError):
    """The response isn't a complete Statcast CSV."""


def statcast_day_url(day: dt.date) -> str:
    params = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "hfGT": "".join(f"{t}|" for t in GAME_TYPES),
        "game_date_gt": day.isoformat(),
        "game_date_lt": day.isoformat(),
    }
    return f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"


def count_rows(content: bytes) -> int:
    """Validate a Statcast CSV response and return its number of data rows.

    An empty body means no pitches that day. Raises SavantResponseError if the body
    isn't a Statcast CSV (e.g., an HTML error page) or looks truncated at the row cap.
    """
    if not content.strip():
        return 0
    # utf-8-sig strips the byte-order mark Savant puts at the start of the file.
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    header = next(reader, [])
    if not header or header[0] != EXPECTED_FIRST_COLUMN:
        raise SavantResponseError(f"Unexpected header start: {header[:3]}")
    rows = sum(1 for _ in reader)
    if rows >= ROW_CAP:
        raise SavantResponseError(f"{rows} rows: response likely truncated at the row cap")
    return rows

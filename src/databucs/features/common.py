"""Shared pieces of the feature SQL."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

# Point-in-time windows cover this many days of pitches, ending with the last game day
# included. At opening day, that is the whole previous season.
WINDOW_DAYS = 365


@dataclass(frozen=True)
class Source:
    """Where the feature SQL reads from, and which game days it produces rows for."""

    fact_pitch: str  # fully qualified gold.fact_pitch
    players: str  # fully qualified silver.players
    start: dt.date | None = None  # first game day to produce rows for; None means all
    end: dt.date | None = None  # last game day to produce rows for; None means all

    def history_start(self) -> dt.date | None:
        """First game day the trailing windows for `start` need."""
        return None if self.start is None else self.start - dt.timedelta(days=WINDOW_DAYS - 1)

    def all_days(self) -> Source:
        """The same source without a date range."""
        return replace(self, start=None, end=None)


def date_filter(column: str, start: dt.date | None, end: dt.date | None) -> str:
    """A SQL condition limiting `column` to [start, end]; TRUE when both are None."""
    conditions = []
    if start is not None:
        conditions.append(f"{column} >= DATE'{start.isoformat()}'")
    if end is not None:
        conditions.append(f"{column} <= DATE'{end.isoformat()}'")
    return " AND ".join(conditions) or "TRUE"


def trailing_window(partition_by: str | None) -> str:
    """A window over the last WINDOW_DAYS days of game days, ending with the current row."""
    partition = f"PARTITION BY {partition_by} " if partition_by else ""
    return (
        f"({partition}ORDER BY unix_date(game_date) "
        f"RANGE BETWEEN {WINDOW_DAYS - 1} PRECEDING AND CURRENT ROW)"
    )


def shrunk(successes: str, trials: str, league_rate: str, prior: int) -> str:
    """SQL for a rate shrunk toward the league rate, as if `prior` league-average trials
    were added: (successes + prior * league) / (trials + prior)."""
    return f"({successes} + {prior} * {league_rate}) / ({trials} + {prior})"

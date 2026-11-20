"""Fetch raw MLB data and write it to the landing volume.

1. MLB Stats API schedule for the window, one file per month (with probable pitchers
   and lineups)
2. Statcast pitch-level CSV for each date with a completed game
3. MLB Stats API teams and players for each season that had games

Files are content-addressed (see `databucs.ingestion.landing`), so re-running a window
only writes files whose contents changed. Runs locally too:

    uv run fetch-raw --landing-path ./landing --start-date 2025-07-01 --end-date 2025-07-03
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from databucs.ingestion import savant, statsapi
from databucs.ingestion.dates import DEFAULT_LOOKBACK_DAYS, resolve_window, split_by_month
from databucs.ingestion.http import get_bytes
from databucs.ingestion.landing import has_landed, land

Fetcher = Callable[[str], bytes]


@dataclass
class Outcome:
    item: str
    written: bool = False
    skipped: bool = False  # already landed and outside the re-fetch window
    rows: int | None = None
    error: str | None = None


def optional_date(value: str) -> dt.date | None:
    """Parse YYYY-MM-DD; an empty string (unset job parameter) means None."""
    return dt.date.fromisoformat(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw MLB data to the landing volume.")
    parser.add_argument("--landing-path", required=True, type=Path)
    parser.add_argument("--start-date", type=optional_date, default=None)
    parser.add_argument("--end-date", type=optional_date, default=None)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--workers", type=int, default=3, help="Concurrent Statcast requests.")
    parser.add_argument(
        "--refetch-all",
        action="store_true",
        help="Re-download days that already have a landed file, not just recent ones.",
    )
    return parser.parse_args(argv)


def fetch_statcast_day(
    day: dt.date, landing: Path, fetch: Fetcher, refetch_before: dt.date | None = None
) -> Outcome:
    """Fetch and land one day of Statcast.

    Days before `refetch_before` that already have a landed file are skipped, so a
    restarted backfill resumes instead of re-downloading. Pass None to always fetch.
    """
    item = f"statcast {day}"
    directory = landing / "statcast" / str(day.year)
    stem = f"statcast_{day}"
    if refetch_before and day < refetch_before and has_landed(directory, stem, "csv"):
        return Outcome(item, skipped=True)
    try:
        content = fetch(savant.statcast_day_url(day))
        rows = savant.count_rows(content)
        if rows == 0:
            # Schedule says a game finished but Savant has no pitches (yet).
            return Outcome(item, rows=0, error="no pitches returned")
        result = land(content, directory, stem, "csv")
        return Outcome(item, written=result.written, rows=rows)
    except Exception as e:
        return Outcome(item, error=f"{type(e).__name__}: {e}")


def fetch_json(url: str, directory: Path, stem: str, fetch: Fetcher) -> Outcome:
    try:
        result = land(fetch(url), directory, stem, "json")
        return Outcome(stem, written=result.written)
    except Exception as e:
        return Outcome(stem, error=f"{type(e).__name__}: {e}")


def run(
    landing: Path,
    start: dt.date,
    end: dt.date,
    workers: int = 3,
    fetch: Fetcher = get_bytes,
    refetch_before: dt.date | None = None,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    api_dir = landing / "statsapi"
    seasons_with_games: set[int] = set()

    # Schedule one month at a time: a full season with lineups is ~15 MB, close to the
    # 16 MB limit for a single VARIANT value in the bronze table.
    for chunk_start, chunk_end in split_by_month(start, end):
        # Schedule first: it tells us which days need a Statcast pull.
        schedule_stem = f"schedule_{chunk_start}_{chunk_end}"
        try:
            schedule = fetch(statsapi.schedule_url(chunk_start, chunk_end))
        except Exception as e:
            outcomes.append(Outcome(schedule_stem, error=f"{type(e).__name__}: {e}"))
            continue
        result = land(schedule, api_dir / "schedule" / str(chunk_start.year), schedule_stem, "json")
        outcomes.append(Outcome(schedule_stem, written=result.written))

        game_days = statsapi.completed_game_dates(schedule)
        if game_days:
            seasons_with_games.add(chunk_start.year)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes += pool.map(
                lambda d: fetch_statcast_day(d, landing, fetch, refetch_before), game_days
            )

    for season in sorted(seasons_with_games):
        outcomes.append(
            fetch_json(statsapi.teams_url(season), api_dir / "teams", f"teams_{season}", fetch)
        )
        outcomes.append(
            fetch_json(
                statsapi.players_url(season), api_dir / "players", f"players_{season}", fetch
            )
        )
    return outcomes


def summarize(outcomes: list[Outcome]) -> str:
    errors = [o for o in outcomes if o.error]
    lines = [f"ERROR {o.item}: {o.error}" for o in errors]
    written = sum(o.written for o in outcomes)
    skipped = sum(o.skipped for o in outcomes)
    unchanged = len(outcomes) - written - skipped - len(errors)
    pitches = sum(o.rows or 0 for o in outcomes if o.written)
    lines.append(
        f"{len(outcomes)} items: {written} new files ({pitches} new pitch rows), "
        f"{unchanged} unchanged, {skipped} skipped (already landed), {len(errors)} errors"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start, end = resolve_window(args.start_date, args.end_date, dt.date.today(), args.lookback_days)
    print(f"Fetching {start} to {end} into {args.landing_path}")
    today = dt.date.today()
    refetch_before = None if args.refetch_all else today - dt.timedelta(days=args.lookback_days)
    outcomes = run(
        args.landing_path, start, end, workers=args.workers, refetch_before=refetch_before
    )
    print(summarize(outcomes))
    # Days where a finished game has no Statcast data yet are warnings, not failures:
    # the next incremental run picks them up.
    failures = [o for o in outcomes if o.error and o.error != "no pitches returned"]
    if failures:
        raise RuntimeError(f"{len(failures)} fetches failed; see log above")


if __name__ == "__main__":
    main()

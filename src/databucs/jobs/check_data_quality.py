"""Check the data quality results for recent days and fail the run if any look wrong.

Runs in the ingest job after the lakehouse pipeline. It reads `pitches_daily_quality` for
the job's date window (the same window fetch_raw used) and prints one line per day plus
the quarantined pitches by rule. If any day has issues it raises, so the run fails
and the job's failure notification goes out. The data is still published; this only
raises the alarm.

One exception: a played game with no pitches on the newest day of the window is a warning.
Savant sometimes posts a day late, fetch_raw treats that as a warning too, and the next
run re-fetches the day. If it is still missing then, it fails.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass

from databucs.ingestion.dates import DEFAULT_LOOKBACK_DAYS, optional_date, resolve_window

LATE_DATA_ISSUE = "games_missing_pitches"


@dataclass(frozen=True)
class Day:
    game_date: dt.date
    games_played: int
    games_with_pitches: int
    pitches: int
    pitches_rejected: int
    pitches_with_values_nulled: int
    issues: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--start-date", type=optional_date, default=None)
    parser.add_argument("--end-date", type=optional_date, default=None)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    return parser.parse_args(argv)


def classify(days: list[Day], newest: dt.date) -> tuple[list[str], list[str]]:
    """Split the days' issues into (failures, warnings), as printable lines."""
    failures, warnings = [], []
    for day in days:
        for issue in day.issues:
            line = f"{day.game_date}: {issue}"
            if issue == LATE_DATA_ISSUE and day.game_date == newest:
                warnings.append(f"{line} (newest day; Savant may post it late)")
            else:
                failures.append(line)
    return failures, warnings


def format_day(day: Day) -> str:
    return (
        f"{day.game_date}: {day.games_with_pitches}/{day.games_played} games with pitches, "
        f"{day.pitches} pitches, {day.pitches_rejected} rejected, "
        f"{day.pitches_with_values_nulled} with values nulled"
        + (f"  ISSUES: {', '.join(day.issues)}" if day.issues else "")
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start, end = resolve_window(args.start_date, args.end_date, dt.date.today(), args.lookback_days)
    silver = f"`{args.catalog}`.`{args.silver_schema}`"

    from pyspark.sql import SparkSession  # imported here so unit tests don't need Spark

    spark = SparkSession.builder.getOrCreate()
    window = {"start": start, "end": end}

    days = [
        Day(
            game_date=r.game_date,
            games_played=r.games_played,
            games_with_pitches=r.games_with_pitches,
            pitches=r.pitches,
            pitches_rejected=r.pitches_rejected,
            pitches_with_values_nulled=r.pitches_with_values_nulled,
            issues=tuple(r.issues),
        )
        for r in spark.sql(
            f"SELECT * FROM {silver}.pitches_daily_quality "
            "WHERE game_date BETWEEN :start AND :end ORDER BY game_date",
            args=window,
        ).collect()
    ]
    print(f"Data quality for {start} to {end}:")
    print("\n".join(format_day(d) for d in days) if days else "No games or pitches in window.")

    quarantined = spark.sql(
        f"SELECT action, rule, count(*) AS pitches FROM {silver}.pitches_quarantine "
        "LATERAL VIEW explode(failed_rules) AS rule "
        "WHERE game_date BETWEEN :start AND :end GROUP BY ALL ORDER BY pitches DESC",
        args=window,
    ).collect()
    if quarantined:
        print("Quarantined pitches by rule:")
        for r in quarantined:
            print(f"  {r.action:<14} {r.rule:<26} {r.pitches}")

    failures, warnings = classify(days, newest=end)
    for line in warnings:
        print(f"WARNING {line}")
    if failures:
        raise RuntimeError(f"{len(failures)} data quality issues:\n" + "\n".join(failures))
    print("No data quality issues.")


if __name__ == "__main__":
    main()

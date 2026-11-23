"""Build the feature tables in Unity Catalog with Feature Engineering.

Runs in the ingest job after the data quality check. For each feature table it computes the
rows for the job's date window from gold.fact_pitch, then writes them with
FeatureEngineeringClient:

* first run (table missing): fe.create_table with the primary keys and time series column,
  which makes it a feature table usable with FeatureLookup
* later runs: fe.write_table, which merges on the primary key, so re-running a window
  replaces those rows and leaves the rest alone

A table that doesn't exist yet is always built from all of fact_pitch, and
--full-rebuild drops each table first so that happens to every one. Use it after
changing a feature definition. See the modules in databucs.features for what each table
holds.
"""

from __future__ import annotations

import argparse
import datetime as dt
from types import ModuleType

from databucs.features import batter_profile, pitcher_arsenal
from databucs.features.common import Source
from databucs.ingestion.dates import DEFAULT_LOOKBACK_DAYS, optional_date, resolve_window

TABLES: list[ModuleType] = [batter_profile, pitcher_arsenal]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the feature tables.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--start-date", type=optional_date, default=None)
    parser.add_argument("--end-date", type=optional_date, default=None)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument(
        "--full-rebuild",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Drop and rebuild every table from scratch (true/false; a bare flag means true).",
    )
    parser.add_argument(
        "--tables",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="Comma-separated table names to build (default: all).",
    )
    return parser.parse_args(argv)


def parse_bool(value: str) -> bool:
    """Parse a job parameter like "true"/"false"; empty means false."""
    lowered = value.strip().lower()
    if lowered in ("", "false", "0", "no"):
        return False
    if lowered in ("true", "1", "yes"):
        return True
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def select_tables(names: list[str] | None) -> list[ModuleType]:
    if not names:
        return TABLES
    by_name = {t.TABLE: t for t in TABLES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown feature tables: {', '.join(unknown)}")
    return [by_name[n] for n in names]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    gold = f"`{args.catalog}`.`{args.gold_schema}`"
    silver = f"`{args.catalog}`.`{args.silver_schema}`"
    start, end = resolve_window(args.start_date, args.end_date, dt.date.today(), args.lookback_days)
    source = Source(
        fact_pitch=f"{gold}.fact_pitch", players=f"{silver}.players", start=start, end=end
    )
    print(f"Window: game days {start} to {end}" + (" (ignored: full rebuild)" * args.full_rebuild))

    # Imported here so unit tests don't need Spark or the Feature Engineering client
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()

    for module in select_tables(args.tables):
        name = f"{args.catalog}.{args.ml_schema}.{module.TABLE}"
        if args.full_rebuild:
            spark.sql(f"DROP TABLE IF EXISTS {name}")
        if spark.catalog.tableExists(name):
            df = spark.sql(module.build_sql(source))
            fe.write_table(name=name, df=df, mode="merge")
            action = "merged the window"
        else:
            # A new table always starts from all of history, whatever the window
            df = spark.sql(module.build_sql(source.all_days()))
            fe.create_table(
                name=name,
                primary_keys=module.PRIMARY_KEYS,
                timeseries_column=module.TIMESERIES_COLUMN,
                df=df,
                description=(module.__doc__ or "").strip().splitlines()[0],
            )
            action = "created from all of history"
        rows = spark.table(name).count()
        print(f"{name}: {action}; {rows} rows in table")


if __name__ == "__main__":
    main()

"""Check that the feature tables give point-in-time correct training sets.

Builds a training set with FeatureEngineeringClient.create_training_set: pitches from the
last week of game days (labelled with run_value), joined to both feature tables with
FeatureLookup. Then it checks:

1. No leakage: every looked-up batter and arsenal row only covers game days before the
   pitch's game day.
2. The windows are right: for each batter row used, pitches_seen equals a direct count of
   that batter's pitches in the trailing window.
3. Coverage: most pitches have batter and arsenal features (debuts and new pitch types
   have none yet).
4. Future lookup: looking up the same batters and pitchers for a game 30 days after the
   last game day returns their latest rows.

Raises on any failure, so the job run fails.
"""

from __future__ import annotations

import argparse
import datetime as dt

from databucs.features import batter_profile, pitcher_arsenal
from databucs.features.common import WINDOW_DAYS

SAMPLE_DAYS = 7
FUTURE_DAYS = 30
# Opening week of 2026 had 99.4% batter and 96.3% arsenal coverage (debuts and new pitch
# types cluster early); midseason both are above 98.5%. 90% catches a broken join without
# false alarms in April.
MIN_BATTER_COVERAGE = 0.90
MIN_ARSENAL_COVERAGE = 0.90


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check point-in-time feature lookups.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    return parser.parse_args(argv)


def coverage_failures(total: int, with_batter: int, with_arsenal: int) -> list[str]:
    """Coverage problems as printable lines (empty when coverage is fine)."""
    failures = []
    if total == 0:
        return ["no pitches in the sample"]
    if with_batter / total < MIN_BATTER_COVERAGE:
        failures.append(f"batter features for only {with_batter / total:.1%} of pitches")
    if with_arsenal / total < MIN_ARSENAL_COVERAGE:
        failures.append(f"arsenal features for only {with_arsenal / total:.1%} of pitches")
    return failures


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"
    ml = f"{args.catalog}.{args.ml_schema}"

    from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()

    last_day: dt.date = spark.sql(f"SELECT max(game_date) FROM {fact_pitch}").first()[0]
    first_day = last_day - dt.timedelta(days=SAMPLE_DAYS - 1)
    print(f"Sample: pitches from {first_day} to {last_day}")

    labels = spark.sql(
        f"""
        SELECT game_pk, at_bat_number, pitch_number, game_date, batter_id, pitcher_id,
               pitch_type, run_value
        FROM {fact_pitch}
        WHERE game_date BETWEEN :first AND :last AND pitch_type IS NOT NULL
        """,
        args={"first": first_day, "last": last_day},
    )

    def lookups() -> list:
        return [
            FeatureLookup(
                table_name=f"{ml}.{batter_profile.TABLE}",
                lookup_key="batter_id",
                timestamp_lookup_key="game_date",
            ),
            FeatureLookup(
                table_name=f"{ml}.{pitcher_arsenal.TABLE}",
                lookup_key=["pitcher_id", "pitch_type"],
                timestamp_lookup_key="game_date",
            ),
        ]

    training = fe.create_training_set(df=labels, feature_lookups=lookups(), label="run_value")
    # No .cache(): serverless compute does not support persisting DataFrames
    df = training.load_df()
    print(f"Training set: {df.count()} rows, {len(df.columns)} columns")

    failures: list[str] = []

    # 1. No leakage
    leaks = df.filter(
        (F.col("batter_data_through") >= F.col("game_date"))
        | (F.col("arsenal_data_through") >= F.col("game_date"))
    ).count()
    print(f"Rows using data from the game day or later: {leaks}")
    if leaks:
        failures.append(f"{leaks} rows use feature data from the game day or later")

    # 2. Windows: recount each batter row's pitches straight from fact_pitch
    used = df.select("batter_id", "batter_data_through", "pitches_seen").distinct()
    used.createOrReplaceTempView("used_batter_rows")
    mismatches = spark.sql(
        f"""
        SELECT u.batter_id, u.batter_data_through, u.pitches_seen, count(p.batter_id) AS counted
        FROM used_batter_rows u
        LEFT JOIN {fact_pitch} p
          ON p.batter_id = u.batter_id
          AND p.game_date BETWEEN date_sub(u.batter_data_through, {WINDOW_DAYS - 1})
            AND u.batter_data_through
          AND NOT p.is_pitch_clock_violation
        WHERE u.batter_data_through IS NOT NULL
        GROUP BY ALL
        HAVING u.pitches_seen <> counted
        """
    ).collect()
    print(
        f"Batter rows checked against a direct count: {used.count()}, mismatches: {len(mismatches)}"
    )
    if mismatches:
        failures.append(
            f"{len(mismatches)} batter rows disagree with a direct count, e.g. {mismatches[0]}"
        )

    # 3. Coverage
    counts = df.agg(
        F.count("*").alias("total"),
        F.count("batter_data_through").alias("batter"),
        F.count("arsenal_data_through").alias("arsenal"),
    ).first()
    print(f"Coverage of {counts.total} pitches: batter {counts.batter}, arsenal {counts.arsenal}")
    failures += coverage_failures(counts.total, counts.batter, counts.arsenal)

    # 4. Future lookup: a game FUTURE_DAYS after the last game day gets the latest rows
    future_day = last_day + dt.timedelta(days=FUTURE_DAYS)
    future = (
        labels.select("batter_id", "pitcher_id", "pitch_type")
        .distinct()
        .withColumn("game_date", F.lit(future_day))
    )
    future_df = fe.create_training_set(df=future, feature_lookups=lookups(), label=None).load_df()
    latest = spark.sql(
        f"SELECT batter_id, max(batter_data_through) AS latest FROM {ml}.{batter_profile.TABLE} "
        "GROUP BY batter_id"
    )
    stale = (
        future_df.select("batter_id", "batter_data_through")
        .distinct()
        .join(latest, "batter_id")
        .filter(F.col("batter_data_through") != F.col("latest"))
        .count()
    )
    print(
        f"Future lookup for {future_day}: {future_df.count()} rows, "
        f"{stale} batters not on their latest row"
    )
    if stale:
        failures.append(f"future lookup returned an older row for {stale} batters")

    if failures:
        raise RuntimeError("Feature checks failed:\n" + "\n".join(failures))
    print("All feature checks passed.")


if __name__ == "__main__":
    main()

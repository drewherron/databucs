"""Score every Pirates pitcher's pitch options against an opponent's batters.

1. Pick the series (databucs.recommender.inputs.choose_series): the next one within
   --max-days-ahead, or a demo on the last one played, or the --series-start/--opponent
   given. The lineup is projected from the opponent's recent starts unless --lineup lists
   batter ids; the pitchers are everyone who pitched for the team recently plus the
   probable starters, unless --pitchers lists them.
2. Build one row per pitcher x batter x scenario x pitch type x zone x point (nine points
   per zone; see databucs.recommender.options) and score it with the five @champion
   outcome models.
3. Expected run value per point, from the models' result probabilities and the run values
   of the training seasons (positive is good for the batter, so lower is better for the
   pitcher). Each option gets the average over its nine points and their range.
4. Compare with the pitcher's usual choice: his pitch type x zone mix in this count
   (or count group, if he has few pitches in it) against same- or opposite-side hitters,
   weighted over the options scored.
5. Write ml.pitch_option_scores, replacing the rows for this team, opponent and series
   start.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict

from databucs.ingestion.dates import optional_date
from databucs.jobs.train_outcome_model import parse_seasons
from databucs.models.run_values import RESULTS
from databucs.models.scoring import expected_value_sql, register_run_values, score_outcomes
from databucs.recommender import inputs, options

TABLE = "pitch_option_scores"


def parse_ids(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score pitch options for a series.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--silver-schema", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--team-id", type=int, default=inputs.PIRATES_TEAM_ID)
    parser.add_argument("--opponent-id", type=lambda s: int(s) if s.strip() else None)
    parser.add_argument("--series-start", type=optional_date, default=None)
    parser.add_argument("--today", type=optional_date, default=None, help="Default: today.")
    parser.add_argument("--max-days-ahead", type=int, default=10)
    parser.add_argument("--lineup", type=parse_ids, default=[], help="Batter ids in batting order.")
    parser.add_argument("--pitchers", type=parse_ids, default=[], help="Pitcher ids.")
    parser.add_argument(
        "--run-value-seasons",
        type=parse_seasons,
        default=[2023, 2024, 2025],
        help="Seasons to average run values over (the models' training seasons).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    silver = f"`{args.catalog}`.`{args.silver_schema}`"
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"
    ml = f"{args.catalog}.{args.ml_schema}"
    games_table, lineups_table, players_table = (
        f"{silver}.games",
        f"{silver}.lineups",
        f"{silver}.players",
    )
    arsenal_table = f"{ml}.pitcher_arsenal_features"

    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()

    # 1. Target series, lineup and pitchers
    games = [
        inputs.Game(**row.asDict())
        for row in spark.sql(inputs.games_sql(games_table, args.team_id)).collect()
    ]
    target = inputs.choose_series(
        games,
        args.team_id,
        args.today or dt.date.today(),
        args.max_days_ahead,
        args.series_start,
        args.opponent_id,
    )
    print(f"Target: {target}")

    if args.lineup:
        lineup = spark.createDataFrame(
            [(b, spot) for spot, b in enumerate(args.lineup, start=1)],
            "batter_id int, lineup_spot int",
        )
        lineup_source = "custom"
    else:
        lineup = spark.sql(inputs.projected_lineup_sql(games_table, lineups_table, target)).select(
            "batter_id", "lineup_spot"
        )
        lineup_source = "projected"
    lineup.createOrReplaceTempView("target_lineup")

    if args.pitchers:
        pitcher_ids = set(args.pitchers)
    else:
        pitcher_ids = {
            r.pitcher_id
            for r in spark.sql(
                inputs.recent_pitchers_sql(fact_pitch, games_table, players_table, target)
            ).collect()
        } | set(target.probable_starters)
    spark.createDataFrame(
        [(p, p in target.probable_starters) for p in sorted(pitcher_ids)],
        "pitcher_id int, is_probable_starter boolean",
    ).createOrReplaceTempView("target_pitchers")
    print(f"Lineup ({lineup_source}): {lineup.count()} batters; pitchers: {len(pitcher_ids)}")

    spark.createDataFrame([asdict(s) for s in options.SCENARIOS]).withColumnRenamed(
        "name", "scenario"
    ).drop("description").createOrReplaceTempView("scenarios")
    spark.createDataFrame([asdict(p) for p in options.zone_points()]).createOrReplaceTempView(
        "zone_points"
    )

    # 2. Request rows, scored by each outcome model
    requests = spark.sql(inputs.request_rows_sql(fact_pitch, players_table, arsenal_table, target))
    requests.createOrReplaceTempView("requests")
    print(f"Scoring {requests.count()} hypothetical pitches")
    score_outcomes(fe, requests, ml, inputs.POINT_KEY).createOrReplaceTempView("scored")

    # 3. Expected run value per point, then per option
    register_run_values(spark, fact_pitch, args.run_value_seasons, "run_values_wide")
    probabilities, expected = expected_value_sql()
    point_key = ", ".join(inputs.POINT_KEY)
    spark.sql(
        f"""
        WITH probabilities AS (
          SELECT r.*, {probabilities}
          FROM requests r
          JOIN scored USING ({point_key})
        )
        SELECT p.*, {expected} AS expected_run_value
        FROM probabilities p
        JOIN run_values_wide v USING (balls, strikes, outs, base_state)
        """
    ).createOrReplaceTempView("points")

    option_key = ", ".join(inputs.OPTION_KEY)
    mean_probabilities = ", ".join(f"avg(p_{r}) AS p_{r}" for r in RESULTS)
    spark.sql(
        f"""
        SELECT {option_key},
          first(pitcher_name) AS pitcher_name, first(is_probable_starter) AS is_probable_starter,
          first(batter_name) AS batter_name, first(lineup_spot) AS lineup_spot,
          first(pitcher_throws) AS pitcher_throws, first(batter_stands) AS batter_stands,
          first(balls) AS balls, first(strikes) AS strikes,
          first(pitch_usage_rate) AS pitch_usage_rate,
          first(pitcher_pitches_thrown) AS pitcher_pitches_thrown,
          avg(expected_run_value) AS expected_run_value,
          min(expected_run_value) AS expected_run_value_low,
          max(expected_run_value) AS expected_run_value_high,
          {mean_probabilities}
        FROM points
        GROUP BY {option_key}
        """
    ).createOrReplaceTempView("option_values")

    # 4. The pitcher's usual choice, over the options scored
    spark.sql(inputs.usage_sql(fact_pitch, target)).createOrReplaceTempView("usage")
    situation = "pitcher_id, batter_id, scenario"
    result = spark.sql(
        f"""
        WITH options AS (
          SELECT o.*, coalesce(u.usage, 0) AS usual_share, u.usual_by_exact_count
          FROM option_values o
          LEFT JOIN usage u
            ON u.pitcher_id = o.pitcher_id
            AND u.balls = o.balls
            AND u.strikes = o.strikes
            AND u.same_side = (o.pitcher_throws = o.batter_stands)
            AND u.pitch_type = o.pitch_type
            AND u.zone = o.zone
        ),
        usual AS (
          SELECT {situation},
            -- NULL when the pitcher has no history in this count group
            sum(usual_share * expected_run_value) / nullif(sum(usual_share), 0)
              AS usual_expected_run_value,
            sum(usual_share) AS usual_share_covered,
            bool_or(usual_by_exact_count) AS usual_by_exact_count
          FROM options
          GROUP BY ALL
        )
        SELECT
          {args.team_id} AS team_id,
          {target.opponent_id} AS opponent_id,
          DATE'{target.series_start}' AS series_start,
          '{target.mode}' AS target_mode,
          '{lineup_source}' AS lineup_source,
          o.* EXCEPT (balls, strikes, usual_by_exact_count),
          usual.usual_expected_run_value,
          usual.usual_share_covered,
          usual.usual_by_exact_count,
          o.expected_run_value - usual.usual_expected_run_value AS vs_usual,
          rank() OVER (PARTITION BY {situation} ORDER BY o.expected_run_value) AS option_rank,
          current_timestamp() AS scored_at
        FROM options o
        JOIN usual USING ({situation})
        """
    )

    # 5. Replace this series' rows
    table = f"{ml}.{TABLE}"
    writer = result.write.format("delta")
    if spark.catalog.tableExists(table):
        # mergeSchema: new output columns are added rather than failing the write
        writer = (
            writer.mode("overwrite")
            .option("mergeSchema", "true")
            .option(
                "replaceWhere",
                f"team_id = {args.team_id} AND opponent_id = {target.opponent_id}"
                f" AND series_start = DATE'{target.series_start}'",
            )
        )
    writer.saveAsTable(table)

    written = spark.table(table).filter(
        (F.col("opponent_id") == target.opponent_id)
        & (F.col("series_start") == F.lit(target.series_start))
    )
    print(f"Wrote {written.count()} options to {table}")
    written.filter("option_rank <= 3").orderBy(
        F.desc("is_probable_starter"), "pitcher_id", "lineup_spot", "scenario", "option_rank"
    ).select(
        "pitcher_name",
        "batter_name",
        "scenario",
        "option_rank",
        "pitch_type",
        "zone",
        F.round("expected_run_value", 3).alias("erv"),
        F.round("expected_run_value_low", 3).alias("low"),
        F.round("expected_run_value_high", 3).alias("high"),
        F.round("vs_usual", 3).alias("vs_usual"),
    ).show(30, truncate=False)


if __name__ == "__main__":
    main()

"""Score every Pirates pitcher's pitch options against an opponent's batters.

1. Pick the series (databucs.recommender.inputs.choose_series): the next one within
   --max-days-ahead, or a demo on the last one played, or the --series-start/--opponent
   given. The lineup is projected from the opponent's recent starts unless --lineup lists
   batter ids; the pitchers are everyone who pitched for the team recently plus the
   probable starters, unless --pitchers lists them.
2. Score each pitcher x batter x scenario x pitch type at every landing point of a regular
   grid with the five @champion outcome models (databucs.recommender.options), and turn
   that into expected run value with the run values of the training seasons (positive is
   good for the batter, so lower is better for the pitcher).
3. Options: aiming at a zone. Each of a zone's nine aim points is worth the average over
   where an aimed pitch lands (a normal miss of --miss-sd-ft); the option gets the average
   of its aim points and their range.
4. The pitcher's usual choice, valued the same way but at where his pitches really landed:
   his mix of pitch type x landing point in this count (or count group, if he has few
   pitches in it) against same- or opposite-side hitters. Both sides of the comparison
   include misses. Each option also shows how often he throws that pitch type into that
   zone (usual_share), as context, and his mix's result probabilities (usual_p_<result>)
   for the plate-appearance simulator.
5. Label each option for the report: pitch name, location from the batter's point of view,
   and a tier relative to the pitcher's usual mix (options.STRONG_MARGIN / AVOID_MARGIN).
6. Write ml.pitch_option_scores, replacing the rows for this team, opponent and series
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
    parser.add_argument(
        "--miss-sd-ft",
        type=float,
        default=options.MISS_SD_FT,
        help="How far pitches miss where they're aimed: a normal spread, in feet.",
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

    batters = spark.sql(inputs.batters_sql(fact_pitch, players_table, target))
    batters.createOrReplaceTempView("target_batters")
    spark.createDataFrame(
        [{**asdict(s), "scenario_order": i + 1} for i, s in enumerate(options.SCENARIOS)]
    ).withColumnRenamed("name", "scenario").withColumnRenamed(
        "description", "scenario_label"
    ).createOrReplaceTempView("scenarios")
    spark.createDataFrame([asdict(z) for z in options.zone_labels()]).createOrReplaceTempView(
        "zone_labels"
    )
    opponent_name = (
        spark.sql(inputs.team_name_sql(f"{silver}.teams", target.opponent_id)).first().team_name
    )
    spark.createDataFrame([asdict(p) for p in options.landing_points()]).createOrReplaceTempView(
        "landing_points"
    )
    # Where pitches aimed at each aim point land, per batter (the miss is in feet, the grid
    # in fractions of the batter's zone)
    spark.createDataFrame(
        [
            (b.batter_id, *row)
            for b in batters.collect()
            for row in options.aim_weights(b.sz_top - b.sz_bot, args.miss_sd_ft)
        ],
        "batter_id int, zone int, point int, land int, weight double",
    ).createOrReplaceTempView("aim_weights")

    # 2. Every landing point, scored by each outcome model, as expected run value
    requests = spark.sql(inputs.request_rows_sql(fact_pitch, players_table, arsenal_table, target))
    requests.createOrReplaceTempView("requests")
    print(f"Scoring {requests.count()} hypothetical pitches")
    score_outcomes(fe, requests, ml, inputs.LANDING_KEY).createOrReplaceTempView("scored")

    register_run_values(spark, fact_pitch, args.run_value_seasons, "run_values_wide")
    probabilities, expected = expected_value_sql()
    landing_key = ", ".join(inputs.LANDING_KEY)
    p_columns = ", ".join(f"p_{r}" for r in RESULTS)
    spark.sql(
        f"""
        WITH probabilities AS (
          SELECT r.*, {probabilities}
          FROM requests r
          JOIN scored USING ({landing_key})
        )
        SELECT {landing_key}, {p_columns}, {expected} AS expected_run_value
        FROM probabilities p
        JOIN run_values_wide v USING (balls, strikes, outs, base_state)
        """
    ).createOrReplaceTempView("landings")

    # 3. Options: aim points valued over where they land, then zones over their aim points
    situation = ", ".join(inputs.SITUATION_KEY)
    option_key = ", ".join(inputs.OPTION_KEY)
    weighted = ", ".join(f"sum(w.weight * l.p_{r}) AS p_{r}" for r in RESULTS)
    mean_probabilities = ", ".join(f"avg(p_{r}) AS p_{r}" for r in RESULTS)
    spark.sql(
        f"""
        WITH aims AS (
          SELECT l.pitcher_id, l.batter_id, l.scenario, l.pitch_type, w.zone, w.point,
            sum(w.weight * l.expected_run_value) AS expected_run_value,
            {weighted}
          FROM landings l
          JOIN aim_weights w ON w.batter_id = l.batter_id AND w.land = l.land
          GROUP BY ALL
        )
        SELECT {option_key},
          avg(expected_run_value) AS expected_run_value,
          min(expected_run_value) AS expected_run_value_low,
          max(expected_run_value) AS expected_run_value_high,
          {mean_probabilities}
        FROM aims
        GROUP BY ALL
        """
    ).createOrReplaceTempView("option_values")
    spark.sql(
        """
        SELECT DISTINCT r.pitcher_id, r.batter_id, r.scenario, r.pitch_type, r.pitch_name,
          r.pitcher_name, r.is_probable_starter, r.batter_name, r.lineup_spot,
          r.pitcher_throws, r.batter_stands, r.balls, r.strikes, r.pitch_usage_rate,
          r.pitcher_pitches_thrown, s.scenario_label, s.scenario_order
        FROM requests r
        JOIN scenarios s USING (scenario)
        """
    ).createOrReplaceTempView("option_info")

    # 4. The pitcher's usual choice: his real landing spots, valued like the options
    spark.sql(inputs.usage_sql(fact_pitch, target, "land")).createOrReplaceTempView("usage_land")
    spark.sql(inputs.usage_sql(fact_pitch, target, "zone")).createOrReplaceTempView("usage_zone")
    same_usage = """
            u.pitcher_id = i.pitcher_id AND u.balls = i.balls AND u.strikes = i.strikes
            AND u.same_side = (i.pitcher_throws = i.batter_stands)
            AND u.pitch_type = i.pitch_type"""
    # His usual mix's result probabilities too, for the plate-appearance simulator
    usual_probabilities = ", ".join(
        f"sum(u.usage * l.p_{r}) / nullif(sum(u.usage), 0) AS usual_p_{r}" for r in RESULTS
    )
    usual_p_columns = ", ".join(f"usual.usual_p_{r}" for r in RESULTS)
    result = spark.sql(
        f"""
        WITH usual AS (
          SELECT i.pitcher_id, i.batter_id, i.scenario,
            -- NULL when the pitcher has no history in this count
            sum(u.usage * l.expected_run_value) / nullif(sum(u.usage), 0)
              AS usual_expected_run_value,
            {usual_probabilities},
            sum(u.usage) AS usual_share_covered,
            bool_or(u.usual_by_exact_count) AS usual_by_exact_count
          FROM option_info i
          JOIN usage_land u ON {same_usage}
          JOIN landings l
            ON l.pitcher_id = i.pitcher_id AND l.batter_id = i.batter_id
            AND l.scenario = i.scenario AND l.pitch_type = i.pitch_type AND l.land = u.land
          GROUP BY ALL
        )
        SELECT
          {args.team_id} AS team_id,
          {target.opponent_id} AS opponent_id,
          DATE'{target.series_start}' AS series_start,
          '{target.mode}' AS target_mode,
          '{lineup_source}' AS lineup_source,
          '{opponent_name}' AS opponent_name,
          {args.miss_sd_ft} AS miss_sd_ft,
          o.*,
          i.* EXCEPT ({situation}, pitch_type, balls, strikes),
          coalesce(u.usage, 0) AS usual_share,
          usual.usual_expected_run_value,
          usual.usual_share_covered,
          usual.usual_by_exact_count,
          {usual_p_columns},
          o.expected_run_value - usual.usual_expected_run_value AS vs_usual,
          zl.location, zl.grid_x, zl.grid_y,
          CASE
            WHEN o.expected_run_value <= usual.usual_expected_run_value - {options.STRONG_MARGIN}
              THEN 'strong'
            WHEN o.expected_run_value >= usual.usual_expected_run_value + {options.AVOID_MARGIN}
              THEN 'avoid'
            ELSE 'fine'
          END AS tier,
          rank() OVER (
            PARTITION BY o.pitcher_id, o.batter_id, o.scenario ORDER BY o.expected_run_value
          ) AS option_rank,
          current_timestamp() AS scored_at
        FROM option_values o
        JOIN option_info i
          ON i.pitcher_id = o.pitcher_id AND i.batter_id = o.batter_id
          AND i.scenario = o.scenario AND i.pitch_type = o.pitch_type
        LEFT JOIN usage_zone u ON {same_usage} AND u.zone = o.zone
        LEFT JOIN usual
          ON usual.pitcher_id = o.pitcher_id AND usual.batter_id = o.batter_id
          AND usual.scenario = o.scenario
        JOIN zone_labels zl ON zl.zone = o.zone AND zl.batter_stands = i.batter_stands
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
        F.round("usual_share", 3).alias("his_share"),
    ).show(30, truncate=False)


if __name__ == "__main__":
    main()

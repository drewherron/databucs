"""Simulate every Pirates pitcher against every batter in a series' lineup, a whole plate
appearance at a time (databucs.recommender.simulate), and rank the pitchers.

1. Read the series' options from <ml schema>.pitch_option_scores: the bases-empty, one-out
   scenarios, which cover all twelve counts. By default the series scored most recently.
2. Check that each option's result probabilities and the run values give back its
   expected run value, so the simulator values pitches exactly as the report does.
3. Simulate each pitcher x batter under three policies: his usual mix, the report's top
   option at each count, and the best plan for the whole plate appearance.
4. Write <ml schema>.matchup_simulations, replacing this series' rows, and print the
   pitchers ranked by runs per plate appearance against the lineup with their usual mix.
"""

from __future__ import annotations

import argparse

import numpy as np

from databucs.ingestion.dates import optional_date
from databucs.jobs.recommend_pitches import TABLE as OPTIONS_TABLE
from databucs.jobs.train_outcome_model import parse_seasons
from databucs.models.run_values import RESULTS
from databucs.models.scoring import register_run_values
from databucs.recommender import inputs, options, simulate

TABLE = "matchup_simulations"
TOLERANCE = 1e-6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate plate appearances for a series.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--gold-schema", required=True)
    parser.add_argument("--ml-schema", required=True)
    parser.add_argument("--team-id", type=int, default=inputs.PIRATES_TEAM_ID)
    parser.add_argument("--opponent-id", type=lambda s: int(s) if s.strip() else None)
    parser.add_argument("--series-start", type=optional_date, default=None)
    parser.add_argument(
        "--run-value-seasons",
        type=parse_seasons,
        default=[2023, 2024, 2025],
        help="Seasons to average run values over (as in recommend_pitches).",
    )
    parser.add_argument(
        "--options-table", default=OPTIONS_TABLE, help="recommend_pitches output, in ml."
    )
    parser.add_argument("--table", default=TABLE, help="Output table in the ml schema.")
    return parser.parse_args(argv)


def simulation_scenarios() -> dict[str, tuple[int, int]]:
    """Scenario name -> count, for the scenarios in the simulator's starting state."""
    return {s.name: (s.balls, s.strikes) for s in options.SCENARIOS if options.in_simulation(s)}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ml = f"{args.catalog}.{args.ml_schema}"
    fact_pitch = f"`{args.catalog}`.`{args.gold_schema}`.fact_pitch"
    options_table = f"{ml}.{args.options_table}"

    import pandas as pd
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    # 1. The series and its options
    if args.opponent_id is None or args.series_start is None:
        latest = spark.sql(
            f"""
            SELECT opponent_id, series_start FROM {options_table}
            WHERE team_id = {args.team_id}
            ORDER BY scored_at DESC LIMIT 1
            """
        ).first()
        if latest is None:
            raise SystemExit(f"No scored series in {options_table}; run recommend_pitches")
        opponent_id = args.opponent_id or latest.opponent_id
        series_start = args.series_start or latest.series_start
    else:
        opponent_id, series_start = args.opponent_id, args.series_start
    series = (
        f"team_id = {args.team_id} AND opponent_id = {opponent_id}"
        f" AND series_start = DATE'{series_start}'"
    )
    scenarios = simulation_scenarios()
    scenario_list = ", ".join(f"'{s}'" for s in scenarios)
    p_columns = [f"p_{r}" for r in RESULTS]
    usual_columns = [f"usual_p_{r}" for r in RESULTS]
    rows = spark.sql(
        f"""
        SELECT pitcher_id, batter_id, scenario, pitch_type, zone, expected_run_value,
          {", ".join(p_columns + usual_columns)}
        FROM {options_table}
        WHERE {series} AND scenario IN ({scenario_list})
        """
    ).toPandas()
    rows.attrs = {}
    if rows.empty:
        raise SystemExit(f"No options for {series}")
    rows[["balls", "strikes"]] = pd.DataFrame(
        rows["scenario"].map(scenarios).tolist(), index=rows.index
    )
    print(f"Series {series}: {len(rows)} options")

    # 2. Run values in the starting state, and the check against the report's values
    register_run_values(spark, fact_pitch, args.run_value_seasons, "run_values_wide")
    rv = spark.sql(
        f"""
        SELECT * FROM run_values_wide
        WHERE outs = {options.SIMULATION_OUTS}
          AND base_state = '{options.SIMULATION_BASE_STATE}'
        """
    ).toPandas()
    run_values = {
        (int(r.balls), int(r.strikes)): np.array([getattr(r, f"rv_{x}") for x in RESULTS])
        for r in rv.itertuples()
    }
    recomputed = np.array(
        [
            rows.loc[i, p_columns].to_numpy(dtype=float) @ run_values[(b, s)]
            for i, b, s in zip(rows.index, rows["balls"], rows["strikes"], strict=True)
        ]
    )
    gap = float(np.max(np.abs(recomputed - rows["expected_run_value"].to_numpy())))
    print(f"Largest gap from the table's expected run value: {gap:.2e}")
    if gap > TOLERANCE:
        raise SystemExit("Run values don't match the ones the options were scored with")

    # 3. Simulate
    results = simulate.simulate_matchups(rows, run_values)
    info = spark.sql(
        f"""
        SELECT DISTINCT team_id, opponent_id, series_start, opponent_name, target_mode,
          lineup_source, miss_sd_ft, pitcher_id, pitcher_name, is_probable_starter,
          pitcher_limited_data,
          batter_id, batter_name, lineup_spot, pitcher_throws, batter_stands
        FROM {options_table}
        WHERE {series}
        """
    ).toPandas()
    info.attrs = {}
    out = info.merge(results, on=["pitcher_id", "batter_id"])
    out["first_pitch_zone"] = out["first_pitch_zone"].astype("Int64")
    out["outs"], out["base_state"] = options.SIMULATION_OUTS, options.SIMULATION_BASE_STATE

    # 4. Write, replacing this series
    table = f"{ml}.{args.table}"
    frame = spark.createDataFrame(out).selectExpr("*", "current_timestamp() AS simulated_at")
    writer = frame.write.format("delta")
    if spark.catalog.tableExists(table):
        writer = (
            writer.mode("overwrite").option("mergeSchema", "true").option("replaceWhere", series)
        )
    writer.saveAsTable(table)
    print(f"Wrote {len(out)} rows to {table}")

    wide = out.pivot_table(
        index=["pitcher_name", "is_probable_starter"],
        columns="policy",
        values="run_value",
        aggfunc="mean",
    )
    usual = out[out["policy"] == "usual"].groupby("pitcher_name")
    ranking = pd.DataFrame(
        {
            "runs_per_100_pa": 100 * wide["usual"].droplevel(1),
            "k_pct": 100 * usual["p_strikeout"].mean(),
            "bb_pct": 100 * usual["p_walk"].mean(),
            "pitches_per_pa": usual["pitches"].mean(),
            "recommended_gain": 100 * (wide["usual"] - wide["recommended"]).droplevel(1),
            "best_gain": 100 * (wide["usual"] - wide["best"]).droplevel(1),
        }
    ).sort_values("runs_per_100_pa")
    print("Pitchers vs the lineup (runs per 100 PA above average; gains per 100 PA):")
    print(ranking.round(2).to_string())


if __name__ == "__main__":
    main()
